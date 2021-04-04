import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GaussianNoise, GaussianDropout, Conv1D, multiply
from tensorflow.keras.models import Model
import tensorflow.keras.backend as backend
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize, RobustScaler
import sklearn.metrics as sk
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import CSVLogger, History
import data_provider
import os
from options import opt, MODEL_DIR
import time
import shap
import pandas as pd


def mlc_ae(training=False):
    load_file = '../data_process/tcga_pfi.h5' if opt.pfi else '../data_process/tcga.h5'
    m_rna, label, gene, sample_id = data_provider.load_h5_all(load_file, True)
    if not opt.pfi:
        m_rna_o, label_o, gene_o, sample_id_o = data_provider.load_h5_all('../data_process/other.h5', True)
        m_rna, label, sample_id = np.concatenate((m_rna, m_rna_o)), np.concatenate((label, label_o)), \
                                  np.concatenate((sample_id, sample_id_o))
    m_rna = normalize(X=m_rna, axis=0, norm="max")
    print('feat and label size', m_rna.shape, label)
    """first random: train and test sets"""
    indices = np.arange(m_rna.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    m_rna2 = m_rna[indices]
    label2 = label[indices]
    sample_id = sample_id[indices]
    if opt.use_all:
        categorical_label = np_utils.to_categorical(label2, num_classes=6)
    else:
        categorical_label = np_utils.to_categorical(label2, num_classes=2)
    """to save the data split strategy for other analysis"""
    # tofile = np.stack((sample_id.astype(str), label2.astype(str)), axis=1)
    # np.savetxt(X=tofile, fname=MODEL_DIR + "/sample_id.txt", delimiter=",", fmt='%s')

    m_rna_train = m_rna2[:-opt.test_size, ]
    m_rna_test = m_rna2[-opt.test_size:, ]
    categorical_label_train = categorical_label[:-opt.test_size, ]
    categorical_label_test = categorical_label[-opt.test_size:, ]
    label_train = label2[:-opt.test_size, ]
    label_test = label2[-opt.test_size:, ]

    # pr sample operations
    pr_idx_train = np.array([i for i, e in enumerate(label_train) if e == 2 or e == 3])
    pr_idx_test = np.array([i for i, e in enumerate(label_test) if e == 2 or e == 3])
    print('healthy samples in training set:', sum([1 for x in label_train if x == 3]))
    print('healthy samples in testing set:', sum([1 for x in label_test if x == 3]))

    pr_m_rna_train = m_rna_train[pr_idx_train]
    pr_m_rna_test = m_rna_test[pr_idx_test]
    pr_label_train = label_train[pr_idx_train] - 2
    pr_label_test = label_test[pr_idx_test] - 2

    print('PR train and test size:', pr_label_train.shape, pr_idx_test.shape)
    pr_categorical_label_train = np_utils.to_categorical(pr_label_train)
    pr_categorical_label_test = np_utils.to_categorical(pr_label_test)
    print("data loading has just been finished")

    def create_model():
        inputs = Input(shape=(m_rna.shape[1],), name="inputs")
        inputs_0 = BatchNormalization(name="inputs_0")(inputs)
        inputs_1 = Dense(1024, activation="relu", name="inputs_1")(inputs_0)
        inputs_2 = BatchNormalization(name="inputs_2")(inputs_1)
        inputs_3 = Dense(256, activation="relu", name="inputs_3")(inputs_2)
        inputs_4 = BatchNormalization(name="inputs_4")(inputs_3)
        encoded = Dense(units=12, activation='relu', name='encoded')(inputs_4)
        inputs_5 = Dense(512, activation="relu", name="inputs_5")(encoded)
        decoded_tcga = Dense(units=m_rna.shape[1], activation='linear', name="m_rna")(inputs_5)
        if opt.use_all:
            cl_0 = Dense(units=categorical_label_train.shape[1], activation="softmax", name="category")(encoded)
        else:
            cl_0 = Dense(units=pr_categorical_label_train.shape[1], activation="softmax", name="category")(encoded)
        m = Model(inputs=inputs, outputs=[decoded_tcga, cl_0])
        m.compile(optimizer='adam',
                     loss=["mse", "cosine_similarity"],     # "cosine_similarity"],
                     loss_weights=[0.001, 0.5],     # , 0.5
                     metrics={"m_rna": ["mae", "mse"], "category": "acc"})     # , "cl_disease": "acc"

        return m

    model = create_model()
    checkpoint_path = os.path.join(MODEL_DIR, 'my_model.h5')
    # model.summary()
    if training:
        # file_writer = tf.summary.create_file_writer(MODEL_DIR + "/metrics")
        # file_writer.set_as_default()
        def lr_scheduler(epoch):
            lr = 0.01
            if epoch < 200:
                lr *= 0.99999999
            else:
                lr *= 0.99999
            tf.summary.scalar('Learning Rate', data=lr, step=epoch)
            return lr

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR)
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        if opt.use_all:
            model.fit(m_rna_train, [m_rna_train, categorical_label_train], batch_size=opt.batch_size,
                      epochs=opt.max_epoch,
                      callbacks=[tensorboard_callback],
                      validation_data=(m_rna_test, [m_rna_test, categorical_label_test]),
                      verbose=2)
        else:
            model.fit(pr_m_rna_train, [pr_m_rna_train, pr_categorical_label_train], batch_size=opt.batch_size,
                      epochs=opt.max_epoch,
                      callbacks=[tensorboard_callback],
                      validation_data=(pr_m_rna_test, [pr_m_rna_test, pr_categorical_label_test]),
                      verbose=2)
        model.save_weights(filepath=checkpoint_path)
        print("fitting has just been finished")

    else:
        model.load_weights(checkpoint_path)
        if not opt.use_all:
            m_rna_test = pr_m_rna_test
            categorical_label_test = pr_categorical_label_test
        data_pred = model.predict(m_rna_test, batch_size=opt.batch_size, verbose=2)

        """ argmax """
        y_pred = np.argmax(data_pred[1], axis=1)
        y_gt = label_test # np.argmax(categorical_label_test, axis=1)
        if opt.use_all:
            confusion_0 = sk.confusion_matrix(y_gt, y_pred, labels=[0, 1, 2, 3, 4, 5])
        else:
            confusion_0 = sk.confusion_matrix(y_gt, y_pred, labels=[0, 1])
        print(confusion_0)
        balanced_acc = sk.balanced_accuracy_score(y_gt, y_pred)
        acc = sk.accuracy_score(y_gt, y_pred)
        """ logits for roc """
        y_logit = data_pred[1][:, 0]
        x_logit = categorical_label_test[:, 0]
        """save for records"""
        np.savetxt(X=m_rna_test, fname=MODEL_DIR + "/test_gene.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=label, fname=MODEL_DIR + "/label.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=data_pred[0], fname=MODEL_DIR + "/pred_gene.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=y_pred, fname=MODEL_DIR + "/pred_label.csv", delimiter=",", fmt='%1.3f')
        """ get latent representation """
        layer_name = "encoded"
        encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        encoded_output = encoded_layer_model.predict(m_rna_test)
        np.savetxt(X=encoded_output, fname=MODEL_DIR + "/latent_feat.csv", delimiter=",")

        """ log """
        log1 = open(os.path.join(MODEL_DIR, 'log_blacc.txt'), 'a')
        log2 = open(os.path.join(MODEL_DIR, 'log_roc.txt'), 'a')
        log3 = open(os.path.join(MODEL_DIR, 'log_acc.txt'), 'a')

        if not opt.use_all:
            """Only execute when using PR data (two labels)"""
            auc = sk.roc_auc_score(x_logit, y_logit)
            fpr, tpr, thresh = sk.roc_curve(x_logit, y_logit)
            roc_feat = {'auc': auc, 'fpr': [], 'tpr': []}
            for e in fpr:
                roc_feat['fpr'].append(str(e))
            for e in tpr:
                roc_feat['tpr'].append(str(e))

        if opt.use_shap:
            """ Depth Explainer """
            print("Processing SHAP...")
            model.load_weights(checkpoint_path)
            from matplotlib import colors as plt_colors
            layer_name = "category"
            encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            input_feat = m_rna_train if opt.sample_all == 'all' else pr_m_rna_train
            e = shap.GradientExplainer(encoded_layer_model, input_feat)
            # e = shap.DeepExplainer(encoded_layer_model, input_feat)
            shap_values = e.shap_values(input_feat)
            shap_load_gene = '../data_process/gene_pfi.csv' if opt.pfi else '../data_process/gene.csv'
            feat_name = np.loadtxt(shap_load_gene, dtype=str, delimiter=",")   #[:, 3]
            class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
            print('class_inds', class_inds)
            colors = np.array(['yellowgreen', 'palevioletred', 'lightcoral', 'mediumpurple', 'cornflowerblue',
                               'orange'])[class_inds]
            cmap = plt_colors.ListedColormap(colors)
            shap.summary_plot(shap_values, input_feat, feature_names=feat_name, max_display=40,
                              plot_size=(12.0, 16.0, 2.0), plot_type='bar',
                              color=cmap, show=True, sort=True,
                              class_names=['Ovarian (T)', 'Ovarian (N)', 'Prostate (T)', 'Prostate (N)', 'Breast (T)',
                                           'Breast (N)'])

        def log_string(out1, out2, out3):
            log1.write(str(out1))
            log1.write('\n')
            log1.flush()
            print(out1)
            if out2:
                log2.write(str(out2['auc']))
                log2.write('\n')
                roc_x, roc_y = ' '.join(out2['fpr']), ' '.join(out2['tpr'])
                log2.write(roc_x)
                log2.write('\n')
                log2.write(roc_y)
                log2.write('\n')
                log2.flush()
                print(out2)
            if out3:
                log3.write(str(out3))
                log3.write('\n')
                log3.flush()

        if opt.use_argmax:
            if opt.use_all:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), 36)))
                log_string(confusion_1, None, None)
            else:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), 4)))
                log_string(confusion_1, roc_feat, acc)
        else:
            log_string(balanced_acc, None, acc)


if __name__ == '__main__':
    if opt.phase == 'train':
        if not os.path.exists(os.path.join(MODEL_DIR, 'code/')):
            os.makedirs(os.path.join(MODEL_DIR, 'code/'))
            os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model def
        mlc_ae(True)
    elif opt.phase == 'test':
        mlc_ae(False)
