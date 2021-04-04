import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GaussianNoise, GaussianDropout, Conv1D, multiply
from tensorflow.keras.models import Model
import tensorflow.keras.backend as backend
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize
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


def eval_encoder(training=False):
    load_file = '../data_process/tcga_pfi.h5' if opt.pfi else '../data_process/tcga.h5'
    m_rna, label, gene, sample_id = data_provider.load_h5_all(load_file, True)
    if not opt.pfi:
        m_rna_o, label_o, gene_o, sample_id_o = data_provider.load_h5_all('../data_process/other.h5', True)
        m_rna, label, sample_id = np.concatenate((m_rna, m_rna_o)), np.concatenate((label, label_o)), np.concatenate((sample_id, sample_id_o))
    m_rna = normalize(X=m_rna, axis=0, norm="max")
    print('Data size:', m_rna.shape, label)

    if not opt.use_all:
        # according to our indexes
        top_idx_doctor = np.array([471, 1213, 1632, 1635, 1636, 2743, 2774, 3020, 4880, 7057,
                                   7146, 7213, 8282, 9619, 9899, 9914, 10079, 10319, 10479, 11629,
                                   12569, 13075, 13343, 13815, 15103, 15481, 15716, 17130, ])

        if opt.random_input:
            np.random.seed(opt.seed)
            top_idx = np.arange(m_rna.shape[1])
        else: # modify the following filenames accordingly
            if opt.top_type == 0:
                if opt.pfi:
                    with open('../shap_log_pfi.txt') as f:
                        lines = f.readlines()
                        top_idx = np.array(lines[2].split(' ')).astype(int)[-10:]
                else:
                    with open('../shap_log.txt') as f:
                        lines = f.readlines()
                        top_idx = np.array(lines[2].split(' ')).astype(int)[-10:]
            elif opt.top_type == 1:
                top_idx = top_idx_doctor
        np.random.seed(opt.seed)
        np.random.shuffle(top_idx)
        top_idx = top_idx[:opt.top_k]
        print(top_idx)
        m_rna = m_rna[:, top_idx]
    """first random: train and test sets"""
    indices = np.arange(m_rna.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    m_rna2 = m_rna[indices]
    label2 = label[indices]
    sample_id = sample_id[indices]
    categorical_label = np_utils.to_categorical(label2)
    """to save the data split strategy for other analysis"""
    # tofile = np.stack((sample_id.astype(str), label2.astype(str)), axis=1)
    # np.savetxt(X=tofile, fname=MODEL_DIR + "/sample_id.txt", delimiter=",", fmt='%s')

    m_rna_train = m_rna2[:-opt.test_size, ]
    m_rna_test = m_rna2[-opt.test_size:, ]
    categorical_label_train = categorical_label[:-opt.test_size, ]
    categorical_label_test = categorical_label[-opt.test_size:, ]
    label_train = label2[:-opt.test_size, ]
    label_test = label2[-opt.test_size:, ]
    sample_id_test = sample_id[-opt.test_size:, ]

    """pr sample operations"""
    pr_idx_train = np.array([i for i, e in enumerate(label_train) if e == 2 or e == 3])
    pr_idx_test = np.array([i for i, e in enumerate(label_test) if e == 2 or e == 3])
    pr_m_rna_train, pr_m_rna_test = m_rna_train[pr_idx_train], m_rna_test[pr_idx_test]
    pr_label_train, pr_label_test = label_train[pr_idx_train] - 2, label_test[pr_idx_test] - 2

    print('pr samples in training set:', len(pr_idx_train))
    print('pr samples in testing set:', len(pr_idx_test))
    print('healthy samples in training set:', sum([1 for x in pr_label_train if x == 1]))
    print('healthy samples in testing set:', sum([1 for x in pr_label_test if x == 1]))
    print('PR train and test size:', pr_m_rna_train.shape, pr_m_rna_test.shape)
    pr_categorical_label_train = np_utils.to_categorical(pr_label_train)
    pr_categorical_label_test = np_utils.to_categorical(pr_label_test)

    print("data loading has just been finished")

    def create_model():
        inputs = Input(shape=(m_rna.shape[1],), name="inputs")
        inputs_1 = Dense(8, activation="relu", name="inputs_1")(inputs)
        encoded = Dense(4, activation='relu', name='encoded')(inputs_1)
        cl_0 = Dense(units=pr_categorical_label_train.shape[1], activation="softmax", name="category")(encoded)
        m = Model(inputs=inputs, outputs=[cl_0])

        m.compile(optimizer='adam',
                     loss=[tf.keras.losses.CategoricalCrossentropy()],     # "cosine_similarity"],
                     loss_weights=[1],     # , 0.5
                     metrics={"category": "acc"})     # , "cl_disease": "acc"
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
        model.fit(pr_m_rna_train, pr_categorical_label_train, batch_size=opt.batch_size,
                  epochs=opt.max_epoch,
                  callbacks=[tensorboard_callback],
                  validation_data=(pr_m_rna_test, pr_categorical_label_test),
                  verbose=2)
        model.save_weights(filepath=checkpoint_path)
        print("fitting has just been finished")

    else:
        model.load_weights(checkpoint_path)
        m_rna_test = pr_m_rna_test
        categorical_label_test = pr_categorical_label_test
        data_pred = model.predict(m_rna_test, batch_size=opt.batch_size, verbose=2)
        np.savetxt(X=m_rna_test, fname=MODEL_DIR + "/test_gene.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=label, fname=MODEL_DIR + "/label.csv", delimiter=",", fmt='%1.3f')
        np.savetxt(X=data_pred, fname=MODEL_DIR + "/pred_label.csv", delimiter=",", fmt='%1.3f')
        """ argmax """
        y_pred = np.argmax(data_pred, axis=1)
        y_gt = np.argmax(categorical_label_test, axis=1)
        confusion_0 = sk.confusion_matrix(y_gt, y_pred, labels=[0, 1])
        print(confusion_0)
        balanced_acc = sk.balanced_accuracy_score(y_gt, y_pred)
        """ logits for roc """
        pred_logit = data_pred[:, 0]
        gt_logit = categorical_label_test[:, 0]
        """ log """
        log1 = open(os.path.join(MODEL_DIR, 'log_blacc.txt'), 'a')
        log2 = open(os.path.join(MODEL_DIR, 'log_roc.txt'), 'a')

        roc_feat = ''
        """Only execute when using PR data (two labels)"""
        auc = sk.roc_auc_score(gt_logit, pred_logit)
        fpr, tpr, thresh = sk.roc_curve(gt_logit, pred_logit)
        roc_feat = {'auc': auc, 'fpr': [], 'tpr': []}
        for e in fpr:
            roc_feat['fpr'].append(str(e))
        for e in tpr:
            roc_feat['tpr'].append(str(e))

        def log_string(out1, out2):
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

        if opt.use_argmax:
            if opt.use_all:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), 36)))
                log_string(confusion_1, None)
            else:
                confusion_1 = ' '.join(list(np.reshape(confusion_0.astype(str), 4)))
                log_string(confusion_1, roc_feat)
        else:
            log_string(balanced_acc, roc_feat)


if __name__ == '__main__':
    if opt.phase == 'train':
        if not os.path.exists(os.path.join(MODEL_DIR, 'code/')):
            os.makedirs(os.path.join(MODEL_DIR, 'code/'))
            os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model def
        eval_encoder(True)
    elif opt.phase == 'test':
        eval_encoder(False)
