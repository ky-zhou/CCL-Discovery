import numpy as np
import os
# import tensorflow as tf
import h5py


def load_h5_all(file, is_training):
    hf = h5py.File(file, 'r+')
    label = hf['label'][:][:]
    num_samples = len(label)
    train_size = num_samples - test_size
    feat = hf['feature'][:][:, :]
    gene = hf['gene_name'][:]
    sample = hf['sample'][:]
    print('%s has data:', feat.shape)
    # train_dataset = tf.data.Dataset.from_tensor_slices((feat[:train_size, :], label[:train_size]))     #not using now
    # test_dataset = tf.data.Dataset.from_tensor_slices((feat[-test_size:], label[-test_size:]))     #not using now
    # train_dataset = tf.data.Dataset.from_generator((feat0[:train_size, :], label[:train_size]))
    # test_dataset = tf.data.Dataset.from_generator((feat0[-test_size:], label[-test_size:]))

    return feat, label, gene, sample



if __name__ == '__main__':
    m_rna, label, gene, sample_id = load_h5_all('../data_process/tcga.h5', True)

