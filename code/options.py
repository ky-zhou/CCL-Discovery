import argparse
import os
import numpy as np

BASE_DIR = os.getcwd()
Debug_Index = '01'
MODEL_DIR = os.path.join('../model', Debug_Index)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test or gen [default: train]')
parser.add_argument('--max_epoch', type=int, default=600, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=256, help='train:70, test:32')
parser.add_argument('--test_size', type=int, default=100, help='0')
parser.add_argument('--top_k', type=int, default=10, help='how many top features are used for validation')
parser.add_argument('--top_type', type=int, default=0, help='0: shap, 1: exp')
parser.add_argument('--seed', type=int, default=4, help='0')
parser.add_argument('--sample_all', type=str, default='pr', help='shap for all or pr samples')
parser.add_argument('--pfi', type=bool, default=False, help='label is pfi or not')
parser.add_argument('--use_shap', type=bool, default=False, help='get shap or not')
parser.add_argument('--use_all', type=bool, default=True, help='use all or top genes to train')
parser.add_argument('--random_input', type=bool, default=False, help='T or F')
parser.add_argument('--use_argmax', type=bool, default=False, help='used when testing')
opt = parser.parse_args()

