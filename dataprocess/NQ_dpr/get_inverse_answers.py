# this scripts should be used after get_answers.py
import os.path as osp
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def inverse(args, phase):
    with open(osp.join(args.data_dir, f'{phase}_output.pkl'), 'rb') as fr:
        mapping = pickle.load(fr)
    inverse = defaultdict(list)
    for i, v in enumerate(tqdm(mapping)):
        for vv in v:
            inverse[vv].append(i)
    ndoc = 21015324
    offsets = np.zeros((ndoc + 1,), dtype=np.int32)
    for i in tqdm(range(ndoc)):
        offsets[i+1] = offsets[i] + len(inverse[i])
    offsets.tofile(osp.join(args.data_dir, f'{phase}_inverse_offsets.bin'))
    array = np.memmap(osp.join(
        args.data_dir, f'{phase}_inverse_array.bin'), mode='w+', dtype=np.int32, shape=(offsets[-1],))
    for i in tqdm(range(ndoc)):
        if len(inverse[i]) > 0:
            array[offsets[i]:offsets[i+1]] = inverse[i]
            array.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    if args.dev:
        inverse(args, 'dev')
    if args.test:
        inverse(args, 'test')
