from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import os.path as osp
import numpy as np
import pickle
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm


def parallel_tokenize_passage(args):
    mp.spawn(tokenize_passage,
             args=(args,),
             nprocs=args.nrank,
             join=True)
    ndoc = 0
    for i in range(args.nrank):
        cur_token_mem = np.memmap(osp.join(
            args.output_dir, f'all_document{args.suffix}_tokens_{i}.bin'), mode='r', dtype=np.int64).reshape(-1, args.passage_truncate)
        ndoc += cur_token_mem.shape[0]
    print(f'Please check: dataset NQ_dpr has {ndoc} documents.')
    token_mem = np.memmap(osp.join(
        args.output_dir, f'all_document{args.suffix}_tokens.bin'), mode='w+', dtype=np.int64, shape=(ndoc, args.passage_truncate))
    mask_mem = np.memmap(osp.join(
        args.output_dir, f'all_document{args.suffix}_masks.bin'), mode='w+', dtype=np.int64, shape=(ndoc, args.passage_truncate))
    offset = 0
    all_indices = []
    for i in range(args.nrank):
        cur_token_mem = np.memmap(osp.join(
            args.output_dir, f'all_document{args.suffix}_tokens_{i}.bin'), mode='r', dtype=np.int64).reshape(-1, args.passage_truncate)
        cur_mask_mem = np.memmap(osp.join(
            args.output_dir, f'all_document{args.suffix}_masks_{i}.bin'), mode='r', dtype=np.int64).reshape(-1, args.passage_truncate)
        new_offset = offset + cur_token_mem.shape[0]
        token_mem[offset:new_offset] = cur_token_mem
        mask_mem[offset:new_offset] = cur_mask_mem
        token_mem.flush()
        mask_mem.flush()
        with open(osp.join(args.output_dir, f'all_document{args.suffix}_indices_{i}.pkl'), 'rb') as fr:
            cur_indices = pickle.load(fr)
        all_indices.extend(cur_indices)
        offset = new_offset
    all_indices = {v: i for i, v in enumerate(all_indices)}
    with open(osp.join(args.output_dir, f'all_document{args.suffix}_indices.pkl'), 'wb') as fw:
        pickle.dump(all_indices, fw)
    assert offset == ndoc


def tokenize_passage(rank, args):
    tok_name = 'bert-base-uncased'
    kwargs = {'do_lower_case': True}
    passage_tokenizer = AutoTokenizer.from_pretrained(tok_name, **kwargs)
    idkey = 'input_ids'
    makey = 'attention_mask'
    index = []
    ndoc = 21015324
    cur_ndoc = ndoc // args.nrank
    offset = cur_ndoc * rank
    if rank == args.nrank - 1:
        cur_ndoc += (ndoc % args.nrank)
    tokens = np.memmap(osp.join(args.output_dir, f'all_document{args.suffix}_tokens_{rank}.bin'),
                       dtype=np.int64, mode='w+', shape=(cur_ndoc, args.passage_truncate))
    masks = np.memmap(osp.join(args.output_dir, f'all_document{args.suffix}_masks_{rank}.bin'),
                      dtype=np.int64, mode='w+', shape=(cur_ndoc, args.passage_truncate))
    with open(osp.join(args.document_path), 'r') as fr:
        for _ in range(offset):
            _ = fr.readline()
        for i in tqdm(range(cur_ndoc)):
            line = fr.readline()
            key, title, content = line.rstrip('\n').split('\t')
            index.append(key)
            passage_encoded = passage_tokenizer.encode_plus(
                title,
                text_pair=content.strip(),
                add_special_tokens=True,
                max_length=args.passage_truncate,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=None,
                return_tensors='np',
            )
            tokens[i, :] = passage_encoded[idkey]
            masks[i, :] = passage_encoded[makey]
    tokens.flush()
    masks.flush()
    # index = {v: i for i, v in enumerate(index)}
    with open(osp.join(args.output_dir, f'all_document{args.suffix}_indices_{rank}.pkl'), 'wb') as fw:
        pickle.dump(index, fw)


def main():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--document_path', type=str, default=None)
    parser.add_argument('--passage_truncate', type=int, default=128)
    parser.add_argument('--nrank', type=int, default=20)

    args = parser.parse_args()

    assert args.document_path
    args.suffix = str(
        args.passage_truncate) if args.passage_truncate != 128 else ''
    os.makedirs(args.output_dir, exist_ok=True)
    print('Tokenizing passages...')
    parallel_tokenize_passage(args)


if __name__ == '__main__':
    main()
