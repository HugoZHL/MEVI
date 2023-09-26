from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp


def _tokenize_once(batch_query, batch_did, omem, query_tokenizer, truncate, indices):
    query_encoded = query_tokenizer.batch_encode_plus(
        batch_query,
        max_length=truncate,
        truncation=True,
        padding='max_length',
        return_tensors='np',
    )
    omem[indices, :truncate] = query_encoded['input_ids'].astype(
        np.int32)
    omem[indices, truncate:truncate *
         2] = query_encoded['attention_mask'].astype(np.int32)
    omem[indices, -1] = np.array(batch_did, dtype=np.int32)
    omem.flush()


def _tokenize_all(input_file, output_mem, args, offset=0, cur_nquery=None, iscorpus=False):
    query_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name)
    batch_query = []
    batch_did = []
    start = 0
    with open(input_file, 'r') as fr:
        while offset > 0:
            _ = fr.readline()
            offset -= 1
        for _ in tqdm(range(cur_nquery)):
            line = fr.readline()
            if iscorpus:
                did, title, content = line.strip('\n').split('\t')
                query = 'Title: ' + title + ' Text: ' + content
            else:
                query, did = line.strip('\n').split('\t')
            batch_query.append(query)
            batch_did.append(did)
            if len(batch_query) == args.batch_size:
                ending = start + args.batch_size
                indices = np.arange(start, ending)
                _tokenize_once(batch_query, batch_did, output_mem,
                               query_tokenizer, args.truncate, indices)
                start = ending
                batch_query = []
                batch_did = []

    if len(batch_query) > 0:
        ending = start + len(batch_query)
        indices = np.arange(start, ending)
        _tokenize_once(batch_query, batch_did, output_mem,
                       query_tokenizer, args.truncate, indices)


def parallel_tokenize(rank, args, nquery, fname, iscorpus=False):
    input_file = osp.join(args.output_dir, fname)
    output_file = osp.join(
        args.output_dir, fname.split('.')[0] + f'_tokenized{args.truncate}_{rank}.bin')
    cur_nquery = nquery // args.nrank
    offset = cur_nquery * rank
    if rank == args.nrank - 1:
        cur_nquery += (nquery % args.nrank)
    output_mem = np.memmap(output_file, mode='w+',
                           dtype=np.int32, shape=(cur_nquery, args.truncate * 2 + 1))
    _tokenize_all(input_file, output_mem, args, offset=offset,
                  cur_nquery=cur_nquery, iscorpus=iscorpus)


def start_tokenize(args, nquery, fname, iscorpus=False):
    mp.spawn(parallel_tokenize,
             args=(args, nquery, fname, iscorpus),
             nprocs=args.nrank,
             join=True)
    output_file = osp.join(
        args.output_dir, fname.split('.')[0] + f'_tokenized{args.truncate}.bin')
    output_mem = np.memmap(output_file, mode='w+',
                           dtype=np.int32, shape=(nquery, args.truncate * 2 + 1))
    offset = 0
    for i in range(args.nrank):
        input_file = osp.join(
            args.output_dir, fname.split('.')[0] + f'_tokenized{args.truncate}_{i}.bin')
        input_mem = np.memmap(input_file, mode='r',
                              dtype=np.int32).reshape(-1, args.truncate * 2 + 1)
        new_offset = offset + input_mem.shape[0]
        output_mem[offset:new_offset] = input_mem
        output_mem.flush()
        offset = new_offset
    assert offset == nquery


def tokenize_query(args):
    start_tokenize(args, 498816, 'train_mevi.tsv')


def tokenize_corpus(args):
    start_tokenize(args, 21015324, 'corpus.tsv', iscorpus=True)


def tokenize_qg(args):
    start_tokenize(args, 210153240, f'qg{args.num_qg}.tsv')


def main():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str,
                        default='t5-base')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--truncate', type=int, default=64)
    parser.add_argument('--num_qg', type=int, default=10)
    parser.add_argument('--nrank', type=int, default=40)
    parser.add_argument('--tok_train', type=int, default=0)
    parser.add_argument('--tok_corpus', type=int, default=0)
    parser.add_argument('--tok_qg', type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tok_train:
        tokenize_query(args)
    if args.tok_corpus:
        tokenize_corpus(args)
    if args.tok_qg:
        tokenize_qg(args)


if __name__ == '__main__':
    main()
