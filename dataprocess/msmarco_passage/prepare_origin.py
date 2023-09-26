from argparse import ArgumentParser
import os
import os.path as osp
import csv
import pickle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import random


def prepare_clus(args, mapping):
    reverse_mapping = {v: k for k, v in mapping.items()}
    args.layers = eval(args.layers)
    for layer in args.layers:
        cluster_mapping = defaultdict(list)
        for new_id, ori_id in tqdm(reverse_mapping.items(), desc=f'initial cluster {layer}'):
            cluster_mapping[new_id[:layer]].append(str(ori_id))
        temp_mapping = defaultdict(list)
        for k in tqdm(cluster_mapping, desc=f'complement cluster {layer}'):
            for j in range(1, len(k) + 1):
                temp_mapping[k[:j]].extend(cluster_mapping[k])
                if j != len(k) and k[:j] in reverse_mapping:
                    temp_mapping[k].append(str(reverse_mapping[k[:j]]))
        for k in tqdm(temp_mapping, desc=f'deduplicate cluster {layer}'):
            assert len(set(temp_mapping[k])) == len(temp_mapping[k])
        with open(osp.join(args.output_dir, f'doc_cluster_layer{layer}.pkl'), 'wb') as fw:
            pickle.dump(dict(temp_mapping), fw)


def prepare_original_data(args):
    # build train
    queries = {}
    with open(osp.join(args.raw_dir, f'train.query.txt')) as fr:
        for line in tqdm(fr, desc=f'read train queries'):
            qid, query = line.rstrip('\n').split('\t')
            queries[qid] = query
    with open(osp.join(args.raw_dir, f'qrels.train.tsv'), 'r') as fr, \
            open(osp.join(args.origin_dir, f'train_mevi.tsv'), 'w') as fw:
        tsvreader = csv.reader(fr, delimiter="\t")
        for row in tqdm(tsvreader, desc=f'generate train data'):
            if len(row) == 4:
                [qid, _, posid, rel] = row
                assert rel == "1"
            else:
                [qid, posid] = row
            new_line = [queries[qid], posid]
            print('\t'.join(new_line), file=fw)

    # build dev
    queries = {}
    with open(osp.join(args.raw_dir, f'dev.query.txt')) as fr:
        for line in tqdm(fr, desc=f'read dev queries'):
            qid, query = line.rstrip('\n').split('\t')
            queries[qid] = query
    query_to_dids = defaultdict(list)
    with open(osp.join(args.raw_dir, f'qrels.dev.tsv'), 'r') as fr:
        tsvreader = csv.reader(fr, delimiter="\t")
        for row in tqdm(tsvreader, desc=f'combine dev data'):
            if len(row) == 4:
                [qid, _, posid, rel] = row
                assert rel == "1"
            else:
                [qid, posid] = row
            query_to_dids[queries[qid]].append(posid)
    with open(osp.join(args.origin_dir, f'dev_mevi_dedup.tsv'), 'w') as fw:
        for query, dids in tqdm(query_to_dids.items(), desc='generate dev data'):
            print(query, ','.join(dids), sep='\t', file=fw)

    # # for BM25 hard negatives
    # qrels = defaultdict(list)
    # with open(osp.join(args.raw_dir, 'qrels.train.tsv'), 'r') as fr:
    #     tsvreader = csv.reader(fr, delimiter="\t")
    #     for row in tqdm(tsvreader, desc=f'Read Qrels'):
    #         [qid, _, posid, rel] = row
    #         assert rel == "1"
    #         qrels[qid].append(posid)
    # qrels = dict(qrels)
    # queries = {}
    # with open(osp.join(args.raw_dir, 'train.query.txt')) as fr:
    #     for line in tqdm(fr, desc='Read Queries'):
    #         qid, query = line.rstrip('\n').split('\t')
    #         queries[qid] = query
    # with open(osp.join(args.raw_dir, f'train.negatives.tsv'), 'r') as fr, \
    #         open(osp.join(args.origin_dir, f'train_mevi_bm25neg.tsv'), 'w') as fwt, \
    #         open(osp.join(args.origin_dir, f'val_mevi_bm25neg.tsv'), 'w') as fwv:
    #     lines = fr.readlines()
    #     nlines = len(lines)
    #     ntrain = nlines - args.nval
    #     for i, line in enumerate(tqdm(lines, desc='Write Data')):
    #         qid, nn = line.strip().split('\t')
    #         if i < ntrain:
    #             fw = fwt
    #         else:
    #             fw = fwv
    #         print(queries[qid], ','.join(qrels[qid]), nn, sep='\t', file=fw)


def prepare_document_for_augmentation(args):
    with open(osp.join(args.raw_dir, 'corpus.tsv'), 'r') as fr, open(osp.join(args.origin_dir, 'doc_aug.tsv'), 'w') as fw:
        for line in tqdm(fr):
            did, title, content = line.rstrip('\n').split('\t')
            content = title.split(' ') + content.split(' ')
            add_num = max(0, len(content) - 3000) / 3000
            for _ in range(10 + int(add_num)):
                begin = random.randrange(0, len(content))
                # if begin >= (len(content)-64):
                #     begin = max(0, len(content)-64)
                end = begin + 64 if len(content) > begin + 64 else len(content)
                doc_aug = content[begin:end]
                doc_aug = ' '.join(doc_aug)
                print(f'{doc_aug}\t{did}', file=fw, flush=True)


def prepare_qg_data(args):
    qg_nums = args.qg_nums
    qg_intervals = [10 // q for q in qg_nums]
    outfiles = [
        open(osp.join(args.origin_dir, f'qg{q}.tsv'), 'w') for q in qg_nums]
    with open(osp.join(args.origin_dir, 'qg10.tsv'), 'r') as fr:
        for i, line in tqdm(enumerate(fr)):
            for inter, fw in zip(qg_intervals, outfiles):
                if i % inter == 0:
                    fw.write(line)


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--origin', action='store_true', default=False)
    parser.add_argument('--qg', action='store_true', default=False)
    parser.add_argument('--qg_nums', type=list, default=[1, 5])
    # parser.add_argument('--nval', type=int, default=500)
    parser.add_argument('--cluster', action='store_true', default=False)
    parser.add_argument('--mapping_path', type=str, default='old_newid.pkl')
    parser.add_argument('--output_path', type=str, default='processed')
    parser.add_argument('--layers', type=str, default='[4]')
    parser.add_argument('--doc_aug', action='store_true', default=False)

    args = parser.parse_args()

    assert osp.isdir(
        args.data_dir), f'Data directory {args.data_dir} not exists!'
    args.raw_dir = osp.join(args.data_dir, 'raw')
    args.origin_dir = osp.join(args.data_dir, 'origin')
    os.makedirs(args.origin_dir, exist_ok=True)

    if args.cluster:
        print('Reading old to new did mapping...')
        args.output_dir = osp.join(args.data_dir, args.output_path)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(osp.join(args.output_dir, args.mapping_path), 'rb') as fr:
            mapping = pickle.load(fr)
        # make hard negatives for cluster; only for MEVI-KMeans
        print('Preparing hard negatives of cluster...')
        prepare_clus(args, mapping)

    if args.origin:
        # build input
        print('Preparing original data...')
        prepare_original_data(args)

    if args.doc_aug:
        # build document augmentation; not in use
        print('Preparing document data for augmentation...')
        prepare_document_for_augmentation(args)

    if args.qg:
        # build qg1, qg5, qg10
        print('Preparing QG...')
        prepare_qg_data(args)


if __name__ == '__main__':
    main()
