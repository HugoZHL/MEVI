import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
import pickle


def sample(args):
    def join(fpath):
        return osp.join(args.dir_path, fpath)

    # get out
    num_documents = 8841823
    out_docs = np.random.choice(np.arange(num_documents), int(
        num_documents * args.drop_rate), replace=False)
    out_docs = set(out_docs.tolist())

    # read files and make statistics
    trains = set()
    devs = set()
    with open(join('train_mevi.tsv'), 'r') as fr:
        for line in fr:
            _, did = line.rstrip('\n').split('\t')
            trains.add(int(did))
    with open(join('dev_mevi_dedup.tsv'), 'r') as fr:
        for line in fr:
            _, dids = line.rstrip('\n').split('\t')
            for did in dids.split(','):
                devs.add(int(did))
    inter_train_dev = trains.intersection(devs)
    inter_train_out = trains.intersection(out_docs)
    inter_dev_out = devs.intersection(out_docs)
    print(
        f'Num trains: {len(trains)}; num devs: {len(devs)}; num outs: {len(out_docs)}')
    print(
        f'Num out docs in train: {len(inter_train_out)}; num out docs in dev: {len(inter_dev_out)}')

    # get mapping
    full_to_sampled_mapping = np.full(
        (num_documents,), fill_value=-1, dtype=np.int32)
    target = 0
    for i in tqdm(range(num_documents), desc='Get Mapping'):
        if i not in out_docs:
            full_to_sampled_mapping[i] = target
            target += 1
    full_to_sampled_mapping.tofile(join(f'drop{args.drop_rate}mapping.bin'))

    # process data
    with open(join('train_mevi.tsv'), 'r') as fr, open(join(f'train_mevi_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process Train'):
            query, did = line.rstrip('\n').split('\t')
            did = int(did)
            newdid = full_to_sampled_mapping[did]
            if newdid != -1:
                print(f'{query}\t{newdid}', file=fw)
    with open(join('dev_mevi_dedup.tsv'), 'r') as fr, open(join(f'dev_mevi_dedup_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process Dev'):
            query, dids = line.rstrip('\n').split('\t')
            newdids = []
            for did in dids.split(','):
                newdid = full_to_sampled_mapping[int(did)]
                if newdid != -1:
                    newdids.append(str(newdid))
            if len(newdids) > 0:
                newdids = ','.join(newdids)
                print(f'{query}\t{newdids}', file=fw)
    with open(join('../raw/corpus.tsv'), 'r') as fr, open(join(f'corpus_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process Corpus'):
            did, title, content = line.rstrip('\n').split('\t')
            newdid = full_to_sampled_mapping[int(did)]
            if newdid != -1:
                print(f'{newdid}\t{title}\t{content}', file=fw)
    with open(join('qg10.tsv'), 'r') as fr, open(join(f'qg10_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process QG10'):
            query, did = line.rstrip('\n').split('\t')
            newdid = full_to_sampled_mapping[int(did)]
            if newdid != -1:
                print(f'{query}\t{newdid}', file=fw)
    co_negs = {}
    with open(join('mevi_bm25neg.tsv'), 'r') as fr, open(join(f'mevi_bm25neg_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process BM25'):
            query, dids, negs = line.rstrip('\n').split('\t')
            newdids = []
            for did in dids.split(','):
                newdid = full_to_sampled_mapping[int(did)]
                if newdid != -1:
                    newdids.append(str(newdid))
            if len(newdids) > 0:
                newnegs = []
                for neg in negs.split(','):
                    newneg = full_to_sampled_mapping[int(neg)]
                    newnegs.append(newneg)
                if len(newnegs) > 0:
                    newdids = ','.join(newdids)
                    co_negs[query] = newnegs
                    newnegs = ','.join([str(newneg) for newneg in newnegs])
                    print(f'{query}\t{newdids}\t{newnegs}', file=fw)
    with open(join(f'mevi_bm25neg_drop{args.drop_rate}.pkl'), 'wb') as fw:
        pickle.dump(co_negs, fw)
    with open(join('../ance/old_newid.pkl'), 'rb') as fr, open(join(f'../ance/old_newid_drop{args.drop_rate}.pkl'), 'wb') as fw:
        ori = pickle.load(fr)
        new = {}
        for k, v in ori.items():
            nk = full_to_sampled_mapping[k]
            if nk != -1:
                new[nk] = v
        pickle.dump(new, fw)

    # for binary files and numpy files
    reserve_mask = (full_to_sampled_mapping != -1)
    docemb = np.fromfile(join('newdocembance.bin'),
                         dtype=np.float32).reshape(num_documents, 768)
    docemb = docemb[reserve_mask]
    docemb.tofile(join(f'newdocembance_drop{args.drop_rate}.bin'))
    del docemb
    tokens = np.fromfile(join('../ance/all_document_tokens.bin'),
                         dtype=np.int64).reshape(num_documents, 128)
    tokens = tokens[reserve_mask]
    tokens.tofile(
        join(f'../ance/all_document_tokens_drop{args.drop_rate}.bin'))
    del tokens
    masks = np.fromfile(join('../ance/all_document_masks.bin'),
                        dtype=np.int64).reshape(num_documents, 128)
    masks = masks[reserve_mask]
    masks.tofile(join(f'../ance/all_document_masks_drop{args.drop_rate}.bin'))
    del masks
    del reserve_mask

    # other files
    with open(join('train_mevi_dedup.tsv'), 'r') as fr, open(join(f'train_mevi_dedup_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process Train Dedup'):
            query, dids = line.rstrip('\n').split('\t')
            newdids = []
            for did in dids.split(','):
                newdid = full_to_sampled_mapping[int(did)]
                if newdid != -1:
                    newdids.append(str(newdid))
            if len(newdids) > 0:
                newdids = ','.join(newdids)
                print(f'{query}\t{newdids}', file=fw)
    with open(join('qg1.tsv'), 'r') as fr, open(join(f'qg1_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process QG1'):
            query, did = line.rstrip('\n').split('\t')
            newdid = full_to_sampled_mapping[int(did)]
            if newdid != -1:
                print(f'{query}\t{newdid}', file=fw)
    with open(join('qg5.tsv'), 'r') as fr, open(join(f'qg5_drop{args.drop_rate}.tsv'), 'w') as fw:
        for line in tqdm(fr, desc='Process QG5'):
            query, did = line.rstrip('\n').split('\t')
            newdid = full_to_sampled_mapping[int(did)]
            if newdid != -1:
                print(f'{query}\t{newdid}', file=fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, required=True)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    np.random.seed(args.seed)
    sample(args)
