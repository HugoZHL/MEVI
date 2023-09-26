import os.path as osp
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm


def load(f):
    with open(f, 'rb') as fr:
        results = pickle.load(fr)
    return results


def dump(obj, f):
    with open(f, 'wb') as fw:
        pickle.dump(obj, fw)


def compute_scores(a, b):
    result = - ((a - b) ** 2)
    return torch.sum(result, dim=-1)


def gen_sampled_to_full(args):
    def join(fpath):
        return osp.join(args.dir_path, fpath)
    full_to_sampled_mapping = np.fromfile(
        join(f'drop{args.drop_rate}mapping.bin'), dtype=np.int32)
    middle = f'{args.subvector_num}_{args.subvector_bit}_drop{args.drop_rate}'
    codebook = torch.load(
        join(f'rqcodebook{middle}.pt'), map_location='cpu')
    cluster = load(join(f'rqclus{middle}.pkl'))
    mapping = load(join(f'rqmapping{middle}.pkl'))

    # get reverse mapping
    sampled_to_full_mapping = {}
    rest = (full_to_sampled_mapping == -1).nonzero()[0]
    print('rest shape', rest.shape)
    contain = (full_to_sampled_mapping != -1).nonzero()[0]
    print('contain shape', contain.shape)
    sampled_to_full_mapping = {full_to_sampled_mapping[i]: i for i in contain}
    print('reverse length', len(sampled_to_full_mapping))
    print(len(sampled_to_full_mapping), len(full_to_sampled_mapping),
          len(full_to_sampled_mapping) * (1 - float(args.drop_rate)))

    dump(sampled_to_full_mapping, join(
        f'drop{args.drop_rate}mapping_reverse.pkl'))

    # transform clusters and mappings first
    new_cluster = {}
    for k, v in tqdm(cluster.items(), desc='cluster'):
        new_cluster[k] = [sampled_to_full_mapping[vv] for vv in v]
    new_mapping = {}
    for k, v in tqdm(mapping.items(), desc='mapping'):
        new_mapping[sampled_to_full_mapping[k]] = v
    for k, v in new_cluster.items():
        for vv in v:
            assert new_mapping[vv] == k

    # add dropped ones
    docemb = np.memmap(join('newdocembance.bin'), mode='r',
                       dtype=np.float32).reshape(-1, 768)
    print(docemb.shape)
    for start in tqdm(range(0, len(rest), args.batch_size), desc='rest'):
        ending = start + args.batch_size
        rs = rest[start:ending]
        cur_emb = docemb[rs]
        vecs = torch.FloatTensor(cur_emb)
        index = []
        for i in range(args.subvector_num):
            cur_codebook = codebook[i:i+1].expand(vecs.size(0), -1, -1)
            proba = compute_scores(vecs.unsqueeze(-2), cur_codebook)
            part_index = proba.max(dim=-1)[1]
            index.append(part_index)
            cur_centroid = codebook[i][part_index]
            if i != args.subvector_num - 1:
                vecs -= cur_centroid.detach()
        index = torch.stack(index, dim=1)
        index = index.detach().numpy().tolist()
        for r, ind in zip(rs, index):
            ind = tuple(ind)
            new_mapping[r] = ind
            if ind not in new_cluster:
                new_cluster[ind] = []
            new_cluster[ind].append(r)
    dump(new_cluster, join(f'rqclus{middle}_all.pkl'))
    dump(new_mapping, join(f'rqmapping{middle}_all.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, required=True)
    parser.add_argument('--drop_rate', type=str, required=True)
    parser.add_argument('--subvector_num', type=int, default=4)
    parser.add_argument('--subvector_bit', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    gen_sampled_to_full(args)
