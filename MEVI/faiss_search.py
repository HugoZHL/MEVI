import numpy as np
import argparse
import faiss
from time import time
from contextlib import contextmanager
import pickle


def read(path, dim):
    return np.fromfile(path, dtype=np.float32).reshape(-1, dim)


def search(query, doc, dim, topk, param):
    measure = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(dim, param, measure)
    print(f'Param {param} trained: {index.is_trained}.')
    if not index.is_trained:
        index.train(doc)
    index.add(doc)
    dists, indices = index.search(query, topk)
    return dists, indices


@contextmanager
def timer(record, key):
    start = time()
    yield
    ending = time()
    record[key] += (ending - start)


def profile(query, doc, dim, topk, param, bs=[1, 2, 4, 8]):
    measure = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(dim, param, measure)
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    all_time = {'train': 0, 'add': 0}
    for b in bs:
        all_time[f'search_bs{b}'] = 0
    print(f'Param {param} trained: {index.is_trained}.')
    with timer(all_time, 'train'):
        if not index.is_trained:
            index.train(doc)
    with timer(all_time, 'add'):
        index.add(doc)
    nquery = len(query)
    for b in bs:
        print(f'Profile batch size {b}...')
        batched_queries = []
        for start in range(0, nquery, b):
            ending = start + b
            if ending <= nquery:
                cur_queries = query[start:ending]
            else:
                cur_queries = query[start:ending]
                rest = np.random.choice(
                    np.arange(nquery), size=b - len(cur_queries), replace=False)
                cur_queries = np.concatenate(
                    (cur_queries, query[rest]), axis=0)
                assert len(cur_queries) == b
            batched_queries.append(cur_queries)
            if len(batched_queries) >= 10:
                break
        with timer(all_time, f'search_bs{b}'):
            for bq in batched_queries:
                dists, indices = index.search(bq, topk)
        all_time[f'search_bs{b}'] /= len(batched_queries)
    return all_time


def to_file(query_path, output_path, dists, indices):
    with open(query_path, 'r') as fr, open(output_path, 'w') as fw:
        for i, line in enumerate(fr):
            query = line.split('\t')[0]
            preds = ','.join([str(ind) for ind in indices[i].tolist()])
            scores = ','.join([str(sco) for sco in dists[i].tolist()])
            print(f'{query}\t\t{preds}\t{scores}', file=fw)


if __name__ == '__main__':
    # simply generate embeddings for further knn or ann
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True)
    parser.add_argument('--doc_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--raw_query_path', type=str, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--param', type=str, default='IVF100,Flat')
    args = parser.parse_args()
    query = read(args.query_path, args.dim)
    doc = read(args.doc_path, args.dim)
    # all_time = profile(query, doc, args.dim, args.topk, args.param)
    # with open(f'profile_faiss_{args.param}.pkl', 'wb') as fw:
    #     pickle.dump(all_time, fw)
    dists, indices = search(query, doc, args.dim, args.topk, args.param)
    print(indices.dtype, indices.shape, dists.dtype, dists.shape)
    to_file(args.raw_query_path, args.output_path, dists, indices)
