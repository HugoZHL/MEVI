import pickle
from tqdm import tqdm
import os.path as osp
from argparse import ArgumentParser
from itertools import chain


def get_metric(v, recalls, mrrs, hitrates):
    v_valid = [vv for vv in v if vv is not None]
    min_valid = min(v_valid) if len(v_valid) > 0 else None
    for recnum in recalls:
        if len(v_valid) > 0:
            cur_recall = sum(
                [vv < recnum for vv in v_valid]) / len(v)
            cur_mrr = 1 / \
                (min_valid + 1) if min_valid < recnum else 0
            cur_hit = (min_valid < recnum)
        else:
            cur_recall = 0
            cur_mrr = 0
            cur_hit = 0
        recalls[recnum] += cur_recall
        mrrs[recnum] += cur_mrr
        hitrates[recnum] += cur_hit
    return v_valid, min_valid


def evaluate(scoring, recall_num, ofile, gts, scores=None, ranks=None):
    assert ranks is not None or scores is not None
    recalls = {r: 0 for r in recall_num}
    mrrs = {m: 0 for m in recall_num}
    hitrates = {h: 0 for h in recall_num}
    for q in gts:
        if ranks is not None:
            preds = ranks[q]
        else:
            score = scores[q]
            preds = sorted(score.items(), key=lambda x: -x[1])
            preds = [p[0] for p in preds]
        gt = gts[q]
        vs = [preds.index(g) if g in preds else None for g in gt]
        get_metric(vs, recalls, mrrs, hitrates)

    nqueries = len(gts)
    for recnum in recalls:
        recalls[recnum] /= nqueries
        mrrs[recnum] /= nqueries
        hitrates[recnum] /= nqueries

    print(f'{scoring}')
    for k, v in recalls.items():
        print(f"Recall{k}", v)

    for k, v in mrrs.items():
        print(f"MRR{k}", v)

    # for k, v in hitrates.items():
    #     print(f"HitRate{k}", v)
    print()

    if ofile is not None:
        with open(ofile, 'a') as fw:
            print(f'Scoring {scoring}', file=fw)
            for k, v in recalls.items():
                print(f"Recall{k}", v, file=fw)

            for k, v in mrrs.items():
                print(f"MRR{k}", v, file=fw)

            # for k, v in hitrates.items():
            #     print(f"HitRate{k}", v, file=fw)
            print(file=fw)


def check_exists(fpath, dpath, nonexist_ok=False):
    if fpath is not None and not osp.exists(fpath):
        fpath = osp.join(dpath, fpath)
    if nonexist_ok:
        return fpath, fpath is not None and osp.exists(fpath)
    else:
        assert osp.exists(fpath)
        return fpath


def eval_list(item):
    if item[0] != '[':
        item = f'[{item}]'
    item = eval(item)
    return item


def parse_file(fpath, template):
    qind = template['query']
    pind = template.get('pred', None)
    cur_pred = {}
    sind = template.get('score', None)
    cur_score = {}
    cind = template.get('cluster', None)
    cur_cluster = {}

    with open(fpath, 'r') as fr:
        for line in tqdm(fr, desc=f'File {osp.split(fpath)[-1]}'):
            items = line.rstrip('\n').split('\t')
            query = items[qind]
            if pind is not None:
                cur_pred[query] = eval_list(items[pind])
            if sind is not None:
                cur_score[query] = eval_list(items[sind])
            if cind is not None:
                cur_cluster[query] = eval_list(items[cind])
    return cur_pred, cur_score, cur_cluster


def load(path):
    with open(path, 'rb') as fr:
        results = pickle.load(fr)
    return results


def dump(items, path):
    with open(path, 'wb') as fw:
        pickle.dump(items, fw)


def remove_extension(fpath):
    extension = fpath.split('.')[-1]
    lextension = len(extension) + 1
    return fpath[:-lextension]


def check_cache(fpath, template):
    cache_file = remove_extension(fpath) + '.pkl'
    if fpath.endswith('.pkl'):
        results = load(fpath)
    elif osp.exists(cache_file):
        results = load(cache_file)
    else:
        results = parse_file(fpath, template)
        dump(results, cache_file)
    return results


def check_same(ori, new):
    assert ori in (new, None)
    return new


def split(xs, dtype=float):
    return [dtype(x) for x in xs.split(',')]


def combine_main(args):
    assert osp.exists(args.mapping_file)
    args.alphas = split(args.alphas)
    args.betas = split(args.betas)
    args.gammas = split(args.gammas)
    args.recall_num = split(args.recall_num, int)
    args.gt_file = check_exists(args.gt_file, args.dir_path)
    args.ance_file = check_exists(args.ance_file, args.dir_path)
    args.fine_file, fexists = check_exists(
        args.fine_file, args.dir_path, True)
    args.coarse_file = check_exists(args.coarse_file, args.dir_path)

    gt_template = {'query': 0, 'pred': -1}
    fine_template = {'query': 0, 'pred': 2, 'score': 3}
    coarse_template = {'query': 0, 'cluster': 1}

    gts, _, _ = check_cache(args.gt_file, gt_template)
    ance_preds, ance_scores, _ = check_cache(args.ance_file, fine_template)
    if fexists:
        fine_preds, fine_scores, _ = check_cache(args.fine_file, fine_template)
    _, _, coarse_clusters = check_cache(args.coarse_file, coarse_template)
    mapping = load(args.mapping_file)

    num_clusters = None
    cr4gt = remove_extension(args.coarse_file) + '_cr4gt.pkl'
    if osp.exists(cr4gt):
        cluster_rankings_gt, new_num_clusters = load(cr4gt)
        num_clusters = check_same(num_clusters, new_num_clusters)
    else:
        cluster_rankings_gt = {}
        for q, apreds in tqdm(ance_preds.items(), desc='Cluster Ranking for GT'):
            cr = {}
            for i, clus in enumerate(coarse_clusters[q]):
                cr[tuple(clus)] = i
            assert num_clusters in (None, len(cr))
            new_num_clusters = len(cr)
            num_clusters = check_same(num_clusters, new_num_clusters)
            cluster_rankings_gt[q] = [
                cr.get(mapping[p] if p != -1 else -1, len(cr)) for p in apreds]
        dump((cluster_rankings_gt, num_clusters), cr4gt)

    if fexists:
        cr4fine = remove_extension(args.fine_file) + '_cr.pkl'
        if osp.exists(cr4fine):
            cluster_rankings_fine, new_num_clusters = load(cr4fine)
            num_clusters = check_same(num_clusters, new_num_clusters)
        else:
            cluster_rankings_fine = {}
            for q, apreds in tqdm(ance_preds.items(), desc='Cluster Ranking for Fine'):
                cr = {}
                for i, clus in enumerate(coarse_clusters[q]):
                    cr[tuple(clus)] = i
                assert num_clusters in (None, len(cr))
                new_num_clusters = len(cr)
                num_clusters = check_same(num_clusters, new_num_clusters)
                cluster_rankings_fine[q] = [
                    cr.get(mapping[p] if p != -1 else -1, len(cr)) for p in apreds]
            dump((cluster_rankings_fine, num_clusters), cr4fine)

    if args.ofile is not None:
        with open(args.ofile, 'w') as fw:
            pass

    evaluate(f'ANCE Pred', args.recall_num, args.ofile, gts, ranks=ance_preds)

    if fexists:
        evaluate(f'Fine Pred', args.recall_num,
                 args.ofile, gts, ranks=fine_preds)

    for alpha in args.alphas:
        for beta in args.betas:
            for gamma in args.gammas:
                scores = {q: {} for q in gts}
                for q, apreds in ance_preds.items():
                    ascores = ance_scores[q]
                    cluster_ranking = cluster_rankings_gt[q]
                    if fexists:
                        fscores = fine_scores[q]
                        fpreds = fine_preds[q]
                        apreds = apreds + fpreds
                        ascores = ascores + fscores
                        cluster_ranking = chain(
                            cluster_ranking, cluster_rankings_fine[q])
                    for i, (p, s, crank) in enumerate(zip(apreds, ascores, cluster_ranking)):
                        scores[q][p] = s + alpha / (beta * crank + 1)
                        if crank == num_clusters:
                            scores[q][p] *= (1 - gamma * alpha)
                evaluate(f'score + {alpha} / ({beta} * crank + 1); punishment (1 - {gamma} * {alpha})',
                         args.recall_num, args.ofile, gts, scores=scores)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=None)
    parser.add_argument('--gt_file', type=str, required=True)
    parser.add_argument('--ance_file', type=str, required=True)
    parser.add_argument('--fine_file', type=str, default=None)
    parser.add_argument('--coarse_file', type=str, default=None)
    parser.add_argument('--mapping_file', type=str, default=None)
    parser.add_argument('--alphas', type=str, default='0.6')
    parser.add_argument('--betas', type=str, default='0.03')
    parser.add_argument('--gammas', type=str, default='0.02')
    parser.add_argument('--recall_num', type=str,
                        default='10,50,1000')
    parser.add_argument('--ofile', type=str, default=None)
    args = parser.parse_args()

    combine_main(args)
