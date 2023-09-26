# from DPR
import os.path as osp
import argparse
import regex
import unicodedata
import pickle
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp


def get_queries(args, phase):
    tokenizer = SimpleTokenizer()
    all_answers = []
    with open(osp.join(args.data_dir, f'nq-{phase}.qa.csv'), 'r', encoding='utf-8') as fr:
        for item in tqdm(fr, desc=f'read {phase}'):
            query, answers = item.split('\t')
            query = query.replace("’", "'")

            processed_answers = []
            answers = answers.strip()
            if answers[0] != '[':
                answers = answers.strip('"').replace('""', '"')
            for single_answer in eval(answers):
                single_answer = _normalize(single_answer)
                single_answer = tokenizer.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)
                processed_answers.append(single_answer)
            all_answers.append(processed_answers)

    with open(osp.join(args.data_dir, f'{phase}_texts.pkl'), 'wb') as fw:
        pickle.dump(all_answers, fw)
    return all_answers


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def _normalize(text):
    return unicodedata.normalize('NFD', text)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]


def strStr(s, p):
    def getNext(p):
        nex = [-1] * (len(p) + 1)
        i = 0
        j = -1
        while i < len(p):
            if j == -1 or p[i] == p[j]:
                i += 1
                j += 1
                nex[i] = j
            else:
                j = nex[j]

        return nex

    nex = getNext(p)
    i = 0
    j = 0
    while i < len(s) and j < len(p):
        if j == -1 or s[i] == p[j]:
            i += 1
            j += 1
        else:
            j = nex[j]

    if j == len(p):
        return i - j
    else:
        return -1


def get_relevant(rank, args, phase):
    tokenizer = SimpleTokenizer()
    ndoc = 21015324
    docs_per_rank = ndoc // args.nrank
    docs_rest = ndoc % args.nrank
    local_start = docs_per_rank * rank + min(rank, docs_rest)
    local_ending = local_start + docs_per_rank + (rank < docs_rest)
    with open(osp.join(args.data_dir, f'{phase}_texts.pkl'), 'rb') as fr:
        all_answers = pickle.load(fr)
    results = [[] for _ in range(len(all_answers))]

    with open(osp.join(args.data_dir, 'corpus.tsv'), 'r') as fr:
        offset = local_start
        while offset > 0:
            _ = fr.readline()
            offset -= 1
        for lidx in tqdm(range(local_start, local_ending)):
            line = fr.readline()
            did, title, text = line.rstrip('\n').split('\t')
            assert int(did) == lidx
            text = _normalize(text)
            # Answer is a list of possible strings
            text = tokenizer.tokenize(text).words(uncased=True)
            for res, answers in zip(results, all_answers):
                flag = False
                for ans in answers:
                    if strStr(text, ans) >= 0:
                        flag = True
                        break
                if flag:
                    res.append(lidx)
    with open(osp.join(args.data_dir, f'{phase}_output_{rank}.pkl'), 'wb') as fw:
        pickle.dump(results, fw)


def start_process(args, phase):
    get_queries(args, phase)
    print('Start multi processing...')
    mp.spawn(get_relevant,
             args=(args, phase),
             nprocs=args.nrank,
             join=True)
    print('Finish multi processing...')
    with open(osp.join(args.data_dir, f'{phase}_output_0.pkl'), 'rb') as fr:
        all_results = pickle.load(fr)
    for i in range(1, args.nrank):
        with open(osp.join(args.data_dir, f'{phase}_output_{i}.pkl'), 'rb') as fr:
            cur_results = pickle.load(fr)
        all_results = [a + b for a, b in zip(all_results, cur_results)]
    with open(osp.join(args.data_dir, f'{phase}_output.pkl'), 'wb') as fw:
        pickle.dump(all_results, fw)
    output_offset = np.zeros(shape=(len(all_results) + 1,), dtype=np.int32)
    output_offset[0] = 0
    for i, res in enumerate(all_results):
        output_offset[i+1] = output_offset[i] + len(res)
    output_offset.tofile(osp.join(args.data_dir, f'{phase}_offsets.bin'))
    output_array = np.memmap(osp.join(
        args.data_dir, f'{phase}_array.bin'), mode='w+', dtype=np.int32, shape=(output_offset[-1] + 1,))
    for i, res in enumerate(all_results):
        output_array[output_offset[i]:output_offset[i+1]] = res
    output_array.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--nrank', type=int, default=40)
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    if args.dev:
        start_process(args, 'dev')
    if args.test:
        start_process(args, 'test')
