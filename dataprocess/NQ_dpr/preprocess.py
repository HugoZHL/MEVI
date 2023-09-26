# from DPR
import csv
import json
import os.path as osp
import argparse
# import regex
# import unicodedata
# import pickle
from tqdm import tqdm


def get_collection(args):
    with open(osp.join(args.data_dir, 'psgs_w100.tsv'), 'r', encoding='utf-8') as fin, \
            open(osp.join(args.data_dir, 'corpus.tsv'), 'w', encoding='utf-8') as fout:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    fout.write(str(int(row[0]) - 1) +
                               '\t' + row[2] + '\t' + row[1] + '\n')
                except:
                    print(
                        f'The following input line has not been correctly loaded: {row}')
    # tokenizer = SimpleTokenizer()
    # all_texts = []
    # with open(osp.join(args.data_dir, 'corpus.tsv'), 'r', encoding='utf-8') as fr:
    #     for line in tqdm(fr, desc='Read Docs'):
    #         did, title, text = line.rstrip('\n').split('\t')
    #         text = _normalize(text)
    #         # Answer is a list of possible strings
    #         text = tokenizer.tokenize(text).words(uncased=True)
    #         all_texts.append(text)
    # with open(osp.join(args.data_dir, 'texts.pkl'), 'wb') as fw:
    #     pickle.dump(all_texts, fw)
    # return all_texts


def get_train(args):
    with open(osp.join(args.data_dir, 'biencoder-nq-train.json'), 'r') as fr, \
            open(osp.join(args.data_dir, 'train_mevi_dedup.tsv'), 'w', encoding='utf-8') as fw:
        data = json.load(fr)
        for idx, item in enumerate(tqdm(data, desc='Write Train')):
            query = item['question']
            query = query.replace("’", "'")
            positives = item['positive_ctxs']

            pids = [str(int(p['passage_id']) - 1) for p in positives]
            print(query, ','.join(pids), sep='\t', file=fw)

def get_train_expand(args):
    with open(osp.join(args.data_dir, 'train_mevi_dedup.tsv'), 'r') as fr, \
            open(osp.join(args.data_dir, 'train_mevi.tsv'), 'w', encoding='utf-8') as fw:
        for line in tqdm(fr, desc='Expand Train'):
            query, pids = line.rstrip('\n').split('\t')
            for p in pids.split(','):
                p = int(p)
                print(query, p, sep='\t', file=fw)



def get_simple_dev(args):
    queries = set()
    with open(osp.join(args.data_dir, 'biencoder-nq-dev.json'), 'r') as fr, \
            open(osp.join(args.data_dir, 'dev_mevi_dedup.tsv'), 'w', encoding='utf-8') as fw:
        data = json.load(fr)
        for idx, item in enumerate(tqdm(data, desc='Write Dev')):
            query = item['question']
            query = query.replace("’", "'")
            queries.add(query)
            positives = item['positive_ctxs']

            pids = [str(int(p['passage_id']) - 1) for p in positives]
            print(query, ','.join(pids), sep='\t', file=fw)
    return queries


# class SimpleTokenizer:
#     ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
#     NON_WS = r'[^\p{Z}\p{C}]'

#     def __init__(self, **kwargs):
#         """
#         Args:
#             annotators: None or empty set (only tokenizes).
#         """
#         self._regexp = regex.compile(
#             '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
#         )
#         self.annotators = set()

#     def tokenize(self, text):
#         data = []
#         matches = [m for m in self._regexp.finditer(text)]
#         for i in range(len(matches)):
#             # Get text
#             token = matches[i].group()

#             # Get whitespace
#             span = matches[i].span()
#             start_ws = span[0]
#             if i + 1 < len(matches):
#                 end_ws = matches[i + 1].span()[0]
#             else:
#                 end_ws = span[1]

#             # Format data
#             data.append((
#                 token,
#                 text[start_ws: end_ws],
#                 span,
#             ))
#         return Tokens(data, self.annotators)


# def _normalize(text):
#     return unicodedata.normalize('NFD', text)


# class Tokens(object):
#     """A class to represent a list of tokenized text."""
#     TEXT = 0
#     TEXT_WS = 1
#     SPAN = 2
#     POS = 3
#     LEMMA = 4
#     NER = 5

#     def __init__(self, data, annotators, opts=None):
#         self.data = data
#         self.annotators = annotators
#         self.opts = opts or {}

#     def __len__(self):
#         """The number of tokens."""
#         return len(self.data)

#     def words(self, uncased=False):
#         """Returns a list of the text of each token

#         Args:
#             uncased: lower cases text
#         """
#         if uncased:
#             return [t[self.TEXT].lower() for t in self.data]
#         else:
#             return [t[self.TEXT] for t in self.data]


# def strStr(s, p):
#     def getNext(p):
#         nex = [-1] * (len(p) + 1)
#         i = 0
#         j = -1
#         while i < len(p):
#             if j == -1 or p[i] == p[j]:
#                 i += 1
#                 j += 1
#                 nex[i] = j
#             else:
#                 j = nex[j]

#         return nex

#     nex = getNext(p)
#     i = 0
#     j = 0
#     while i < len(s) and j < len(p):
#         if j == -1 or s[i] == p[j]:
#             i += 1
#             j += 1
#         else:
#             j = nex[j]

#     if j == len(p):
#         return i - j
#     else:
#         return -1


# def get_dev_n_test(args, all_texts, phase='dev'):
#     assert phase in ('dev', 'test')
#     tokenizer = SimpleTokenizer()
#     with open(osp.join(args.data_dir, f'nq-{phase}.qa.csv'), 'r', encoding='utf-8') as fr, \
#             open(osp.join(args.data_dir, f'{phase}_mevi_dedup.tsv'), 'w', encoding='utf-8') as fw:

#         for idx, item in enumerate(tqdm(fr, desc=f'Write {phase.capitalize()}')):
#             query, answers = item.split('\t')
#             query = query.replace("’", "'")

#             positives = []

#             processed_answers = []
#             for single_answer in eval(answers):
#                 single_answer = _normalize(single_answer)
#                 single_answer = tokenizer.tokenize(single_answer)
#                 single_answer = single_answer.words(uncased=True)
#                 processed_answers.append(single_answer)

#             for did, text in enumerate(all_texts):
#                 for single_answer in processed_answers:
#                     if strStr(text, single_answer) >= 0:
#                         positives.append(did)
#                         break
#             positives = sorted(list(set(positives)))
#             positives = ','.join([str(p) for p in positives])
#             print(query, positives, sep='\t', file=fw, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    all_texts = get_collection(args)
    get_train(args)
    get_train_expand(args)
    queries = get_simple_dev(args)
    print('Processed dev queries: ', len(queries))
    # get_dev_n_test(args, all_texts, 'dev')
    # get_dev_n_test(args, all_texts, 'test')
