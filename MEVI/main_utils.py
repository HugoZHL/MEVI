import random
from collections import defaultdict
from typing import List
import torch
import numpy as np
from os import listdir
import os.path as osp
from os.path import isfile, join
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def grad_status(model):
    return (par.requires_grad for par in model.parameters())


def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(
        model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"


def dec_2d(dec, size):
    if isinstance(dec, torch.Tensor):
        return dec.reshape(-1, size, dec.shape[-1])
    else:
        res = []
        i = 0
        while i < len(dec):
            res.append(dec[i: i + size])
            i = i + size
        return res


# decoder helper
def numerical_decoder(args, cuda_ids, output):
    np_ids = cuda_ids.cpu().numpy()
    begin_and_end_token = np.where(np_ids == 1)

    if output:
        if len(begin_and_end_token) != 1 or begin_and_end_token[0].size < 1:
            print("Invalid Case")
            return "0"
        if args.hierarchic_decode:
            np_ids = np_ids[1:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[1:begin_and_end_token[0][0]] -
                      2) % args.output_vocab_size
    else:
        if args.hierarchic_decode:
            np_ids = np_ids[:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[:begin_and_end_token[0][0]] -
                      2) % args.output_vocab_size

    bits = int(np.log10(args.output_vocab_size))
    num_list = list(map(str, np_ids))
    str_ids = ''.join([c.zfill(bits) if (i != len(num_list) - 1)
                      else c for i, c in enumerate(num_list)])
    return str_ids


def random_shuffle(doc_id):
    new_doc_id = ""
    for index in range(0, len(doc_id)):
        while True:
            rand_digit = np.random.randint(0, 9)
            if not rand_digit == int(doc_id[index]):
                new_doc_id += str(rand_digit)
                break
    return new_doc_id


def augment(query):
    if len(query) < 20*10:
        start_pos = np.random.randint(0, int(len(query)+1/2))
        end_pos = np.random.randint(start_pos, len(query))
        span_length = max(start_pos-end_pos, 10*10)
        new_query = str(query[start_pos:start_pos+span_length])
    else:
        start_pos = np.random.randint(0, len(query)-10*10)
        end_pos = np.random.randint(start_pos+5*10, len(query))
        span_length = min(start_pos-end_pos, 20*10)
        new_query = str(query[start_pos:start_pos+span_length])
    # print(new_query)
    return new_query


def load_data(args):
    df = None
    q_emb, query_id_dict_train = None, None
    prefix_embedding = None
    prefix_mask = None
    prefix2idx_dict = None
    doc_to_query_list = None
    assert args.contrastive_variant == ''

    if 'gtq' in args.query_type:
        if args.dataset in ('marco', 'nq_dpr'):
            fpath = 'train_mevi.tsv'
            if args.drop_data_rate > 0:
                fpath = f'train_mevi_drop{args.drop_data_rate}.tsv'
            train_file = join(args.data_dir, fpath)
            df = pd.read_csv(
                train_file,
                names=['query', 'oldid'],
                encoding='utf-8',
                header=None,
                sep='\t',
                dtype={'query': str, 'oldid': int}
            )
        assert not df.isnull().values.any()
        doc_to_query_list = defaultdict(set)
        for [query, docid] in df.values.tolist():
            doc_to_query_list[docid].add(query)

        if 'qg' in args.query_type:
            gq_df1 = None
            if args.dataset in ('marco', 'nq_dpr'):
                entries = args.query_type.split('_')
                qg_num = None
                for en in entries:
                    if en.startswith('qg'):
                        qg_num = int(en[2:])
                        break
                assert qg_num is not None and qg_num > 0
                fpath = f'qg{qg_num}.tsv'
                if args.drop_data_rate > 0:
                    fpath = f'qg{qg_num}_drop{args.drop_data_rate}.tsv'
                qg_file = join(args.data_dir, fpath)
                gq_df1 = pd.read_csv(
                    qg_file,
                    names=['query', 'oldid'],
                    encoding='utf-8',
                    header=None,
                    sep='\t',
                    dtype={'query': str, 'oldid': int}
                )
            if gq_df1 is not None:
                gq_df1 = gq_df1.dropna(axis=0)
                print(len(gq_df1))
                for [query, docid] in gq_df1.values.tolist():
                    doc_to_query_list[docid].add(query)
                temp = defaultdict(list)
                for k, v in doc_to_query_list.items():
                    temp[k] = list(v)
                doc_to_query_list = temp

                df = pd.concat((df, gq_df1), axis=0, ignore_index=True)

    path_list = []
    if 'doc' in args.query_type:
        if args.dataset == 'marco':
            fpath = '../raw/corpus.tsv'
            if args.drop_data_rate > 0:
                fpath = f'corpus_drop{args.drop_data_rate}.tsv'
            filename = join(args.data_dir, fpath)
        elif args.dataset == 'nq_dpr':
            fpath = 'corpus.tsv'
            filename = join(args.data_dir, fpath)
        path_list.append(filename)

    if 'doc_aug' in args.query_type:
        if args.dataset == 'marco':
            assert args.drop_data_rate == 0
            filename = join(args.data_dir, 'doc_aug.tsv')
        elif args.dataset == 'nq_dpr':
            assert False, 'Not implemented.'
        path_list.append(filename)

    for file_path in path_list:
        print(file_path)
        if args.dataset in ('marco', 'nq_dpr'):
            if osp.split(file_path)[-1].startswith('corpus'):
                ori_doc_data = pd.read_csv(
                    file_path,
                    names=["oldid", "title", "content"],
                    encoding='utf-8',
                    header=None,
                    sep='\t',
                    dtype={'oldid': int, 'title': str, 'content': str}
                )
                ori_doc_data.fillna('', inplace=True)
                if args.document_encoder == 'ance':
                    doc_data = pd.concat(
                        ('Title: ' + ori_doc_data['title'] + ' Text: ' + ori_doc_data['content'], ori_doc_data['oldid']), axis=1)
                else:
                    from transformers import AutoTokenizer
                    if args.document_encoder == 'ar2':
                        passage_tokenizer = AutoTokenizer.from_pretrained(
                            "bert-base-uncased", do_lower_case=True)
                    else:
                        passage_tokenizer = AutoTokenizer.from_pretrained(
                            'bert-base-uncased', use_fast=True)
                    doc_data = pd.concat(
                        (ori_doc_data['title'] + passage_tokenizer.sep_token + ori_doc_data['content'], ori_doc_data['oldid']), axis=1)
                doc_data.columns.values[0] = 'query'
                df1 = doc_data
            else:
                df1 = pd.read_csv(
                    file_path,
                    names=["query", "oldid"],
                    encoding='utf-8',
                    header=None,
                    sep='\t',
                    dtype={'query': str, 'oldid': int}
                )

        df1.dropna(axis=0, inplace=True)
        assert not df1.isnull().values.any()
        df = df1 if df is None else pd.concat(
            (df, df1), axis=0, ignore_index=True)
    print('&' * 20)
    print(df.loc[0])
    print('&' * 20)

    return df, doc_to_query_list, q_emb, query_id_dict_train, prefix_embedding, prefix_mask, prefix2idx_dict


def load_data_infer(args):
    df = None
    def comma_split(x): return [int(xx) for xx in x.split(',')]
    if args.test_set == 'dev':
        if args.dataset == 'marco':
            if args.eval_train_data and 'doc' in args.query_type:
                # workaround for generating doc predictions
                if args.dataset == 'marco':
                    fpath = '../raw/corpus.tsv'
                elif args.dataset == 'nq_dpr':
                    fpath = 'corpus.tsv'
                if args.drop_data_rate > 0:
                    fpath = f'corpus_drop{args.drop_data_rate}.tsv'
                file_path = join(args.data_dir, fpath)
                ori_doc_data = pd.read_csv(
                    file_path,
                    names=["oldid", "title", "content"],
                    encoding='utf-8',
                    header=None,
                    sep='\t',
                    dtype={'oldid': int, 'title': str, 'content': str}
                )
                ori_doc_data.fillna('', inplace=True)
                doc_data = pd.concat(
                    ('Title: ' + ori_doc_data['title'] + ' Text: ' + ori_doc_data['content'], ori_doc_data['oldid']), axis=1)
                doc_data.columns.values[0] = 'query'
                doc_data['oldid'] = doc_data['oldid'].map(lambda x: [x])
                df = doc_data
            else:
                if args.eval_train_data:
                    fname = 'train_mevi_dedup.tsv'
                else:
                    fname = 'dev_mevi_dedup.tsv'
                if args.drop_data_rate > 0:
                    fname = f'{fname[:-4]}_drop{args.drop_data_rate}.tsv'
                dev_file = join(args.data_dir, fname)
                df = pd.read_csv(
                    dev_file,
                    names=["query", "oldid"],
                    encoding='utf-8',
                    header=None,
                    sep='\t',
                    converters={'oldid': comma_split}
                )
        elif args.dataset == 'nq_dpr':
            dev_file = join(args.data_dir, 'nq-test.qa.csv')
            df = pd.read_csv(
                dev_file,
                names=['query', 'answers'],
                encoding='utf-8',
                header=None,
                sep='\t',
            )

    assert not df.isnull().values.any()

    return df


def load_data_tokenized(args):
    assert args.dataset == 'nq_dpr', 'Now only support tokenized nq_dpr training data.'
    bin_files = []

    if 'gtq' in args.query_type:
        fpath = f'train_mevi_tokenized{args.max_input_length}.bin'
        train_file = join(args.data_dir, fpath)
        bin_files.append(train_file)

        if 'qg' in args.query_type:
            entries = args.query_type.split('_')
            qg_num = None
            for en in entries:
                if en.startswith('qg'):
                    qg_num = int(en[2:])
                    break
            assert qg_num is not None and qg_num > 0
            fpath = f'qg{qg_num}_tokenized{args.max_input_length}.bin'
            qg_file = join(args.data_dir, fpath)
            bin_files.append(qg_file)

    if 'doc' in args.query_type:
        fpath = f'corpus_tokenized{args.max_input_length}.bin'
        filename = join(args.data_dir, fpath)
        bin_files.append(filename)

    return bin_files
