import os
import os.path as osp
import pickle
import random
from argparse import Namespace
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
if torch.__version__ >= '1.11.0':
    from torch.utils.data import default_collate
else:
    default_collate = torch.utils.data._utils.collate.default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from main_utils import assert_all_frozen, load_data_infer, random_shuffle, augment, \
    load_data, numerical_decoder, dec_2d, load_data_tokenized
from transformers import (
    AdamW,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from time import time


class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self, share_sons=False) -> None:
        self.root = Node(0)
        self.share_sons = share_sons

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        '''
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        '''
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]

    def add_layer(self, seq) -> None:
        # add a whole shared layer
        assert self.share_sons
        cur = self.root
        prev_layer = [cur]
        new_nodes = {tok: Node(tok) for tok in seq if tok != 0}
        while prev_layer[0].children != {}:
            prev_layer = list(prev_layer[0].children.values())
        for p in prev_layer:
            p.children = new_nodes


def encode_single_newid(args, seq):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    if args.kary:
        if isinstance(seq, str):
            en = enumerate(seq.split('-'))
        else:
            en = enumerate(seq)
        kary = args.kary
    else:
        en = enumerate(seq)
        kary = 10
    for i, c in en:
        if args.position:
            cur_token = i * kary + int(c) + 2
        else:
            cur_token = int(c) + 2
        target_id_int.append(cur_token)
    if args.label_length_cutoff:
        target_id_int = target_id_int[:args.max_output_length-2]
    return target_id_int + [1]  # append eos_token


def vq_label_suffix(target_id):
    bs = target_id.shape[0]
    return torch.cat([target_id, target_id.new_ones(
        bs, 1), target_id.new_zeros(bs, 1)], dim=1)


def decode_token(args, seqs):
    '''
    Param:
        seqs: 2d torch Tensor to be decoded
    Return:
        doc_id string, List[str]
    '''
    if args.codebook:
        assert args.subvector_num + 2 == seqs.shape[1]
        eos_idx = None
    else:
        eos_idx = torch.nonzero(seqs == 1, as_tuple=True)[1] - 1
        eos_idx.unsqueeze_(-1)
    seqs = seqs[:, 1:-1]
    seqs -= 2
    if args.position:
        seqs -= torch.arange(seqs.size(1),
                             device=seqs.device) * args.output_vocab_size
    seqs[seqs < 0] = 0
    return seqs, eos_idx


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PinMemoryBatch:
    def __init__(self, data):
        self.data = data

    # custom memory pinning method on custom type
    def pin_memory(self):
        for k in self.data:
            v = self.data[k]
            if isinstance(v, torch.Tensor):
                v = v.pin_memory()
            self.data[k] = v
        return self

    def __getitem__(self, key):
        return self.data[key]


def custom_collate(batch):
    has_docids = 'doc_ids' in batch[0]
    has_newids = 'new_ids' in batch[0]
    has_qinds = 'query_indices' in batch[0]
    if has_docids:
        oldids = [b.pop('doc_ids') for b in batch]
    if has_newids:
        newids = [b.pop('new_ids') for b in batch]
    if has_qinds:
        qinds = [b.pop('query_indices') for b in batch]
    queries = [b.pop('query') for b in batch]
    new_batch = default_collate(batch)
    if has_docids:
        new_batch['doc_ids'] = oldids
    if has_newids:
        new_batch['new_ids'] = newids
    if has_qinds:
        new_batch['query_indices'] = qinds
    new_batch['query'] = queries
    return PinMemoryBatch(new_batch)


def move_to_tmp(filepath):
    dirpath, filename = osp.split(filepath)
    assert osp.isdir(dirpath)
    return osp.join('/tmp', filename)


class LogFile(object):
    def __init__(self, file, rank, nrank, barrier_func):
        self.rank = rank
        self.nrank = nrank
        self.final_file = file
        # if osp.exists(file):
        #     print(f'[Warning]: {file} already exists!')
        self.tmp_fprefix = move_to_tmp(file)
        self.tmp_file = f'{self.tmp_fprefix}_{rank}'
        assert not osp.exists(self.tmp_file)
        self.barrier = barrier_func
        self.lines = []
        self.finished = False

    def add(self, item):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def clear(self):
        self.lines.clear()

    def wrapped_flush(self):
        if len(self.lines) > 0:
            self.flush()
            self.clear()

    def merge(self):
        raise NotImplementedError

    def wrapped_merge(self):
        self.wrapped_flush()
        self.barrier()
        if self.rank == 0:
            self.merge()
            self.safe_rm()
        self.barrier()
        self.finished = True

    def safe_rm(self):
        for i in range(self.nrank):
            tmp_fname = f'{self.tmp_fprefix}_{i}'
            if osp.isfile(tmp_fname):
                os.remove(tmp_fname)

    @classmethod
    def log_file(cls, item, *args):
        instance = cls(*args)
        instance.add(item)
        instance.wrapped_merge()
        del instance


class LogTxtFile(LogFile):
    def __init__(self, file, rank, nrank, barrier_func, maxline=None):
        super().__init__(file, rank, nrank, barrier_func)
        self.maxline = maxline if maxline is not None else float('inf')

    def add(self, item):
        self.lines.append(item)
        if len(self.lines) >= self.maxline:
            self.wrapped_flush()

    def flush(self):
        with open(self.tmp_file, 'a') as fw:
            for line in self.lines:
                print(*line, file=fw, sep='\t')

    def merge(self):
        with open(self.final_file, 'w') as fw:
            for i in range(self.nrank):
                tmp_fname = f'{self.tmp_fprefix}_{i}'
                with open(tmp_fname, 'r') as fr:
                    if isinstance(self.maxline, int):
                        size = self.maxline * 100
                        lines = fr.readlines(size)
                        while lines != []:
                            _ = fw.writelines(lines)
                            # minor adjustment
                            size += (self.maxline - len(lines)) * 16
                            lines = fr.readlines(size)
                    else:
                        _ = fw.writelines(fr.readlines())


class LogPklFile(LogFile):
    def __init__(self, file, rank, nrank, barrier_func, dtype):
        assert dtype in ('cluster', 'list', 'dict')
        super().__init__(file, rank, nrank, barrier_func)
        self.dtype = dtype

    def add(self, item):
        self.lines = item

    def flush(self):
        with open(self.tmp_file, 'wb') as fw:
            pickle.dump(self.lines, fw)

    def merge(self):
        dtype = self.dtype
        with open(self.final_file, 'wb') as fw:
            if dtype == 'list':
                all_items = []
            else:
                all_items = {}
            for i in range(self.nrank):
                tmp_fname = f'{self.tmp_fprefix}_{i}'
                with open(tmp_fname, 'rb') as fr:
                    obj = pickle.load(fr)
                    if dtype == 'list':
                        all_items.extend(obj)
                    elif dtype == 'dict':
                        all_items.update(obj)
                    else:
                        for k, v in obj.items():
                            if k in all_items:
                                all_items[k].extend(v)
                            else:
                                all_items[k] = v
            pickle.dump(all_items, fw)


class LogTorchFile(LogFile):
    def add(self, item):
        self.lines = item

    def flush(self):
        torch.save(self.lines, self.tmp_file)

    def clear(self):
        del self.lines

    def merge(self):
        all_items = []
        for i in range(self.nrank):
            tmp_fname = f'{self.tmp_fprefix}_{i}'
            all_items.append(torch.load(tmp_fname))
        all_items = torch.cat(all_items)
        torch.save(all_items, self.final_file)


class MemmapList(object):
    def __init__(self, mmap_files, dim, dtype=np.int32):
        self.files = [
            np.memmap(f, dtype=dtype, mode='r').reshape(-1, dim) for f in mmap_files]
        self.num = len(self.files)
        self.offsets = [0]
        for f in self.files:
            self.offsets.append(self.offsets[-1] + len(f))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        # brute-force, since currently only 3 files in total
        cnt = 0
        while index >= self.offsets[cnt+1]:
            cnt += 1
        offset = index - self.offsets[cnt]
        return np.array(self.files[cnt][offset])


class l1_query(Dataset):
    def __init__(self, model, args, query_tokenizer, passage_tokenizer, num_samples, mapping, print_text=False, task='train', all_docs=None, doc_cluster=None, rank=None, nrank=None):
        assert task in ['train', 'test']
        self.args = args
        self.tokenizer = query_tokenizer
        self.passage_tokenizer = passage_tokenizer
        self.add_special_tokens = args.dataset in (
            'marco', 'nq_dpr') and args.document_encoder in ('ance', 'ar2')
        input_length = args.max_input_length
        output_length = args.max_output_length * \
            int(np.log10(args.output_vocab_size))
        inf_input_length = args.inf_max_input_length
        random_gen = args.random_gen
        softmax = args.softmax
        aug = args.aug
        self.task = task
        self.model = model
        self.hashed_seed = hash(args.seed)
        self.vq_with_label = args.codebook and (
            self.task == 'test' or not args.pq_runtime_label)

        self.released = False
        self.input_length = input_length
        self.load_dataset(num_samples, rank, nrank)
        self.num_samples = num_samples

        self.doc_length = self.args.doc_length
        self.co_doc_length = self.args.co_doc_length
        self.inf_input_length = inf_input_length
        self.output_length = output_length
        self.print_text = print_text
        self.softmax = softmax
        self.aug = aug
        self.random_gen = random_gen
        if random_gen:
            assert len(self.dataset[0]) >= 3
        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token,
                      self.tokenizer.pad_token, self.tokenizer.cls_token,
                      self.tokenizer.mask_token] + self.tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

        self.mapping = mapping
        self.all_docs = all_docs
        self.doc_cluster = doc_cluster

    def load_dataset(self, num_samples, rank, nrank):
        args = self.args
        task = self.task
        if task == 'train':
            if args.dataset == 'nq_dpr':
                data_files = load_data_tokenized(args)
                self.dataset = MemmapList(
                    data_files, 2 * self.input_length + 1)
                self.q_emb, self.query_dict, \
                    self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict \
                    = None, None, None, None, None
            else:
                if args.split_data and rank is not None:
                    self.load_split_data(rank, nrank)
                else:
                    self.dataset, self.doc_to_query_list, self.q_emb, self.query_dict, \
                        self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = \
                        load_data(args)
        elif task == 'test':
            if args.split_data and rank is not None:
                self.load_split_data(rank, nrank)
            else:
                self.dataset = load_data_infer(args)
                self.q_emb, self.query_dict, \
                    self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict \
                    = None, None, None, None, None
        else:
            raise NotImplementedError("No Corresponding Task.")

        if num_samples:
            self.dataset = self.dataset[:num_samples]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def convert_to_features(self, example_batch, length_constraint):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.args.dataset not in ('marco', 'nq_dpr'):
            input_ = self.clean_text(example_batch)
        else:
            input_ = example_batch
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return output_

    def tokenize_passage(self, contents, max_length):
        output_ = self.passage_tokenizer.batch_encode_plus(
            contents,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=self.add_special_tokens,
            return_tensors='pt',
        )
        return output_

    def __getitem__(self, index):
        args = self.args
        assert args.document_encoder != 'ar2', 'Not support ar2 in training now; please modify the getitem here to enable tokenizing query with two tokenizers.'

        if args.dataset == 'nq_dpr':
            encoded = self.dataset[index]
            source_ids = encoded[:self.input_length]
            src_mask = encoded[self.input_length:self.input_length * 2]
            oldid = encoded[-1].item()
        else:
            query, oldid = self.dataset.loc[index]
            oldid = int(oldid)

        if args.codebook:
            newid = self.model.pq_mapping[oldid]
        else:
            newid = self.mapping[oldid]

        aug_query_list = []
        if args.aug_query:
            if self.task == 'train':
                if args.aug_query:
                    if args.aug_query_type == 'aug_query':
                        if newid in self.doc_to_query_list:
                            aug_query_list = self.doc_to_query_list[newid]
                    else:
                        for i in range(10):
                            aug_query_list.append(augment(query))
            else:  # do not need for inference
                if args.aug_query:
                    for i in range(20):
                        aug_query_list.append(augment(query))
        neg_docid_list = []
        if args.hard_negative:
            for i in range(1):
                neg_docid_list.append(random_shuffle(
                    newid, args.kary if args.kary else 10))
        target, neg_target, aug_query = newid, neg_docid_list, aug_query_list

        query_embedding = torch.tensor([0])
        prefix_embedding, prefix_mask = torch.tensor([0]), torch.tensor([0])

        if hasattr(self, 'query_dict') and self.query_dict is not None:
            query_embedding = self.q_emb[self.query_dict[query]]
        neg_targets_list = []
        if self.args.hard_negative:
            neg_targets_list = np.random.choice(
                neg_target, self.args.sample_neg_num)
        if self.args.aug_query and len(aug_query) >= 1:
            aug_query = np.random.choice(aug_query, 1)[0]
        else:
            aug_query = ""
        if not self.args.codebook and self.args.label_length_cutoff:
            target = target[:self.args.max_output_length - 2]

        if args.dataset != 'nq_dpr':
            source = self.convert_to_features(
                query, self.input_length if self.task == 'train' else self.inf_input_length)
            source_ids = source["input_ids"].squeeze()
            if 'print_token' in self.args.query_type:
                print("Input Text: ", query, '\n', "Output Text: ", source_ids)
            src_mask = source["attention_mask"].squeeze()
        aug_source = self.convert_to_features(
            aug_query, self.input_length if self.task == 'train' else self.inf_input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()

        def target_to_prefix_emb(target, tgt_length):
            tgt_prefix_emb = []
            prefix_masks = []
            for i in range(tgt_length):
                if i < len(target):
                    # fake data
                    _prefix_emb = np.random.rand(10, 768)
                    # real data
                    # _prefix_emb = self.prefix_embedding[self.prefix2idx_dict[target[:i]]]
                    _prefix_emb = torch.tensor(_prefix_emb)
                    tgt_prefix_emb.append(_prefix_emb.unsqueeze(0))
                    ##############################
                    # fake data
                    _prefix_mask = np.random.rand(10,)
                    _prefix_mask[_prefix_mask < 0.5] = 0
                    _prefix_mask[_prefix_mask > 0.5] = 1
                    # real data
                    # _prefix_mask = self.prefix_mask[self.prefix2idx_dict[target[:i]]]
                    _prefix_mask = torch.LongTensor(_prefix_mask)
                    prefix_masks.append(_prefix_mask.unsqueeze(0))
                    ##############################
                else:
                    tgt_prefix_emb.append(torch.zeros((1, 10, 768)))
                    prefix_masks.append(torch.zeros((1, 10)))
            return torch.cat(tgt_prefix_emb, dim=0), torch.cat(prefix_masks, dim=0)

        if self.prefix_embedding is not None:
            if self.args.multiple_decoder:
                target_ids, target_mask = [], []
                for i in range(self.args.decoder_num):
                    targets = self.convert_to_features(
                        target[i], self.output_length)
                    target_ids.append(targets["input_ids"].squeeze())
                    target_mask.append(targets["attention_mask"].squeeze())
            else:
                targets = self.convert_to_features(target, self.output_length)
                target_ids = targets["input_ids"].squeeze()
                target_mask = targets["attention_mask"].squeeze()
            prefix_embedding, prefix_mask = target_to_prefix_emb(
                target, self.output_length)

        neg_target_ids_list = []
        neg_target_mask_list = []
        neg_rank_list = []

        if self.args.hard_negative:
            for cur_target in neg_targets_list:
                cur_targets = self.convert_to_features(
                    cur_target, self.output_length)
                cur_target_ids = cur_targets["input_ids"].squeeze()
                cur_target_mask = cur_targets["attention_mask"].squeeze()
                neg_target_ids_list.append(cur_target_ids)
                neg_target_mask_list.append(cur_target_mask)
                neg_rank_list.append(999)  # denote hard nagative

        if self.args.decode_embedding and (not self.args.codebook or self.vq_with_label):
            # func target_id+target_id2, twice or k
            def decode_embedding_process(target_id):
                lm_labels = torch.zeros(
                    self.args.max_output_length, dtype=torch.long)
                target_id_int = []
                if self.args.kary:
                    for i in range(0, len(target_id)):
                        c = target_id[i]
                        if args.position:
                            temp = i * args.output_vocab_size + int(c) + 2 \
                                if not args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                else:
                    target_id_int = []
                    bits = int(np.log10(args.output_vocab_size))
                    idx = 0
                    for i in range(0, len(target_id), bits):
                        if i + bits >= len(target_id):
                            c = target_id[i:]
                        c = target_id[i:i + bits]
                        if args.position:
                            temp = idx * args.output_vocab_size + int(c) + 2 \
                                if not args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                        idx += 1
                lm_labels[:len(target_id_int)] = torch.LongTensor(
                    target_id_int)
                lm_labels[len(target_id_int)] = 1
                decoder_attention_mask = lm_labels.clone()
                decoder_attention_mask[decoder_attention_mask != 0] = 1
                target_ids = lm_labels
                target_mask = decoder_attention_mask
                return target_ids, target_mask

            def decode_torch_embedding(target_id):
                kary = self.args.kary
                if kary <= 0:
                    kary = 10
                target_id = target_id + 2
                if args.position:
                    target_id = target_id + \
                        torch.arange(target_id.shape[1]) * kary
                lm_labels = vq_label_suffix(target_id)
                decoder_attention_mask = lm_labels.clone()
                decoder_attention_mask[decoder_attention_mask != 0] = 1
                target_ids = lm_labels
                target_mask = decoder_attention_mask
                return target_ids, target_mask

            if self.model.sample_in_training and not self.args.pq_runtime_label:
                target_ids, target_mask = decode_torch_embedding(
                    self.model.pq_doc_topk_mapping[oldid])
            else:
                if self.args.multiple_decoder:
                    target_mask = []
                    for i in range(args.decoder_num):
                        target_ids[i], cur_target_mask = decode_embedding_process(
                            target[i])
                        target_mask.append(cur_target_mask)
                else:
                    target_ids, target_mask = decode_embedding_process(target)

            if self.args.hard_negative:
                for i in range(len(neg_target_ids_list)):
                    cur_target_ids = neg_target_ids_list[i]
                    cur_target_ids, cur_target_mask = decode_embedding_process(
                        cur_target_ids)
                    neg_target_ids_list[i] = cur_target_ids
                    neg_target_mask_list[i] = cur_target_mask
        elif self.args.codebook and self.task == 'train':
            target_ids = target_mask = []

        results = {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "aug_source_ids": aug_source_ids,
            "aug_source_mask": aug_source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "neg_target_ids": neg_target_ids_list,
            "neg_rank": neg_rank_list,
            "neg_target_mask": neg_target_mask_list,
            "query_emb": query_embedding,
            "prefix_emb": prefix_embedding,
            "prefix_mask": prefix_mask,
        }

        if self.args.document_encoder:
            if self.task == 'train' or self.args.recall_level == 'finesampleloss':
                # doc_ids: str
                if self.model.sample_in_training and self.args.pq_runtime_label:
                    hns = []
                else:
                    hns = self.model.sample_negatives(
                        oldid, newid, query)
                doc_ids = [oldid] + hns
                tensor_doc_ids = torch.tensor(
                    doc_ids, dtype=torch.int32)
                if self.args.codebook and self.args.cluster_position_topk > 0:
                    results["doc_ids"] = tensor_doc_ids
                    if hasattr(self.model, 'co_negclus'):
                        results["topk_labels"] = torch.LongTensor(
                            self.model.co_negclus[query])
                if self.args.fixdocenc:
                    results["doc_ids"] = tensor_doc_ids
                    results["doc_token_ids"] = torch.tensor([0])
                    results["doc_token_mask"] = torch.tensor([0])
                else:
                    if self.model.sample_in_training:
                        results["doc_ids"] = tensor_doc_ids
                    if self.all_docs.dtype is None:
                        doc_contents = self.all_docs[doc_ids].tolist()
                        doc_tokenized = self.tokenize_passage(
                            doc_contents, max_length=self.co_doc_length)
                        results["doc_token_ids"] = doc_tokenized['input_ids']
                        results["doc_token_mask"] = doc_tokenized['attention_mask']
                    else:
                        token, mask = self.all_docs[doc_ids]
                        results["doc_token_ids"] = token
                        results["doc_token_mask"] = mask
            if self.task != 'train':
                results['doc_ids'] = oldid

        return results

    def load_split_data(self, rank, nrank):
        # not drop last; codes from DistributedSampler
        args = self.args
        files = [osp.join(
            args.data_dir, f'{args.dataset}_{args.query_type}_{args.recall_level}_neg{args.co_neg_from}_{i}_{nrank}_{self.task}.tsv') for i in range(nrank)]
        others = osp.join(
            args.data_dir, f'{args.dataset}_{args.query_type}_{args.recall_level}_neg{args.co_neg_from}_{self.task}_others.pkl')
        if rank == 0:
            if not osp.isfile(others) or any([not osp.isfile(f) for f in files]):
                if self.task == 'train':
                    df, self.doc_to_query_list, self.q_emb, self.query_dict, \
                        self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = \
                        load_data(args)
                    with open(others, 'wb') as fw:
                        pickle.dump((self.doc_to_query_list, self.q_emb, self.query_dict,
                                    self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict), fw)
                else:
                    df = load_data_infer(args)
                num_samples = math.ceil(len(df) / nrank)
                total_size = num_samples * nrank
                g = torch.Generator()
                g.manual_seed(args.seed)
                indices = torch.randperm(len(df), generator=g).tolist()

                padding_size = total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size /
                                len(indices)))[:padding_size]
                assert len(indices) == total_size
                for i in range(nrank):
                    # subsample
                    cur_indices = indices[i:total_size:nrank]
                    assert len(cur_indices) == num_samples
                    cur_df = df.loc[cur_indices]
                    cur_df.reset_index(drop=True, inplace=True)
                    cur_df.to_csv(
                        files[i], sep='\t', header=False, index=False, encoding='utf-8')
                    del cur_df
                del df
        self.model.barrier()

        self.dataset = pd.read_csv(
            files[rank],
            encoding='utf-8',
            names=["query", "oldid"],
            header=None,
            sep='\t',
            dtype={'query': str, 'oldid': int}
        )
        if self.task == 'train':
            with open(others, 'rb') as fr:
                self.doc_to_query_list, self.q_emb, self.query_dict, self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = pickle.load(
                    fr)
        else:
            self.doc_to_query_list, self.q_emb, self.query_dict, self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = None, None, None, None, None, None

    def release_dataset(self):
        if not self.released:
            del self.dataset
            del self.mapping
            if hasattr(self, 'doc_to_query_list'):
                del self.doc_to_query_list
        self.released = True

    def reload_dataset(self, rank, nrank):
        if self.released:
            self.load_dataset(self.num_samples, rank, nrank)

    def filter_dataset(self, co_neg):
        ori_length = len(self.dataset)
        self.dataset = self.dataset.loc[[
            q in co_neg for q in self.dataset['query']]]
        self.dataset.reset_index(drop=True, inplace=True)
        print(
            f'Change training dataset length from {ori_length} to {len(self.dataset)}, with {len(co_neg)} co_negs.')
        if hasattr(self, 'doc_to_query_list') and self.doc_to_query_list is not None:
            doc_to_query_list = defaultdict(set)
            for [query, docid] in self.dataset.values.tolist():
                doc_to_query_list[docid].add(query)
            self.doc_to_query_list = doc_to_query_list


class l1_query_eval(l1_query):
    def __getitem__(self, index):
        args = self.args

        newid = None
        query, oldids = self.dataset.loc[index]

        if args.dataset != 'nq_dpr':
            def omap(oldid): return int(oldid)
            oldids = list(map(omap, oldids))

            if args.codebook:
                def onmap(oldid): return self.model.pq_mapping[oldid]
            else:
                def onmap(oldid): return self.mapping[oldid]
            newids = list(map(onmap, oldids))

            targets = newids

            if not self.args.codebook and self.args.label_length_cutoff:
                targets = [target[:self.args.max_output_length - 2]
                           for target in targets]

            if self.model.use_pq_topk_label:
                new_ids = [self.model.pq_doc_topk_mapping[oldid].tolist()
                           for oldid in oldids]
            else:
                new_ids = list(map(list, targets))

        source = self.convert_to_features(
            query, self.input_length if self.task == 'train' else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        if 'print_token' in self.args.query_type:
            print("Input Text: ", query, '\n', "Output Text: ", source_ids)
        src_mask = source["attention_mask"].squeeze()

        if args.codebook:
            onelength = args.subvector_num + 1
        else:
            onelength = args.label_length_cutoff + 1
        target_mask = [1 for _ in range(onelength)] + [0]
        target_mask = torch.LongTensor(target_mask)

        results = {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_mask": target_mask,
            "query": query,
        }
        if args.document_encoder not in (None, 'ance'):
            qenc_source = self.tokenize_passage([query], 32)
            results['qenc_source_ids'] = qenc_source['input_ids'].squeeze()
            results['qenc_source_mask'] = qenc_source['attention_mask'].squeeze()
        if self.args.dataset == 'nq_dpr':
            results['query_indices'] = index
        else:
            results.update({
                "doc_ids": oldids,
                "new_ids": new_ids,
            })

        if self.args.document_encoder:
            if self.args.recall_level == 'finesampleloss':
                # doc_ids: str
                oldid = random.sample(oldids, 1)
                if self.model.sample_in_training and self.args.pq_runtime_label:
                    hns = []
                else:
                    newid = random.sample(newids, 1)
                    hns = self.model.sample_negatives(
                        oldid, newid, query)
                doc_ids = [oldid] + hns
                tensor_doc_ids = torch.tensor(
                    doc_ids, dtype=torch.int32)
                if self.args.fixdocenc:
                    results["doc_ids"] = tensor_doc_ids
                    results["doc_token_ids"] = torch.tensor([0])
                    results["doc_token_mask"] = torch.tensor([0])
                else:
                    if self.model.sample_in_training:
                        results["doc_ids"] = tensor_doc_ids
                    if self.all_docs.dtype is None:
                        doc_contents = self.all_docs[doc_ids].tolist()
                        doc_tokenized = self.tokenize_passage(
                            doc_contents, max_length=self.co_doc_length)
                        results["doc_token_ids"] = doc_tokenized['input_ids']
                        results["doc_token_mask"] = doc_tokenized['attention_mask']
                    else:
                        token, mask = self.all_docs[doc_ids]
                        results["doc_token_ids"] = token
                        results["doc_token_mask"] = mask

        return results


class VariableBatchSizeSamplerWithinEpoch(object):
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        assert len(batch_size) == 2
        # batch_size: tuple of tuple; e.g. （(5, 256), （1, 16))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.nsample_per_loop = sum(
            [n * b for n, b in batch_size])
        self.nbatch_per_loop = sum([x[0] for x in batch_size])
        self.nloop = len(sampler) // self.nsample_per_loop
        self.nrest = len(sampler) % self.nsample_per_loop
        self.min_bs = min([x[1] for x in batch_size])
        self.max_bs = max([x[1] for x in batch_size])
        self.bss = [self.min_bs, self.max_bs]
        assert self.min_bs != self.max_bs, 'If the same batch size, can be optimized together!!!'
        self.loop = sum([[b] * n for n, b in batch_size], [])
        ninloop = self.nloop * self.nbatch_per_loop
        idx = 0
        nrest = self.nrest
        while nrest >= self.loop[idx]:
            nrest -= self.loop[idx]
            idx += 1
        self.length = ninloop + idx
        if not self.drop_last and nrest > 0:
            self.length += 1
        self.epoch = 0

    def __iter__(self):
        batch = []
        batch_idx = 0
        cur_bs = self.loop[batch_idx]
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == cur_bs:
                yield batch
                batch = []
                batch_idx = (batch_idx + 1) % self.nbatch_per_loop
                cur_bs = self.loop[batch_idx]
        while len(batch) > 0:
            yield batch[:cur_bs]
            batch = batch[cur_bs:]
            batch_idx = (batch_idx + 1) % self.nbatch_per_loop
            cur_bs = self.loop[batch_idx]
        self.epoch += 1
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self.epoch)

    def __len__(self):
        return self.length


class VariableBatchNumberSamplerCrossEpoch(object):
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.nparts = len(batch_size)
        assert self.nparts == 2
        # batch_size: tuple of tuple; e.g. （(1, 256), （0.1, 16))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = []
        self.lengths = []
        self.bs = []
        for rate, bs in self.batch_size:
            self.bs.append(bs)
            ns = math.ceil(len(sampler) * rate)
            self.num_samples.append(ns)
            if drop_last:
                func = math.floor
            else:
                func = math.ceil
            self.lengths.append(func(ns / bs))
        self.epoch = 0

    def __iter__(self):
        batch = []
        idx = self.epoch % self.nparts
        cur_bs = self.bs[idx]
        cur_nsample = self.num_samples[idx]
        cnt = 0
        for idx in self.sampler:
            if cnt == cur_nsample:
                break
            batch.append(idx)
            cnt += 1
            if len(batch) == cur_bs:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        self.epoch += 1
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self.epoch)

    def __len__(self):
        return max(self.lengths)


class IndexedData(object):
    def __init__(self, content, torch_dtype=torch.int32):
        if not isinstance(content, tuple):
            content = (content,)
        length = None
        for c in content:
            if length is None:
                length = len(c)
            else:
                assert length == len(c)
        self.length = length
        self.content = content
        self.dtype = torch_dtype

    def __getitem__(self, keys):
        results = [c[keys] for c in self.content]
        if self.dtype is not None:
            results = [torch.tensor(r, dtype=self.dtype) for r in results]
        if len(results) == 1:
            results = results[0]
        return results

    def __len__(self):
        return self.length


def get_ranks(ntopk, pred_clusters, gt_clusters, fill_func, dtype):
    pred_shape = pred_clusters.shape
    gt_shape = gt_clusters.shape
    if len(pred_shape) == 3:
        assert len(
            gt_shape) == 3, 'not support eval all documents now!'
    else:
        pred_clusters = pred_clusters.unsqueeze(0)
        gt_clusters = gt_clusters.unsqueeze(0).unsqueeze(0)
    assert len(pred_clusters.shape) == len(gt_clusters.shape) == 3
    length = gt_clusters.shape[-1]
    pred_shape = pred_clusters.shape
    assert pred_shape[-2] == ntopk and pred_shape[-1] == length
    results = ((gt_clusters.unsqueeze(2) == pred_clusters.unsqueeze(
        1)).sum(-1) == length)  # (bs, n, topk)
    results = results.view(-1, results.shape[-1])
    ranks = torch.zeros(
        gt_clusters.shape[:2], dtype=dtype, device=results.device).view(-1)
    indices, positions = results.nonzero(as_tuple=True)
    ranks.scatter_(0, indices, fill_func(positions))
    return ranks


def get_cluster_index(args, clusters):
    indices = clusters[..., 0]
    for i in range(1, args.subvector_num):
        indices = indices * args.subvector_num
        indices = indices + clusters[..., i]
    return indices


class PassageEmbeddingProjection(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if args.cluster_position_embedding == 'emb':
            self.embedding = torch.nn.Embedding(
                args.cluster_position_topk + 1, args.d_model, padding_idx=0)
            start_dim = args.d_model * 2
        elif args.cluster_position_embedding == 'scorerank':
            start_dim = args.d_model + 2
        else:
            start_dim = args.d_model + 1
        if args.cluster_position_proj_style == 'dense':
            self.projection = torch.nn.Linear(
                start_dim, args.d_model)
        elif args.cluster_position_embedding != 'emb' or args.cluster_position_proj_style != 'add':
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(start_dim, args.d_ff),
                torch.nn.ReLU(),
                torch.nn.Linear(args.d_ff, args.d_model),
            )
        self.cuda_gt_clusters = None
        self.hasrank = args.cluster_position_embedding != 'score'
        self.hasscore = args.cluster_position_embedding.startswith('score')

    def get_logprobs(self, gt_clusters, logits, logprobs=None):
        args = self.args
        if logprobs is None:
            if self.cuda_gt_clusters is not None:
                gt_clusters = self.cuda_gt_clusters
            elif not isinstance(gt_clusters, torch.Tensor):
                gt_clusters = torch.LongTensor(gt_clusters).cuda()
            logits = logits.detach()
            logits = F.log_softmax(logits, -1)
            logprobs = logits.gather(-1, gt_clusters.transpose(1, 2))
            logprobs = torch.sum(logprobs, dim=1).view(-1)
        else:
            logprobs = logprobs * \
                (args.subvector_num + 1) ** args.length_penalty
            logprobs = torch.FloatTensor((logprobs,)).cuda()
        return logprobs

    def forward(
        self,
        doc_embeddings,
        pred_clusters=None,
        gt_clusters=None,
        ranks=None,
        logits=None,
        logprobs=None,
    ):
        # direcly use gt_clusters as labels
        args = self.args
        ntopk = args.cluster_position_topk
        assert len(doc_embeddings.shape) == 2
        if self.hasrank:
            if ranks is None:
                if not isinstance(gt_clusters, torch.Tensor):
                    gt_clusters = torch.LongTensor(gt_clusters).cuda()
                    self.cuda_gt_clusters = gt_clusters
                if args.cluster_position_embedding == 'emb':
                    dtype = torch.long
                    def fill_func(x): return x + 1
                else:
                    dtype = torch.float32
                    if args.cluster_position_rank_reciprocal:
                        def fill_func(x): return 1 / (x + 1)
                    else:
                        def fill_func(x): return (ntopk - x) / ntopk
                ranks = get_ranks(ntopk, pred_clusters,
                                  gt_clusters, fill_func, dtype)
            else:
                assert isinstance(ranks, int)
                if args.cluster_position_embedding == 'emb':
                    ranks += 1
                    ranks = torch.LongTensor((ranks,)).cuda()
                else:
                    if args.cluster_position_rank_reciprocal:
                        ranks = 1 / (ranks + 1)
                    else:
                        ranks = (ntopk - ranks) / ntopk
                    ranks = torch.FloatTensor((ranks,)).cuda()
            if args.cluster_position_embedding == 'emb':
                ranks = self.embedding(ranks)
            else:
                ranks = ranks.unsqueeze(-1)
        if self.hasscore:
            logprobs = self.get_logprobs(gt_clusters, logits, logprobs)

        if args.cluster_position_embedding == 'emb' and args.cluster_position_proj_style == 'add':
            doc_embeddings = doc_embeddings + ranks
        else:
            cat_contents = [doc_embeddings]
            if self.hasrank:
                if ranks.shape[0] != doc_embeddings.shape[0]:
                    ranks = ranks.expand(doc_embeddings.shape[0], -1)
                cat_contents.append(ranks)
            if self.hasscore:
                if logprobs.shape[0] != doc_embeddings.shape[0]:
                    logprobs = logprobs.expand(doc_embeddings.shape[0])
                logprobs = logprobs.unsqueeze(-1)
                cat_contents.append(logprobs)
            doc_embeddings = self.projection(
                torch.cat(cat_contents, -1))
        self.cuda_gt_clusters = None
        return doc_embeddings


class UnifiedEmbeddingProjection(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        ntopk = args.cluster_position_topk
        num_clusters = (args.subvector_bits ** 2) ** args.subvector_num
        if args.cluster_adaptor_trainable_token_embedding:
            self.token_embedding_layer = torch.nn.Embedding(
                num_clusters, args.d_model)
        self.position_embedding_layer = torch.nn.Embedding(
            1 + ntopk, args.d_model, padding_idx=0)
        if not args.cluster_adaptor_trainable_position_embedding:
            pe = torch.zeros(1 + ntopk, args.d_model)
            position = torch.arange(0, ntopk).unsqueeze(1)
            div_term = torch.exp(torch.arange(
                0, args.d_model, 2) * -(math.log(10000.0) / args.d_model))
            pe[1:, 0::2] = torch.sin(position * div_term)
            pe[1:, 1::2] = torch.cos(position * div_term)
            self.position_embedding_layer.requires_grad_(False)
            self.position_embedding_layer.weight.copy_(pe)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=args.d_model, nhead=args.cluster_adaptor_head_num)
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, args.cluster_adaptor_layer_num)
        self.relu = torch.nn.ReLU()

    def encode_query(self, query_embeddings, pred_clusters):
        args = self.args

        cluster_indices = get_cluster_index(args, pred_clusters)
        if args.cluster_adaptor_trainable_token_embedding:
            token_embeddings = self.token_embedding_layer(cluster_indices)
        else:
            token_embeddings = self.all_reconstruct[cluster_indices].cuda()

        ranks = torch.arange(
            pred_clusters.shape[1], device=pred_clusters.device, dtype=torch.long)
        position_embeddings = self.position_embedding_layer(ranks + 1)
        cluster_embeddings = token_embeddings + position_embeddings
        if len(query_embeddings.shape) == 2:
            query_embeddings = query_embeddings.unsqueeze(1)
        cluster_embeddings = torch.cat(
            (query_embeddings, cluster_embeddings), dim=1)
        cluster_embeddings = self.encoder(src=cluster_embeddings)
        cluster_embeddings = torch.amax(
            cluster_embeddings, axis=1, keepdim=True)
        result = cluster_embeddings + query_embeddings
        return result[:, 0, :]

    def encode_passage(
        self,
        doc_embeddings,
        pred_clusters,
        gt_clusters,
        ranks=None,
    ):
        args = self.args

        if not isinstance(gt_clusters, torch.Tensor):
            gt_clusters = torch.LongTensor(gt_clusters).cuda()

        cluster_indices = get_cluster_index(args, gt_clusters)
        #print("passage cluster_indices", cluster_indices)
        if args.cluster_adaptor_trainable_token_embedding:
            token_embeddings = self.token_embedding_layer(cluster_indices)
        else:
            token_embeddings = self.all_reconstruct[cluster_indices].cuda()
        token_embeddings = token_embeddings.view(doc_embeddings.shape)

        cluster_embeddings = token_embeddings
        if len(doc_embeddings.shape) == 2:
            doc_embeddings = doc_embeddings.unsqueeze(1)
            cluster_embeddings = cluster_embeddings.unsqueeze(1)
        cluster_embeddings = torch.cat(
            (doc_embeddings, cluster_embeddings), dim=1)
        cluster_embeddings = self.encoder(src=cluster_embeddings)
        result = cluster_embeddings + doc_embeddings
        return result[:, 0, :]


class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train=True):
        super(T5FineTuner, self).__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        if args.timing_step > 0:
            # timing encoder and decoder
            self.timer = {'e': [], 'd': []}
        elif args.timing_infer_step > 0:
            self.timing_step_for_infer = 0
            self.timer = {'nci': [], 'knn': []}
        else:
            self.timer = None
        self.cur_ep = 0  # different from current_epoch; record epochs in all fit
        self.synced = False

        # some simplified flag
        self.sample_in_training = args.codebook and (
            args.pq_loss in ('emdr2', 'adist') or args.topk_sequence or (args.doc_multiclus > 1 and args.multiclus_label != 'top1'))
        self.nrets = args.doc_multiclus if args.doc_multiclus > 1 else args.topk_sequence if args.topk_sequence > 0 else args.aug_topk_clus
        self.topk_from_vq = self.sample_in_training and args.pq_runtime_label and (
            args.topk_sequence or args.pq_loss == 'adist' or args.aug_find_topk_from == 'vq')
        self.alt_cross_epoch = args.nci_twin_train_ratio and args.alt_granularity == 'epoch'
        self.qemb_from_ckpt = args.reserve_decoder and args.qtower == 'ori'
        self.get_codebook_logits = args.codebook and (self.sample_in_training or args.doc_multiclus > 1 or (
            (not args.query_vq_label and args.pq_runtime_label) or args.pq_loss != 'label') or args.centroid_update_loss != 'none')
        if args.codebook and args.reconstruct_for_embeddings:
            assert self.get_codebook_logits and not self.sample_in_training
        self.additional_reconstruct = args.codebook and args.use_topic_model and args.topic_score_ratio > 0
        self.use_pq_topk_label = args.codebook and (
            (not args.pq_runtime_label and self.sample_in_training) or args.doc_multiclus > 1)

        # bulid tree
        if args.tree_path is None:
            if args.codebook:
                tree_save_file = f'pq_tree{args.subvector_num}_{args.subvector_bits}.pkl'
            elif args.document_encoder:
                tree_save_file = f'mevi_tree{args.max_output_length}_{args.newid_suffix}.pkl'
            else:
                tree_save_file = f'tree{args.max_output_length}_{args.newid_suffix}.pkl'
            if args.kary:
                items = tree_save_file.split('.')
                tree_save_file = items[0] + f'_{args.kary}.' + items[1]
            tree_save_path = osp.join(args.data_dir, tree_save_file)
        else:
            tree_save_path = args.tree_path
        if osp.isfile(tree_save_path):
            with open(tree_save_path, "rb") as input_file:
                root = pickle.load(input_file)
            self.root = root
        else:
            self.build_tree(args, tree_save_path)

        # paths = []
        # path = []

        # def dfs(node):
        #     path.append(node.token_id)
        #     if node.children == {}:
        #         paths.append(path.copy())
        #     for ch in node.children.values():
        #         dfs(ch)
        #     path.pop()
        # dfs(self.root)
        # print(paths)
        # print(len(paths))
        # exit()

        self.args = args
        self.args.query_embed_accum = self.args.query_embed_accum.lower()
        if isinstance(self.args.qtower, str):
            self.args.qtower = self.args.qtower.split('_')
        if self.args.query_embed_accum == 'attenpool':
            self.attenpool_weight = torch.nn.Linear(args.d_model, 1)
        else:
            self.attenpool_weight = None
        assert not args.codebook or args.cat_cluster_centroid <= 0 or args.cluster_position_topk <= 0
        if args.codebook and args.cat_cluster_centroid > 0:
            self.qemb_projection = torch.nn.Linear(
                args.d_model * (1 + args.cat_cluster_centroid), args.d_model)
        else:
            self.qemb_projection = None
        self.unified_projection = None
        self.pemb_projection = None
        if args.codebook and args.cluster_position_topk > 0:
            assert args.cluster_position_topk <= args.num_return_sequences
            if args.use_cluster_adaptor:
                self.unified_projection = UnifiedEmbeddingProjection(args)
            else:
                self.pemb_projection = PassageEmbeddingProjection(args)

        self.save_hyperparameters(args)
        # assert args.tie_word_embedding is not args.decode_embedding
        if args.decode_embedding:
            if self.args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                if args.kary:
                    self.decode_vocab_size = args.kary * expand_scale + 2
                else:
                    self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                if args.kary:
                    self.decode_vocab_size = args.kary + 2
                else:
                    self.decode_vocab_size = 12
        else:
            self.decode_vocab_size = None

        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=0 if args.softmax else args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size,
            output_vocab_size=args.output_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            contrastive=args.contrastive,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num=args.adaptor_layer_num,
            embedding_distillation=args.embedding_distillation,
            weight_distillation=args.weight_distillation,
            input_dropout=args.input_dropout,
            denoising=args.denoising,
            multiple_decoder=args.multiple_decoder,
            decoder_num=args.decoder_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
            use_codebook=args.codebook,
            pq_loss=args.pq_loss,
            pq_twin_loss=args.pq_twin_loss,
            reserve_decoder=args.reserve_decoder,
            decoder_integration=args.decoder_integration,
            topk_minpooling=args.doc_multiclus if args.doc_multiclus > 1 and args.multiclus_label == 'minpool' else None,
        )
        model = T5ForConditionalGeneration(t5_config)
        if args.pretrain_encoder:
            pretrain_model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path)
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
        if args.use_ort == 1:
            from onnxruntime.training.ortmodule import ORTModule, LogLevel, DebugOptions
            self.model = ORTModule(model)
        else:
            self.model = model
        if args.fp16_opt:
            os.environ["FP16_OPT"] = "true"
        self.tokenizer = T5Tokenizer.from_pretrained(
            args.tokenizer_name_or_path)
        # self.rouge_metric = load_metric('rouge')

        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        if self.args.softmax:
            # [feature size, num cls]
            self.fc = torch.nn.Linear(args.d_model, self.args.num_cls)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.5)
        if self.args.disc_loss:
            self.dfc = torch.nn.Linear(args.d_model, 1)

        n_observations_per_split = {
            "train": self.args.n_train,
            "validation": self.args.n_val,
            "test": self.args.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k,
                      v in n_observations_per_split.items()}

        if not args.codebook:
            with open(args.mapping_path, 'rb') as fr:
                self.mapping = pickle.load(fr)
        else:
            self.mapping = None

        # document_encoder
        if args.document_encoder:
            if self.args.co_loss_scale:
                items = [float(it)
                         for it in self.args.co_loss_scale.split(':')]
                if len(items) == 1:
                    self.co_loss_scale = items
                elif len(items) == 3:
                    self.co_loss_scale = np.arange(*items).tolist()
                else:
                    assert False, 'Bad argument format for co_loss_scale.'
            else:
                self.co_loss_scale = None

            self.init_document_encoder()
            if self.args.query_encoder == 'nci' and self.args.tie_encoders:
                self.tie_twin_tower()
            if self.args.fixdocenc:
                self.freeze_document_encoder()
            if self.args.fixnci:
                self.freeze_params(self.model)
                if self.attenpool_weight is not None:
                    self.freeze_params(self.attenpool_weight)
            if self.args.fixlmp:
                self.freeze_params(self.document_encoder.lm_p)
            if self.args.fixlmq:
                self.freeze_params(self.document_encoder.lm_q)
            if args.document_path.endswith('.tsv'):
                if args.dataset in ('marco', 'nq_dpr'):
                    all_docs = pd.read_csv(osp.join(args.document_path), sep='\t', names=[
                        'odid', 'title', 'content'], dtype={'odid': str, 'title': str, 'content': str})
                    all_docs.fillna('', inplace=True)
                    if args.document_encoder == 'ance':
                        all_docs = pd.concat(
                            (all_docs['odid'], 'Title: ' + all_docs['title'] + ' Text: ' + all_docs['content']), axis=1)
                    else:
                        all_docs = pd.concat(
                            (all_docs['odid'], all_docs['title'] + self.passage_tokenizer.sep_token + all_docs['content']), axis=1)
                    all_docs.columns.values[1] = 'content'
                self.all_docs = IndexedData(
                    content=all_docs['content'], torch_dtype=None)
            else:
                suffixes = ['_tokens.bin', '_masks.bin']
                if args.drop_data_rate > 0:
                    suffixes = [
                        f'_tokens_drop{args.drop_data_rate}.bin', f'_masks_drop{args.drop_data_rate}.bin']
                token_file, mask_file = [
                    args.document_path + suffix for suffix in suffixes]

                all_tokens = np.memmap(
                    token_file, dtype=np.int64, mode='r').reshape(-1, args.co_doc_length)
                all_masks = np.memmap(
                    mask_file, dtype=np.int64, mode='r').reshape(-1, args.co_doc_length)

                self.all_docs = IndexedData(content=(all_tokens, all_masks))

            if args.codebook:
                self.doc_cluster = None
                from pq import ProductQuantization
                self.pq = ProductQuantization(
                    args.pq_type,
                    subvector_num=args.subvector_num,
                    subvector_bits=args.subvector_bits,
                    dist_mode=args.pq_dist_mode,
                    emb_size=args.d_model,
                    pq_init_method=args.pq_init_method,
                    pq_update_method=args.pq_update_method,
                    tie_nci_pq_centroid=args.tie_nci_pq_centroid,
                    lm_head=self.model.lm_head.weight,
                    centroid_update_loss=args.centroid_update_loss,
                    rq_topk_score=args.rq_topk_score,
                )
                if self.args.fixpq:
                    self.pq.fix()
            else:
                with open(self.args.cluster_path, 'rb') as fr:
                    self.doc_cluster = pickle.load(fr)
                self.doc_cluster = {k: [int(vv) for vv in v]
                                    for k, v in self.doc_cluster.items()}
                if not args.kary:
                    self.doc_cluster = pd.Series(self.doc_cluster)
            if args.nci_twin_train_ratio:
                # e.g. 5:256,1:16
                if args.alt_granularity == 'batch':
                    self.train_ratios = tuple(tuple(int(x) for x in part.split(
                        ':')) for part in args.nci_twin_train_ratio.split(','))
                    self.bs_stage = {b[1]: i for i,
                                     b in enumerate(self.train_ratios)}
                else:
                    self.stage = 0
                    self.train_ratios = tuple(tuple(float(x) if i == 0 else int(x) for i, x in enumerate(part.split(
                        ':'))) for part in args.nci_twin_train_ratio.split(','))
                    if args.nci_twin_alt_epoch:
                        eps = [int(x)
                               for x in args.nci_twin_alt_epoch.split(',')]
                        eps[1] += eps[0]
                    else:
                        eps = [1, 2]
                    self.nci_twin_alt_epoch = tuple(eps)
            if args.nci_vq_alt_epoch:
                eps = [int(x) for x in args.nci_vq_alt_epoch.split(',')]
                eps[1] += eps[0]
                self.nci_vq_alt_epoch = tuple(eps)
                if not args.fixdocenc:
                    print(
                        '[Warning] Document encoder is not fixed in nci and vq alterative training.')
                # args.fixdocenc = True
            else:
                self.nci_vq_alt_epoch = None
            self.all_candidates = range(len(self.all_docs))
        else:
            self.document_encoder = None
            self.passage_tokenizer = None
            self.all_docs = None
            if args.dataset == 'nq_dpr':
                with open(args.cluster_path, 'rb') as fr:
                    self.doc_cluster = pickle.load(fr)
            else:
                self.doc_cluster = None

        if train:
            train_n_samples = self.n_obs['train']
            val_n_samples = self.n_obs['validation']
            if self.trainer is None:
                rank = 0
                nrank = len(self.args.n_gpu)
            else:
                rank = self.rank
                nrank = self.nrank
            self.train_dataset = l1_query(
                self,
                self.args,
                self.tokenizer,
                self.passage_tokenizer,
                train_n_samples,
                self.mapping,
                all_docs=self.all_docs,
                doc_cluster=self.doc_cluster,
                rank=rank,
                nrank=nrank
            )
            self.val_dataset = l1_query_eval(
                self,
                self.args,
                self.tokenizer,
                self.passage_tokenizer,
                val_n_samples,
                self.mapping,
                task='test',
                all_docs=self.all_docs,
                doc_cluster=self.doc_cluster,
                rank=rank,
                nrank=nrank
            )
            self.t_total = (
                (len(self.train_dataset) //
                 (self.args.train_batch_size * max(1, len(self.args.n_gpu))))
                // self.args.gradient_accumulation_steps
                * float(self.args.num_train_epochs)
            )

        if args.document_encoder:
            if args.mode == 'train' and args.co_neg_from not in ('clus', 'notclus', 'clusfile'):
                if args.co_neg_file.endswith('.tsv'):
                    co_negs = {}
                    with open(args.co_neg_file, 'r') as fr:
                        for line in tqdm(fr, desc='Read neg file.'):
                            items = line.strip().split('\t')
                            q = items[0]
                            if args.co_neg_from.startswith('simans'):
                                gt_scores = [float(sc)
                                             for sc in items[-3].split(',')]
                                nn = [int(nnn) for nnn in items[-2].split(',')]
                                scores = [float(sss)
                                          for sss in items[-1].split(',')]
                                probs = [-args.simans_hyper_a *
                                         (sss - sum(gt_scores) / len(gt_scores) - args.simans_hyper_b) ** 2 for sss in scores]
                                probs = softmax(probs)
                                nn = (nn, probs)
                            else:
                                nn = [int(nnn) for nnn in items[-1].split(',')]
                            assert co_negs.get(q, None) in (None, nn)
                            co_negs[q] = nn

                    self.co_negs = co_negs
                else:
                    with open(args.co_neg_file, 'rb') as fr:
                        self.co_negs = pickle.load(fr)
                self.train_dataset.filter_dataset(self.co_negs)
            if args.co_neg_from.endswith('clusfile') or (args.cluster_position_topk > 0 and args.co_neg_clus_file is not None and osp.exists(args.co_neg_clus_file)):
                if args.co_neg_clus_file.endswith('.tsv'):
                    co_negclus = {}
                    with open(args.co_neg_clus_file, 'r') as fr:
                        for line in tqdm(fr, desc='Read top-k cluster file.'):
                            query, cluses, _ = line.split('\t')
                            cluses = eval(cluses)
                            cluses = [tuple(c) for c in cluses]
                            co_negclus[query] = cluses
                else:
                    with open(args.co_neg_clus_file, 'rb') as fr:
                        co_negclus = pickle.load(fr)
                self.co_negclus = co_negclus
                if args.mode == 'train':
                    self.train_dataset.filter_dataset(self.co_negclus)

    def init_document_encoder(self):
        args = self.args
        from document_encoder import DocumentEncoder
        config_path = None
        if args.document_encoder == 'cocondenser':
            model_name = 'co-condenser-marco-retriever'
            self.passage_tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased', use_fast=True)
        elif args.document_encoder == 'ance':
            # all use t5-ance
            # if args.dataset == 'marco':
            model_name = 't5-ance'
            if args.document_encoder_from_pretrained:
                model_name = f't5-{args.model_info}-scaled'
            self.passage_tokenizer = AutoTokenizer.from_pretrained(
                osp.join(args.ckpt_dir, model_name))
        elif args.document_encoder == 'ar2':
            if args.dataset == 'marco':
                model_name = 'ar2g_marco_finetune.pkl'
                config_path = osp.join(
                    args.ckpt_dir, 'co-condenser-marco-retriever')
            elif args.dataset.startswith('nq'):
                model_name = 'ar2g_nq_finetune.pkl'
                config_path = osp.join(args.ckpt_dir, 'ernie-2.0-base-en')
            else:
                raise NotImplementedError
            self.passage_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True)
        else:
            raise NotImplementedError
        model_path = osp.join(args.ckpt_dir, model_name)
        print(f'Build document encoder using checkpoint from {model_path}.')
        self.document_encoder = DocumentEncoder.build(
            model_path,
            config_path=config_path,
            dropout=0.1 if args.document_encoder == 'ar2' else None,
            tied=(args.query_encoder == 'nci' or args.tie_encoders),
            negatives_x_sample=args.negatives_x_sample,
        )
        if args.query_encoder == 'twin':
            self.tokenizer = self.passage_tokenizer

    def on_fit_start(self):
        args = self.args
        if self.nrank > 1:
            obj_list = [args.time_str]
            dist.broadcast_object_list(obj_list, 0)
            args.time_str = obj_list[0]
        args.metric_path = osp.join(
            args.logs_dir, '{}_metrics_{}.txt'.format(args.tag_info, args.time_str))
        args.inf_save_path = osp.join(args.logs_dir, '{}_{}_{}.tsv'.format(
            args.tag_info, args.num_return_sequences, args.time_str))
        self.synced = True

    def build_tree(self, args, tree_save_path):
        if args.codebook:
            max_depth = args.max_output_length - 2
            builder = TreeBuilder(share_sons=True)
            newids = [encode_single_newid(
                args, [i for _ in range(max_depth)]) for i in range(args.kary)]
            for i in range(max_depth):
                cur_layer_ids = [ids[i] for ids in newids]
                builder.add_layer(cur_layer_ids)
            builder.add_layer([1])
        else:
            builder = TreeBuilder()
            if args.dataset in ('marco', 'nq_dpr'):
                all_path = args.mapping_path
                with open(all_path, 'rb') as fr:
                    df = pickle.load(fr)
                if not args.kary:
                    df = pd.DataFrame(df.values(), columns=["k10_c10"])

            if args.kary and (args.dataset in ('marco', 'nq_dpr')):
                for newid in tqdm(df.values(), desc='Build Tree'):
                    toks = encode_single_newid(args, newid)
                    builder.add(toks)
            else:
                for _, newid in tqdm(df.iterrows(), desc='Build Tree'):
                    if args.dataset in ('marco', 'nq_dpr'):
                        newid = newid.tolist()[0]
                        toks = encode_single_newid(args, newid)
                        builder.add(toks)

        if args.tree:
            root = builder.build()
            if tree_save_path is not None and (self.trainer is None or self.rank <= 0):
                with open(tree_save_path, "wb") as input_file:
                    pickle.dump(root, input_file)
        else:
            print('No Tree')
            root = None
        self.root = root

    def tie_twin_tower(self):
        # tie the parameters of query encoder and document encoder
        args = self.args
        assert args.dataset in (
            'marco', 'nq_dpr') and args.document_encoder == 'ance' and args.reserve_decoder
        docenc = self.document_encoder.lm_p
        self.model.encoder = docenc.encoder
        self.model.ori_decoder = docenc.decoder
        self.model.shared = docenc.shared

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def unfreeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = True

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def freeze_document_encoder(self):
        self.freeze_params(self.document_encoder)
        if self.args.fixproj:
            if self.qemb_projection is not None:
                self.freeze_params(self.qemb_projection)
            if self.pemb_projection is not None:
                self.freeze_params(self.pemb_projection)
            if self.unified_projection is not None:
                self.freeze_params(self.unified_projection)

    def unfreeze_document_encoder(self):
        self.unfreeze_params(self.document_encoder)
        if not self.args.fixproj:
            if self.qemb_projection is not None:
                self.unfreeze_params(self.qemb_projection)
            if self.pemb_projection is not None:
                self.unfreeze_params(self.pemb_projection)
            if self.unified_projection is not None:
                self.unfreeze_params(self.unified_projection)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def codebook_decode_embedding_process(self, target_id):
        args = self.args
        if args.kary:
            if args.position and not args.hierarchic_decode:
                target_id = target_id + \
                    torch.arange(
                        target_id.shape[1], device=target_id.device).unsqueeze(0) * args.output_vocab_size + 2
            else:
                target_id = target_id + 2
        lm_labels = vq_label_suffix(target_id)
        decoder_attention_mask = lm_labels.clone()
        decoder_attention_mask[decoder_attention_mask != 0] = 1
        lm_labels[lm_labels[:, :] ==
                  self.tokenizer.pad_token_id] = -100
        return lm_labels, decoder_attention_mask

    def sample_negatives(self, oldid, newid, query):
        def remove_ele(cands, ele):
            try:
                idx = cands.index(ele)
                cands.pop(idx)
            except:
                pass

        co_neg_num = self.args.co_neg_num
        co_neg_from = self.args.co_neg_from
        if co_neg_from not in ('file', 'simans'):
            if co_neg_from.endswith('clusfile'):
                assert self.args.codebook, 'Workaround.'
                top_clus = self.co_negclus[query]
                nn_candidates_from_clus = sum([self.pq_doc_cluster.get(clus, [])
                                               for clus in top_clus], [])
                remove_ele(nn_candidates_from_clus, oldid)
            elif co_neg_from == 'notclus':
                # if self.args.codebook:
                #     nn_candidates_from_clus = self.pq_doc_cluster.get(
                #         newid, [])
                # else:
                #     key = newid[:self.args.label_length_cutoff]
                #     if not isinstance(key, str):
                #         key = tuple(key)
                #     nn_candidates_from_clus = self.doc_cluster[key]
                # set_cands = set(nn_candidates_from_clus)

                def if_in_clus(c):
                    result = any(
                        [i == j for i, j in zip(self.pq_mapping[c], newid)])
                    return result

                assert self.args.codebook
                new_nn_candidates_from_clus = set()
                while len(new_nn_candidates_from_clus) < co_neg_num:
                    newc = random.sample(self.all_candidates, k=2*co_neg_num)
                    new_nn_candidates_from_clus.update(
                        [c for c in newc if not if_in_clus(c)])
                nn_candidates_from_clus = list(new_nn_candidates_from_clus)
            else:
                if self.args.codebook:
                    nn_candidates_from_clus = self.pq_doc_cluster.get(
                        newid, []).copy()
                    remove_ele(nn_candidates_from_clus, oldid)
                    if len(nn_candidates_from_clus) == 0:
                        def make_key(ind, ele):
                            temp = list(newid)
                            temp[ind] = ele
                            return tuple(temp)
                        other_candidates = [self.pq_doc_cluster.get(make_key(i, e), []) for i in range(
                            self.args.subvector_num) for e in range(2 ** self.args.subvector_bits)]
                        nn_candidates_from_clus = sum(other_candidates, [])
                        remove_ele(nn_candidates_from_clus, oldid)
                else:
                    key = newid[:self.args.label_length_cutoff]
                    if not isinstance(key, str):
                        key = tuple(key)
                    nn_candidates_from_clus = self.doc_cluster[key].copy()
                    remove_ele(nn_candidates_from_clus, oldid)
        if co_neg_from not in ('clus', 'notclus', 'clusfile'):
            nn_candidates_from_file = self.co_negs[query]
        if co_neg_from in ('clus', 'notclus', 'clusfile'):
            nn_candidates = nn_candidates_from_clus
        elif co_neg_from == 'file':
            nn_candidates = nn_candidates_from_file
        elif co_neg_from.startswith('inter'):
            set_clus = set(nn_candidates_from_clus)
            inter_part = set_clus.intersection(nn_candidates_from_file)
            rest_part = set_clus - inter_part
            if co_neg_from in ('inter', 'interclusfile'):
                need_more = co_neg_num - len(inter_part)
                nn_candidates = list(inter_part)
                if need_more > 0:
                    if len(rest_part) >= need_more:
                        nn_candidates = nn_candidates + \
                            random.sample(rest_part, k=need_more)
                    else:
                        nn_candidates = nn_candidates_from_clus
            else:
                half_neg = co_neg_num // 2
                if len(inter_part) == 0:
                    nn_candidates = list(rest_part)
                elif len(rest_part) == 0:
                    nn_candidates = list(inter_part)
                else:
                    def sampling(cands, number):
                        if len(cands) >= number:
                            return random.sample(cands, k=number)
                        else:
                            return random.choices(cands, k=number)
                    nn_candidates = sampling(
                        list(inter_part), half_neg) + sampling(list(rest_part), co_neg_num - half_neg)
        elif co_neg_from == 'union':
            nn_candidates = list(
                set(nn_candidates_from_file).union(nn_candidates_from_clus))
        else:
            nn_candidates, nn_scores = nn_candidates_from_file
            if co_neg_from == 'simansinter':
                set_from_clus = set(nn_candidates_from_clus)
                exists = [
                    nnn for nnn in nn_candidates if nnn in set_from_clus]
                nn_candidates = [
                    nnn for nnn, ex in zip(nn_candidates, exists) if ex]
                nn_scores = [sc for sc, ex in zip(nn_scores, exists) if ex]
            if len(nn_candidates) >= co_neg_num:
                # simans
                nn_candidates = np.random.choice(
                    nn_candidates, size=co_neg_num, replace=False, p=nn_scores).tolist()
            elif co_neg_from == 'simansinter':
                if len(nn_candidates_from_clus) >= co_neg_num:
                    nn_candidates = nn_candidates + \
                        random.sample(nn_candidates_from_clus, k=co_neg_num)
                    nn_candidates = list(set(nn_candidates))
                else:
                    nn_candidates = nn_candidates_from_clus
        nn_candidates = nn_candidates.copy()
        nn_length = len(nn_candidates)
        if nn_length >= co_neg_num:
            if nn_length != co_neg_num:
                if self.args.codebook:
                    hns = random.sample(nn_candidates, k=co_neg_num)
                else:
                    _offset = self.current_epoch * \
                        co_neg_num % nn_length
                    nn_candidates = nn_candidates * 2
                    hns = nn_candidates[_offset: _offset + co_neg_num]
            else:
                hns = nn_candidates
        else:
            if nn_length == 0:
                hns = random.sample(self.all_candidates, k=co_neg_num+1)
                if oldid in hns:
                    hns.remove(oldid)
                hns = hns[:co_neg_num]
            else:
                hns = random.choices(nn_candidates, k=co_neg_num)
        return hns

    def get_document_embedding(self, doc_ids=None, doc_token_ids=None, doc_token_mask=None, encoder_trainable=False):
        doc_embeddings = None
        if doc_ids is not None:
            if isinstance(doc_ids, torch.Tensor):
                doc_ids = doc_ids.cpu().numpy()
            elif isinstance(doc_ids, list):
                doc_ids = np.array(doc_ids)
            doc_ids = doc_ids.reshape(-1)
            if hasattr(self, 'all_embeddings'):
                doc_embeddings = self.all_embeddings[doc_ids].cuda()
            elif doc_token_ids is None:
                assert self.all_docs.dtype is not None
                doc_token_ids, doc_token_mask = self.all_docs[doc_ids]
                doc_token_ids = doc_token_ids.cuda()
                doc_token_mask = doc_token_mask.cuda()
        if doc_embeddings is None or encoder_trainable:
            #print("doc_token_ids", doc_token_ids, doc_token_ids.shape)
            doc_embeddings = self.document_encoder.encode_passage({
                'input_ids': doc_token_ids.view(-1, self.args.co_doc_length),
                'attention_mask': doc_token_mask.view(-1, self.args.co_doc_length),
            })
        return doc_embeddings

    def get_query_embedding(self, input_ids, attention_mask, enc_last_hidden_state, ori_dec_last_hidden_state, dec_last_hidden_state, lm_labels, clusters, flatten=False):
        if self.qemb_from_ckpt:
            qemb = ori_dec_last_hidden_state.squeeze(1)
        elif self.args.query_encoder == 'nci':
            qemb = self.clus_repr(enc_last_hidden_state, attention_mask,
                                  ori_dec_last_hidden_state, dec_last_hidden_state, lm_labels, flatten)
        else:
            #print("self.args.query_encoder", self.args.query_encoder)
            qemb = self.document_encoder.encode_query(
                {'input_ids': input_ids, 'attention_mask': attention_mask})
        if self.qemb_projection is not None:
            #print("enter here")
            clusters = clusters.reshape(-1, clusters.shape[-1])
            #print("clusters", clusters)
            indices = get_cluster_index(self.args, clusters)
            centroids = self.all_reconstruct[indices]
            centroids = centroids.view(qemb.shape[0], -1)
            qemb = torch.cat((qemb, centroids.cuda()), dim=-1)
            qemb = self.qemb_projection(qemb)
        elif self.unified_projection is not None:
            self.unified_projection.encode_query(qemb, clusters)
        return qemb

    def clus_repr(self, enc_last_hidden_state, mask, ori_dec_last_hidden_state, dec_last_hidden_state, lm_labels, flatten=False):
        accum = self.args.query_embed_accum
        qtower = self.args.qtower
        cands = []
        if 'enc' in qtower or 'encmask' in qtower:
            cands.append(enc_last_hidden_state)
        if 'ori' in qtower:
            cands.append(ori_dec_last_hidden_state)
        if 'dec' in qtower:
            cands.append(dec_last_hidden_state)
        if 'emb' in qtower:
            if flatten:
                key = lm_labels
            else:
                key = lm_labels[:, self.args.label_length_cutoff - 1]
            lookup = torch.unsqueeze(
                self.model.decode_embeddings.weight[key], dim=1)
            cands.append(lookup)
        hidden_state = torch.cat(cands, dim=1)
        if 'encmask' in qtower:
            nret = hidden_state.shape[0] // mask.shape[0]
            if nret != 1:
                mask = mask.unsqueeze(
                    1).expand(-1, nret, -1).reshape(-1, mask.shape[-1])
            lcat = hidden_state.shape[1] - mask.shape[1]
            mask = torch.cat((mask, torch.ones(
                (mask.shape[0], lcat), dtype=mask.dtype, device=mask.device)), dim=1)
            hidden_state *= mask.unsqueeze(-1)
            neg_inf = mask.float(). \
                masked_fill(mask == 0, float('-inf')). \
                masked_fill(mask == 1, float(0.0))
        else:
            mask = None
        if accum == 'maxpool':
            if mask is not None:
                hidden_state += neg_inf.unsqueeze(-1)
            query_embedding = torch.max(hidden_state, 1)[0]
        elif accum == 'avgpool':
            if mask is not None:
                query_embedding = torch.sum(
                    hidden_state, 1) / torch.sum(mask, 1, keepdim=True)
            else:
                query_embedding = torch.mean(hidden_state, 1)
        elif accum == 'attenpool':
            scores = self.attenpool_weight(hidden_state)
            if mask is not None:
                scores = scores + neg_inf.unsqueeze(-1)
            atten_w = torch.nn.functional.softmax(
                scores, dim=1)
            query_embedding = torch.sum(hidden_state * atten_w, dim=1)
        return query_embedding

    def compute_pq_loss(self, nci_logits, pq_logits, ori_labels):
        args = self.args
        if args.pq_loss == 'adist':
            nci_proba = nci_logits.gather(-1,
                                          ori_labels.unsqueeze(-1)).squeeze(-1)
            nci_proba = nci_proba.prod(dim=-1).reshape(pq_logits.shape)
            loss = F.kl_div(F.log_softmax(nci_proba, dim=-1),
                            F.softmax(pq_logits, dim=-1), reduction='none')
            loss = torch.mean(loss)
        else:
            if args.pq_negative == 'batch':
                assert args.Rdrop == 0
                pq_logits = pq_logits.view(
                    1, -1, args.subvector_num, nci_logits.size(-1))
                nci_logits = nci_logits.unsqueeze(1)
            elif args.pq_negative == 'sample':
                pq_logits = pq_logits.view(nci_logits.size(
                    0), -1, args.subvector_num, nci_logits.size(-1))
                nci_logits = nci_logits.unsqueeze(1)
            else:
                pq_logits = pq_logits.view(nci_logits.size(
                    0), -1, args.subvector_num, nci_logits.size(-1))[:, 0, :, :].squeeze(1)
            if args.pq_loss in ('bce', 'ce', 'kl'):
                if args.pq_negative != 'none':
                    pq_logits = pq_logits.expand(
                        nci_logits.size(0), -1, -1, -1)
                    nci_logits = nci_logits.expand(-1,
                                                   pq_logits.size(1), -1, -1)
            if args.pq_loss == 'ce':
                with torch.no_grad():
                    nci_softmax = self.get_softmax(nci_logits)
                    pq_softmax = self.get_softmax(pq_logits)
                nci_log_softmax = F.log_softmax(
                    nci_logits / args.pq_softmax_tau, dim=-1)
                pq_log_softmax = F.log_softmax(
                    pq_logits / args.pq_softmax_tau, dim=-1)
                #loss = -nci_softmax * pq_log_softmax - pq_softmax * nci_log_softmax
                #print("nci_softmax", nci_softmax)
                loss = -nci_softmax * pq_log_softmax
            elif args.pq_loss == 'bce':
                assert args.use_gumbel_softmax and args.pq_hard_softmax_topk >= 1
                from torch.nn import BCEWithLogitsLoss
                pq_loss_fct = BCEWithLogitsLoss(reduction='none')
                loss = pq_loss_fct(nci_logits, pq_logits)
            elif args.pq_loss == 'kl':
                left_loss = F.kl_div(
                    torch.log(nci_logits + 1e-12), pq_logits, reduction='none')
                right_loss = F.kl_div(
                    torch.log(pq_logits + 1e-12), nci_logits, reduction='none')
                loss = (left_loss + right_loss) / 2
            elif args.pq_loss == 'mse':
                loss = (pq_logits - nci_logits) ** 2
            elif args.pq_loss == 'dot':
                loss = - pq_logits * nci_logits
            elif args.pq_loss == 'cosine':
                pq_logits = F.normalize(pq_logits, dim=-1)
                nci_logits = F.normalize(nci_logits, dim=-1)
                loss = - pq_logits * nci_logits
            else:
                raise NotImplementedError
            loss = loss.sum(-1)
            if args.pq_negative != 'none':
                loss = torch.mean(loss, dim=2)
                if args.pq_negative == 'batch':
                    target = torch.arange(
                        loss.size(0),
                        device=loss.device,
                        dtype=torch.long
                    )
                    target = target * (loss.size(1) // loss.size(0))
                else:
                    target = torch.zeros(
                        loss.size(0), device=loss.device, dtype=torch.long)
                if args.pq_negative_loss == 'cont':
                    pq_loss_fct = CrossEntropyLoss()
                    loss = pq_loss_fct(loss, target)
                else:
                    # need margin; as triplet loss
                    margin = args.pq_negative_margin
                    pos_loss = loss.gather(1, target.unsqueeze(-1))
                    neg_loss = (torch.sum(loss, dim=-1, keepdim=True) -
                                pos_loss) / (loss.size(1) - 1)
                    loss = pos_loss - \
                        torch.min(neg_loss, torch.full_like(
                            neg_loss, margin)) + margin
                    loss = torch.mean(loss)
            else:
                loss = torch.mean(loss)
        return loss

    def compute_emdr2_loss(self, nci_logits, labels):
        # (bs * nbeam, seqlen, vocab), (bs * nbeam, seqlen)
        nci_logits = nci_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        nci_logits = nci_logits.prod(1).view(-1, self.args.aug_topk_clus)
        nci_logits = F.softmax(nci_logits, -1)
        return nci_logits

    def get_softmax(self, proba):
        args = self.args
        if args.use_gumbel_softmax:
            assign = F.gumbel_softmax(
                proba, tau=args.pq_softmax_tau, hard=False, dim=-1)
        else:
            assign = F.softmax(proba / args.pq_softmax_tau, dim=-1)
        if args.pq_hard_softmax_topk > 0:
            topk_index = assign.topk(args.pq_hard_softmax_topk, dim=-1)[1]
            assign_hard = torch.zeros_like(
                assign, device=assign.device, dtype=assign.dtype).scatter_(-1, topk_index, 1.0)
            assign_hard /= args.pq_hard_softmax_topk
            assign = assign_hard.detach() - assign.detach() + assign
        return assign

    def eval_nci_in_train(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask,
        nret,
        nbeam,
        do_sample,
        decode_tree
    ):
        self.model.eval()
        with torch.no_grad():
            nci_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                max_length=self.args.max_output_length,
                length_penalty=self.args.length_penalty,
                num_return_sequences=nret,
                num_beams=nbeam,
                do_sample=do_sample,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=self.decode_vocab_size,
                decode_tree=decode_tree,
                output_hidden_states=True,
                output_scores=True,
                decoder_integration=self.args.decoder_integration,
            )
        topk_labels = nci_outputs[0]
        self.model.train()
        topk_labels = topk_labels[:, 1:-1]
        return topk_labels

    def forward(
        self,
        input_ids,
        aug_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
        aug_attention_mask=None,
        logit_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        query_embedding=None,
        prefix_emb=None,
        prefix_mask=None,
        only_encoder=False,
        decoder_index=-1,
        input_mask=None,
        doc_token_ids=None,
        doc_token_mask=None,
        doc_ids=None,
        topk_labels=None,
    ):
        batch_size = input_ids.shape[0]
        if self.args.nci_twin_train_ratio and self.training:
            if self.args.alt_granularity == 'batch':
                stage = self.bs_stage[batch_size]
            else:
                stage = self.stage
            if self.args.alt_train:
                if self.args.alt_train != 'loss':
                    # alt fix
                    if stage == 1:
                        self.args.fixnci = True
                        if self.attenpool_weight is not None:
                            self.freeze_params(self.attenpool_weight)
                    elif stage == 0:
                        self.args.fixnci = False
                        if self.attenpool_weight is not None:
                            self.unfreeze_params(self.attenpool_weight)
                elif self.args.alt_train != 'fix':
                    # alt loss
                    if stage == 1:
                        self.args.no_nci_loss = True
                        self.args.no_twin_loss = False
                    else:
                        self.args.no_nci_loss = False
                        self.args.no_twin_loss = True
        else:
            stage = self.document_encoder is not None
        ori_labels = None
        centroid_update_loss = None

        if self.get_codebook_logits:
            ori_lm_labels = lm_labels
            p_reps = self.get_document_embedding(
                doc_ids, doc_token_ids, doc_token_mask, encoder_trainable=not self.args.fixdocenc)
            if self.sample_in_training:
                if self.args.pq_runtime_label:
                    do_sample = bool(self.args.aug_sample_topk)
                    nret = self.nrets
                    if self.topk_from_vq:
                        topk_labels, beam_scores = self.pq.beam_search(
                            p_reps, num_return_sequences=nret, do_sample=do_sample, return_proba=True)
                    else:
                        decoder_attention_mask = input_ids.new_ones(
                            input_ids.size(0), self.pq.subvector_num + 2)
                        decoder_attention_mask[:, -1] = 0
                        topk_labels = self.eval_nci_in_train(
                            input_ids,
                            attention_mask,
                            decoder_attention_mask,
                            nret,
                            None if self.args.aug_sample_topk else nret,
                            do_sample,
                            None if self.args.aug_sample_topk else self.root,
                        )
                        topk_labels = topk_labels.reshape(
                            batch_size, -1, topk_labels.shape[-1])
                else:
                    topk_labels = lm_labels
                    ori_labels = lm_labels[..., :-2]
                    ori_labels = ori_labels - 2 - \
                        torch.arange(
                            ori_labels.size(-1), device=ori_labels.device, dtype=ori_labels.dtype) * self.args.output_vocab_size
                    decoder_attention_mask = decoder_attention_mask.reshape(
                        -1, decoder_attention_mask.size(-1))

                topk = topk_labels.size(1)

                def expand_tensor(tensor):
                    return tensor.repeat_interleave(topk, dim=0)

                all_inputs = [input_ids, aug_input_ids, attention_mask, aug_attention_mask,
                              query_embedding, encoder_outputs, prefix_emb, prefix_mask, input_mask]
                for i in range(len(all_inputs)):
                    cur_tensor = all_inputs[i]
                    if cur_tensor is not None:
                        all_inputs[i] = expand_tensor(cur_tensor)
                input_ids, aug_input_ids, attention_mask, aug_attention_mask, query_embedding, \
                    encoder_outputs, prefix_emb, prefix_mask, input_mask = all_inputs

                lm_labels = topk_labels.reshape(-1, topk_labels.size(-1))
                if self.args.pq_loss == 'adist':
                    lm_logits = beam_scores
                else:
                    lm_logits = None
            else:
                target = torch.arange(
                    input_ids.size(0),
                    device=input_ids.device,
                    dtype=torch.long
                )
                target = target * (p_reps.size(0) // target.size(0))
                if self.args.pq_negative == 'none':
                    cur_p_reps = p_reps[target]
                else:
                    cur_p_reps = p_reps
                lm_logits, lm_labels, centroid_update_loss = self.pq(
                    cur_p_reps)
                if self.args.pq_negative != 'none':
                    lm_labels = lm_labels[target]
            if self.args.pq_runtime_label:
                if not self.sample_in_training or self.topk_from_vq:
                    ori_labels = lm_labels
                    lm_labels, decoder_attention_mask = self.codebook_decode_embedding_process(
                        lm_labels)
                else:
                    ori_labels = lm_labels - 2 - self.args.output_vocab_size * \
                        torch.arange(
                            lm_labels.shape[1], dtype=lm_labels.dtype, device=lm_labels.device)
                    lm_labels = vq_label_suffix(lm_labels)
                    decoder_attention_mask = lm_labels.clone()
                    decoder_attention_mask[decoder_attention_mask != 0] = 1
                    lm_labels[lm_labels[:, :] ==
                              self.tokenizer.pad_token_id] = -100
            elif not self.sample_in_training:
                lm_labels = ori_lm_labels
        else:
            lm_logits = None

        if self.qemb_projection is not None or self.pemb_projection is not None or self.unified_projection is not None:
            nbeam = self.args.cat_cluster_centroid if self.qemb_projection is not None else self.args.cluster_position_topk
            if topk_labels is None:
                topk_labels = self.eval_nci_in_train(
                    input_ids,
                    attention_mask,
                    decoder_attention_mask,
                    nbeam,
                    nbeam,
                    False,
                    self.root,
                )
                topk_labels = topk_labels.reshape(
                    batch_size, -1, topk_labels.shape[-1])
                topk_labels = topk_labels - 2 - \
                    torch.arange(topk_labels.size(-1), device=topk_labels.device,
                                 dtype=topk_labels.dtype) * self.args.output_vocab_size
            topk_labels = topk_labels[:, :nbeam, :]

        if self.args.codebook and self.args.query_vq_label:
            qemb = self.document_encoder.encode_query(
                {'input_ids': input_ids, 'attention_mask': attention_mask})
            _, lm_labels, _ = self.pq(qemb)
            lm_labels, decoder_attention_mask = self.codebook_decode_embedding_process(
                lm_labels)

        input_mask = None
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder and self.training:
            if aug_input_ids is not None and self.training:
                input_ids = torch.cat(
                    [input_ids, aug_input_ids.clone()], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, aug_attention_mask], dim=0)
            elif self.training:
                input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.clone()], dim=0)
            if self.args.denoising:
                if input_mask is None:
                    input_mask = torch.rand(
                        input_ids.shape, device=input_ids.device) < 0.9
            if self.args.input_dropout and np.random.rand() < 0.5:
                if input_mask is None:
                    input_mask = torch.rand(
                        input_ids.shape, device=input_ids.device) < 0.9
                input_ids = torch.where(
                    input_mask == True, input_ids, torch.zeros_like(input_ids))
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask], dim=0)
            if lm_labels is not None:
                lm_labels = torch.cat([lm_labels, lm_labels], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, decoder_input_ids], dim=0)

        if self.args.loss_weight:
            loss_weight = torch.ones(
                [input_ids.shape[0], self.args.max_output_length]).to(input_ids.device)
            loss_weight = loss_weight - \
                torch.arange(start=0, end=0.5, step=0.5 /
                             self.args.max_output_length).reshape(1, -1).to(input_ids.device)
        else:
            loss_weight = None

        with torch.set_grad_enabled(not self.args.fixnci):
            out = self.model(
                input_ids,
                input_mask=input_mask,
                logit_mask=logit_mask,
                encoder_outputs=encoder_outputs,
                only_encoder=only_encoder,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                query_embedding=query_embedding,
                prefix_embedding=prefix_emb,
                prefix_mask=prefix_mask,
                return_dict=True,
                output_hidden_states=True,
                decoder_index=decoder_index,
                loss_weight=loss_weight,
            )

        if self.args.codebook and self.args.pq_loss != 'label' and not self.args.fixpq:
            args = self.args
            nci_logits = out.nci_logits
            if not args.pq_loss.endswith('ce'):
                nci_logits = self.get_softmax(nci_logits)
            if args.pq_loss == 'emdr2':
                pqloss = 0.
                new_nci_logits = self.compute_emdr2_loss(
                    nci_logits, ori_labels)
            else:
                if args.pq_loss != 'ce':
                    lm_logits = self.get_softmax(lm_logits)
                if args.Rdrop > 0 and self.training:
                    pq_logits = torch.cat(
                        [lm_logits, lm_logits.clone()], dim=0)
                else:
                    pq_logits = lm_logits
                pqloss = self.compute_pq_loss(
                    nci_logits, pq_logits, ori_labels)
            if args.Rdrop > 0 and self.training:
                out.orig_loss = pqloss
            #out.loss += pqloss
            out.loss = pqloss

        out.co_loss = 0
        if self.args.pq_twin_loss == 'quant':
            if lm_logits is None:
                p_reps = self.get_document_embedding(
                    doc_ids, doc_token_ids, doc_token_mask)
                lm_logits, _, _ = self.pq(p_reps)
            quantized_query_embedding = self.pq.get_reconstruct_vector_matrix_multiply(
                nci_logits)
            quantized_document_embedding = self.pq.get_reconstruct_vector_matrix_multiply(
                lm_logits)
            loss = self.document_encoder(
                quantized_query_embedding, quantized_document_embedding, passage=None)
            out.co_loss = loss.loss
        elif stage == 1:
            if self.sample_in_training:
                nret = self.nrets
                if self.args.pq_loss == 'emdr2':
                    emdr2_loss = 0.
                if self.qemb_from_ckpt or self.args.query_encoder == 'twin':
                    query_embedding = self.get_query_embedding(
                        input_ids,
                        attention_mask,
                        out.encoder_last_hidden_state,
                        out.ori_decoder_last_hidden_state,
                        out.decoder_last_hidden_state,
                        lm_labels,
                        topk_labels,
                    )
                else:
                    query_embedding = None
                co_loss = 0.
                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                all_similarity = []
                ori_labels = ori_labels.reshape(-1, nret, ori_labels.shape[-1])
                for i in range(batch_size):
                    temp_p_reps = p_reps[i].unsqueeze(0)
                    for j in range(nret):
                        offset = i*nret+j
                        doc_clus_id = ori_labels[i][j].tolist()
                        hns = self.sample_negatives(
                            doc_ids[i][0].item(), tuple(doc_clus_id), None)
                        temp_doc_embeddings = self.get_document_embedding(hns)
                        if query_embedding is None:
                            sli = slice(offset, offset+1)
                            temp_query_embedding = self.clus_repr(
                                out.encoder_last_hidden_state[sli],
                                attention_mask[i:i+1],
                                out.ori_decoder_last_hidden_state[
                                    sli] if out.ori_decoder_last_hidden_state is not None else None,
                                out.decoder_last_hidden_state[sli],
                                lm_labels[sli],
                            )
                        else:
                            temp_query_embedding = query_embedding
                        temp_doc_embeddings = torch.cat(
                            [temp_p_reps, temp_doc_embeddings])
                        similarity = self.document_encoder.compute_similarity(
                            temp_query_embedding, temp_doc_embeddings)
                        all_similarity.append(similarity)
                        if self.args.pq_loss == 'emdr2':
                            pdocenc = torch.nn.functional.softmax(
                                similarity, dim=-1).squeeze(0)
                            pnci = new_nci_logits[i][j]
                            emdr2_loss += torch.log(pdocenc[0] * pnci + 1e-12)
                scores = torch.cat(all_similarity)
                # target = torch.arange(
                #     scores.size(0),
                #     device=scores.device,
                #     dtype=torch.long
                # )
                # target = target * scores.shape[1]
                target = torch.zeros(scores.size(
                    0), device=scores.device, dtype=torch.long)
                co_loss = loss_fn(scores, target)
                out.co_loss = co_loss
                if self.args.pq_loss == 'emdr2':
                    emdr2_loss = -emdr2_loss / batch_size / self.args.aug_topk_clus
                    if self.args.Rdrop > 0:
                        out.orig_loss = emdr2_loss
                        out.loss += emdr2_loss
                    else:
                        out.loss = emdr2_loss
            elif not self.args.no_twin_loss:
                # decoder_last_hidden_state (bs, args.max_output_length, dim)
                query_embedding = self.get_query_embedding(
                    input_ids,
                    attention_mask,
                    out.encoder_last_hidden_state,
                    out.ori_decoder_last_hidden_state,
                    out.decoder_last_hidden_state,
                    lm_labels,
                    topk_labels,
                )
                if doc_ids is None:
                    doc_token_ids = doc_token_ids.view(-1,
                                                       self.args.co_doc_length)
                    doc_token_mask = doc_token_mask.view(-1,
                                                         self.args.co_doc_length)
                if not self.get_codebook_logits:
                    p_reps = self.get_document_embedding(
                        doc_ids, doc_token_ids, doc_token_mask)
                if self.pemb_projection is not None:
                    p_reps = self.pemb_projection(
                        p_reps,
                        pred_clusters=topk_labels,
                        gt_clusters=[[self.pq_mapping[d] for d in did]
                                     for did in doc_ids.cpu().numpy()],
                        logits=out.nci_logits,
                    )
                elif self.unified_projection is not None and not hasattr(self, 'all_embeddings'):
                    p_reps = self.unified_projection.encode_passage(p_reps, topk_labels, [
                                                                    [self.pq_mapping[d] for d in did] for did in doc_ids.cpu().numpy()])
                loss = self.document_encoder(
                    query_embedding, p_reps, passage=None)
                out.co_loss = loss.loss

        if self.args.codebook and self.args.reconstruct_for_embeddings:
            if ori_labels is None:
                ori_labels = lm_labels[..., :-2]
                ori_labels = ori_labels - 2 - \
                    torch.arange(
                        ori_labels.size(-1), device=ori_labels.device, dtype=ori_labels.dtype) * self.args.output_vocab_size
            if self.args.pq_negative != 'none':
                target = torch.arange(
                    input_ids.size(0),
                    device=input_ids.device,
                    dtype=torch.long
                )
                target = target * (p_reps.size(0) // target.size(0))
                cur_p_reps = p_reps[target]
            centroid_update_loss = self.pq.get_reconstruct_loss_for_embeddings(
                cur_p_reps, ori_labels)

        if self.args.codebook:
            if centroid_update_loss is None:
                centroid_update_loss = 0
            out.centroid_update_loss = centroid_update_loss

        return out

    def _step(self, batch):
        loss, orig_loss, dist_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss = None, None, None, None, None, None, None
        if self.args.multiple_decoder:
            encoder_outputs, input_mask, generation_loss, denoising_loss = self.forward(input_ids=batch["source_ids"], aug_input_ids=batch["aug_source_ids"],
                                                                                        attention_mask=batch["source_mask"], aug_attention_mask=batch[
                "aug_source_mask"],
                query_embedding=batch["query_emb"], only_encoder=True)
            l1, l2, l3, l4, l5, l6, l7 = [], [], [], [], [], [], []
            for i in range(self.args.decoder_num):
                cl1, cl2, cl3, cl4, cl5, cl6, cl7 = self._step_i(
                    batch, i, encoder_outputs=encoder_outputs, input_mask=input_mask)
                l1.append(cl1)
                l2.append(cl2)
                l3.append(cl3)
                l4.append(cl4)
                l5.append(cl5)
                l6.append(cl6)
                l7.append(cl7)
            loss = torch.stack(l1, dim=0).sum(dim=0) if l1[0] != 0 else 0
            orig_loss = torch.stack(l2, dim=0).sum(dim=0) if l2[0] != 0 else 0
            dist_loss = torch.stack(l3, dim=0).sum(dim=0) if l3[0] != 0 else 0
            co_loss = torch.stack(l4, dim=0).sum(dim=0) if l4[0] != 0 else 0
            q_emb_distill_loss = torch.stack(
                l5, dim=0).sum(dim=0) if l5[0] != 0 else 0
            weight_distillation = torch.stack(
                l6, dim=0).sum(dim=0) if l6[0] != 0 else 0
            centroid_update_loss = torch.stack(
                l7, dim=0).sum(dim=0) if l7[0] != 0 else 0
        else:
            loss, orig_loss, dist_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss = self._step_i(
                batch, -1)

        return loss, orig_loss, dist_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss

    def _step_i(self, batch, i, encoder_outputs=None, input_mask=None):
        if self.args.codebook and self.args.pq_runtime_label:
            lm_labels = target_mask = None
        else:
            if i < 0:
                lm_labels = batch["target_ids"]
                target_mask = batch['target_mask']
            else:
                lm_labels = batch["target_ids"][i]
                target_mask = batch['target_mask'][i]
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.forward(
            input_ids=batch["source_ids"],
            aug_input_ids=batch["aug_source_ids"],
            attention_mask=batch["source_mask"],
            aug_attention_mask=batch["aug_source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=target_mask,
            query_embedding=batch["query_emb"],
            decoder_index=i,
            encoder_outputs=encoder_outputs,
            prefix_emb=batch["prefix_emb"],
            prefix_mask=batch["prefix_mask"],
            input_mask=input_mask,
            doc_token_ids=batch.get('doc_token_ids', None),
            doc_token_mask=batch.get('doc_token_mask', None),
            doc_ids=batch.get('doc_ids', None),
            topk_labels=batch.get('topk_labels', None),
        )

        neg_outputs = None
        if self.args.hard_negative and self.args.sample_neg_num > 0:
            neg_lm_labels = torch.cat(batch['neg_target_ids'], dim=0)
            neg_decoder_attention_mask = torch.cat(
                batch['neg_target_mask'], dim=0)
            attention_mask = batch["source_mask"].repeat(
                [self.args.sample_neg_num, 1])
            sources_ids = batch["source_ids"].repeat(
                [self.args.sample_neg_num, 1])
            neg_lm_labels[neg_lm_labels[:, :] ==
                          self.tokenizer.pad_token_id] = -100
            neg_outputs = self.forward(input_ids=sources_ids, decoder_index=i, encoder_outputs=outputs.encoder_outputs, attention_mask=attention_mask,
                                       lm_labels=neg_lm_labels, decoder_attention_mask=neg_decoder_attention_mask, query_embedding=batch['query_emb'])

        def select_lm_head_weight(cur_outputs):
            lm_head_weight = cur_outputs.lm_head_weight
            vocab_size = lm_head_weight.shape[-1]
            dim_size = lm_head_weight.shape[-2]
            # [batch_size, seq_length, dim_size, vocab_size]
            lm_head_weight = lm_head_weight.view(-1, vocab_size)
            indices = cur_outputs.labels.unsqueeze(
                -1).repeat([1, 1, dim_size]).view(-1, 1)
            indices[indices[:, :] == -100] = self.tokenizer.pad_token_id
            # [batch_size, seq_length, dim_size, 1]
            lm_head_weight = torch.gather(lm_head_weight, -1, indices)
            lm_head_weight = lm_head_weight.view(
                cur_outputs.decoder_hidden_states[-1].shape)
            return lm_head_weight

        def cal_contrastive(outputs, neg_outputs):
            vocab_size = outputs.lm_head_weight.shape[-1]
            dim_size = outputs.lm_head_weight.shape[-2]
            decoder_weight = select_lm_head_weight(outputs)

            if neg_outputs is not None:
                decoder_embed = torch.cat((outputs.decoder_hidden_states[-1], neg_outputs.decoder_hidden_states[-1]), dim=0).transpose(
                    0, 1).transpose(1, 2)  # [seq_length, embed_size, batch_size*2]
                neg_decoder_weight = select_lm_head_weight(neg_outputs)
                decoder_weight = torch.cat(
                    (decoder_weight, neg_decoder_weight), dim=0).transpose(0, 1).transpose(1, 2)
            else:
                decoder_embed = outputs.decoder_hidden_states[-1].transpose(
                    0, 1).transpose(1, 2)
                decoder_weight = decoder_weight.transpose(0, 1).transpose(1, 2)
            seq_length = decoder_embed.shape[0]
            embed_size = decoder_embed.shape[1]
            bz = outputs.encoder_last_hidden_state.shape[0]
            # print("decoder_embed", decoder_embed.shape)  #[seq_length, embed_size, batch_size + neg_bz]
            # print("decoder_weight", decoder_weight.shape) #[seq_length, embed_size, batch_size + neg_bz]
            query_embed = outputs.encoder_last_hidden_state[:, 0, :].unsqueeze(
                0).repeat([seq_length, 1, 1])  # [seq_length, batch_size, embed_size]
            # query_tloss = self.triplet_loss(query_embed, decoder_embed[:,:,0:bz], decoder_embed[:,:,bz:])
            # query_tloss = self.triplet_loss(query_embed, decoder_weight[:,:,0:bz], decoder_weight[:,:,bz:])
            query_tloss = None
            weight_tloss = None
            disc_loss = None
            ranking_loss = None
            if self.args.query_tloss:
                # [seq_length, embed_size, pos_bz+neg_bz]
                all_doc_embed = decoder_embed
                # [sl, bz, bz+neg_bz]
                doc_logits = torch.bmm(query_embed, all_doc_embed)
                contrast_labels = torch.arange(
                    0, bz).to(doc_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(
                    0).repeat(seq_length, 1)
                # masks = outputs.labels.transpose(0, 1).repeat([1, 1+self.args.sample_neg_num])
                contrast_labels[outputs.labels.transpose(
                    0, 1)[:, :] == -100] = -100
                query_tloss = self.ce(doc_logits.view(
                    seq_length*bz, -1), contrast_labels.view(-1))
            if self.args.weight_tloss:
                query_embed = query_embed.transpose(1, 2)
                doc_embed = decoder_embed[:, :, 0:bz].transpose(
                    1, 2)  # [seq_length, batch_size, embed_size]
                query_logits = torch.bmm(
                    doc_embed, query_embed)  # [sl, bz, bz]
                contrast_labels = torch.arange(
                    0, bz).to(query_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(
                    0).repeat(seq_length, 1)  # [sl, bz]
                contrast_labels[outputs.labels.transpose(
                    0, 1)[:, :] == -100] = -100
                weight_tloss = self.ce(query_logits.view(
                    seq_length*bz, -1), contrast_labels.view(-1))
            if self.args.ranking_loss:
                rank_target = torch.ones(bz*seq_length).to(lm_labels.device)
                rank_indices = outputs.labels.detach().clone().reshape([-1, 1])
                rank_indices[rank_indices[:, :] == -
                             100] = self.tokenizer.pad_token_id
                # pos_prob = torch.gather(self.softmax(outputs.lm_logits.detach().clone()).view(-1, vocab_size), -1, rank_indices).squeeze(-1)
                pos_prob = torch.gather(self.softmax(
                    outputs.lm_logits).view(-1, vocab_size), -1, rank_indices)
                pos_prob[rank_indices[:, :] ==
                         self.tokenizer.pad_token_id] = 1.0
                pos_prob = pos_prob.squeeze(-1)
                # [bz, seq_length, vocab_size] -> [bz, seq_length]
                # pos_prob, _ = torch.max(self.softmax(outputs.lm_logits.detach()), -1)
                neg_prob, _ = torch.max(
                    self.softmax(neg_outputs.lm_logits), -1)
                ranking_loss = self.ranking_loss(
                    pos_prob.view(-1), neg_prob.view(-1), rank_target)
            if self.args.disc_loss:
                target = torch.zeros(seq_length, bz).to(lm_labels.device)
                target[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                all_logits = self.dfc(torch.reshape(decoder_embed.transpose(
                    1, 2), (-1, embed_size))).view(seq_length, -1)  # [seq_length, bz+neg_bz]
                all_logits = all_logits.view(
                    seq_length, self.args.sample_neg_num+1, bz).transpose(1, 2)
                # [seq_length*bz, pos+neg_num]
                all_logits = torch.reshape(
                    all_logits, (-1, self.args.sample_neg_num+1))
                disc_loss = self.ce(
                    all_logits.view(-1, self.args.sample_neg_num+1), target.view(-1).long())
            return query_tloss, weight_tloss, disc_loss, ranking_loss

        if self.args.softmax:
            logits = self.fc(outputs.encoder_last_hidden_state)[
                :, 0, :].squeeze()
            loss = self.ce(logits, batch["softmax_index"].squeeze(dim=1))
        else:
            if self.args.hard_negative:
                query_tloss, weight_tloss, disc_loss, ranking_loss = cal_contrastive(
                    outputs, neg_outputs)
                loss = outputs.loss
                if self.args.ranking_loss:
                    loss += ranking_loss
                if self.args.disc_loss:
                    loss += disc_loss
                    loss = outputs.loss
                if self.args.query_tloss:
                    loss += query_tloss
                if self.args.weight_tloss:
                    loss += weight_tloss
            else:
                loss = outputs.loss

        if self.args.Rdrop > 0:
            orig_loss = outputs.orig_loss
            dist_loss = outputs.dist_loss
        else:
            orig_loss, dist_loss = 0, 0
        co_loss = outputs.co_loss
        if self.args.document_encoder and self.co_loss_scale is not None:
            if self.current_epoch >= len(self.co_loss_scale):
                scale = self.co_loss_scale[-1]
            else:
                scale = self.co_loss_scale[self.current_epoch]
            co_loss *= scale

        if self.args.embedding_distillation > 0:
            q_emb_distill_loss = outputs.emb_distill_loss
        else:
            q_emb_distill_loss = 0

        if self.args.weight_distillation > 0:
            weight_distillation = outputs.weight_distillation
        else:
            weight_distillation = 0

        if self.args.codebook and outputs.centroid_update_loss != 0:
            centroid_update_loss = outputs.centroid_update_loss * self.args.centroid_loss_scale
        else:
            centroid_update_loss = 0

        return loss, orig_loss, dist_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss

    def _softmax_generative_step(self, batch):
        assert self.args.softmax
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        pred_index = torch.argmax(outputs[0], dim=1)
        return pred_index

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def on_train_epoch_end(self):
        if self.alt_cross_epoch:
            if (self.current_epoch + 1) % self.nci_twin_alt_epoch[1] in (self.nci_twin_alt_epoch[0], 0):
                self.stage = 1 - self.stage

    def on_train_epoch_start(self):
        self.model.train()
        self.train_dataset.reload_dataset(self.rank, self.nrank)
        if self.alt_cross_epoch:
            # workaround for enable validation when the number of batch not reach len(sampler)
            # if (self.current_epoch) % self.nci_twin_alt_epoch[1] in (self.nci_twin_alt_epoch[0], 0):
            self.trainer.val_check_batch = float('inf')
        if self.document_encoder and self.nci_vq_alt_epoch:
            mod = (self.current_epoch) % self.nci_vq_alt_epoch[1]
            print("enter here, mod", mod, self.nci_vq_alt_epoch)
            if mod == 0:
                self.args.fixnci = True
                if self.attenpool_weight is not None:
                    self.freeze_params(self.attenpool_weight)
                self.args.fixpq = False
                self.args.fixdocenc = False or self.args.original_fixdocenc
                if not self.args.fixdocenc:
                    self.unfreeze_document_encoder()
            elif mod == self.nci_vq_alt_epoch[0]:
                self.args.fixnci = False
                if self.attenpool_weight is not None:
                    self.unfreeze_params(self.attenpool_weight)
                self.args.fixpq = True
                self.args.fixdocenc = True
                self.freeze_document_encoder()

    def training_step(self, batch, batch_idx):
        # set to train
        loss, orig_loss, kl_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss = self._step(
            batch)
        all_loss = 0
        if not self.args.no_nci_loss:
            all_loss += loss
        if not self.args.no_twin_loss:
            all_loss += co_loss
        all_loss += centroid_update_loss
        results = {"loss": all_loss}
        if self.document_encoder:
            results["nci_loss"] = loss
            if co_loss:
                results["co_loss"] = co_loss
        if self.args.Rdrop > 0:
            results["orig_loss"] = orig_loss
            results["kl_loss"] = kl_loss
        if self.args.embedding_distillation > 0:
            results["Query_distill_loss"] = q_emb_distill_loss
        if self.args.weight_distillation > 0:
            results["Weight_distillation"] = weight_distillation
        if self.args.codebook:
            results["centroid_update_loss"] = centroid_update_loss
        for k, v in results.items():
            self.log(k, v)
        assert not torch.isnan(all_loss).item()
        return results

    def training_epoch_end(self, outputs):
        self.cur_ep += 1
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def barrier(self):
        if self.nrank > 1:
            dist.barrier()

    @property
    def rank(self):
        if self.args.mode == 'eval':
            if dist.is_initialized():
                return dist.get_rank()
            else:
                return 0
        else:
            return self.trainer.global_rank

    @property
    def nrank(self):
        if self.args.mode == 'eval':
            if dist.is_initialized():
                return dist.get_world_size()
            else:
                return 1
        else:
            return self.trainer.world_size

    @property
    def doc_emb(self):
        return self.all_embeddings.content[0]

    def safe_rm(self, file_path):
        if osp.isfile(file_path):
            os.remove(file_path)

    def get_ep(self, fixed):
        if fixed:
            ep = ''
        elif self.trainer is not None and self.trainer.sanity_checking:
            ep = '_epcheck'
        else:
            ep = f'_ep{self.current_epoch}'
        return ep

    def get_current_path(self, name, nickname, pq_based=True):
        loaded = hasattr(self, name)
        args = self.args
        if pq_based:
            fixed = args.fixpq and args.pq_update_method == 'grad'
            has_input_args = args.pq_cluster_path is not None
            suffix = 'pt' if nickname != 'clus' else 'pkl'
        else:
            fixed = args.fixdocenc and (
                self.unified_projection is None or args.fixproj)
            has_input_args = args.embedding_path is not None
            suffix = 'bin'
        cur_fixed = (fixed or (self.alt_cross_epoch and self.stage == 0))
        if loaded and cur_fixed:
            return None
        elif not loaded and has_input_args:
            if pq_based and nickname != 'clus':
                current_path = args.pq_cluster_path.replace(
                    'clus', nickname).replace('.pkl', '.pt')
            elif pq_based:
                current_path = args.pq_cluster_path
                assert current_path.endswith('.pkl')
            else:
                current_path = self.args.embedding_path
                assert current_path.endswith('.bin')
            return current_path
        else:
            ep = self.get_ep(fixed)
            if pq_based:
                prefix = f'{args.pq_type}'
            else:
                prefix = 'docemb'
            current_path = f'{prefix}{nickname}{self.args.document_encoder}{ep}_{self.args.time_str}.{suffix}'
            current_path = osp.join(args.data_dir, current_path)
            return current_path

    @torch.no_grad()
    def gen_all_query_embedding(self, query_file, output_path, rank=None, nrank=None):
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        args = self.args
        dim = args.d_model
        batch_size = args.encode_batch_size
        df = pd.read_csv(
            query_file,
            names=['query', 'oldid'],
            encoding='utf-8',
            header=None,
            sep='\t',
        )['query']
        add_one = rank < (len(df) % nrank)
        nbatch_partial = len(df) // nrank
        cur_start = nbatch_partial * rank
        if add_one:
            cur_start += rank
            cur_ending = cur_start + nbatch_partial + 1
        else:
            cur_start += len(df) % nrank
            cur_ending = cur_start + nbatch_partial
        if nrank > 1:
            cur_output_path = output_path[:-4] + f'_{rank}.bin'
        else:
            cur_output_path = output_path
        all_embeddings = np.memmap(
            cur_output_path, dtype=np.float32, mode='w+', shape=(cur_ending - cur_start, dim))
        if self.unified_projection is not None:
            target_mask = torch.ones(
                (batch_size, args.subvector_num + 2), dtype=torch.int64).cuda()
            target_mask[:, -1] = 0
        add_special_tokens = args.document_encoder in ('ance', 'ar2')
        for start in tqdm(range(cur_start, cur_ending, batch_size)):
            ending = min(start+batch_size, cur_ending)
            batch_query = df[start:ending]
            output = self.tokenizer.batch_encode_plus(batch_query, max_length=args.max_input_length, add_special_tokens=add_special_tokens,
                                                      padding='max_length', truncation=True, return_tensors="pt")
            input_ids = output['input_ids'].cuda()
            attention_mask = output['attention_mask'].cuda()
            qemb = self.document_encoder.encode_query(
                {'input_ids': input_ids, 'attention_mask': attention_mask})
            if self.unified_projection is not None:
                assert args.codebook
                if input_ids.shape[0] != batch_size:
                    cur_target_mask = input_ids.new_ones(
                        input_ids.shape[0], args.subvector_num + 2)
                    cur_target_mask[:, -1] = 0
                else:
                    cur_target_mask = target_mask
                nci_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    max_length=args.max_output_length,
                    length_penalty=args.length_penalty,
                    num_return_sequences=args.num_return_sequences,
                    early_stopping=False,
                    decode_embedding=args.decode_embedding,
                    decode_vocab_size=self.decode_vocab_size,
                    decode_tree=self.root,
                    output_hidden_states=True,
                    output_scores=True,
                    decoder_integration=args.decoder_integration,
                    decoder_attention_mask=cur_target_mask,
                    num_beams=args.num_return_sequences,
                )
                outs = nci_outputs[0]
                assert args.decode_embedding == 2
                dec, eos_idx = decode_token(args, outs)
                dec = dec_2d(dec, args.num_return_sequences)
                clusters = dec[:, :args.cluster_position_topk, :]

                qemb = self.unified_projection.encode_query(qemb, clusters)

            all_embeddings[start-cur_start:ending -
                           cur_start] = qemb.detach().cpu().numpy()
        all_embeddings.flush()
        if nrank > 1:
            dist.barrier()
            if rank == 0:
                all_embeddings = np.memmap(
                    output_path, dtype=np.float32, mode='w+', shape=(len(df), dim))
                cur_start = 0
                for i in range(nrank):
                    cur_embeddings = np.memmap(
                        output_path[:-4] + f'_{i}.bin', dtype=np.float32, mode='r').reshape(-1, dim)
                    cur_ending = cur_start + cur_embeddings.shape[0]
                    all_embeddings[cur_start:cur_ending] = cur_embeddings
                    cur_start = cur_ending
                all_embeddings.flush()
                for i in range(nrank):
                    self.safe_rm(output_path[:-4] + f'_{i}.bin')
            dist.barrier()

    def gen_doc_embedding(self, rank=None, nrank=None, get_all=True):
        # generate all doc embeddings before validation
        current_embedding_path = self.get_current_path(
            'current_embedding_path', '', False)
        if current_embedding_path is not None:
            self.current_embedding_path = current_embedding_path
        # self.current_embedding_path = '...' # only generate embedding, with unified embedding
        embedpath_prefix = move_to_tmp(self.current_embedding_path)[:-4]
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        assert rank is not None and nrank is not None
        part_embedding_path = f'{embedpath_prefix}_{rank}.bin'
        batch_size = self.args.encode_batch_size
        dim = self.args.d_model
        num_docs = len(self.all_docs)
        num_docs_per_worker = num_docs // nrank
        start = num_docs_per_worker * rank
        if rank + 1 == nrank:
            ending = num_docs
        else:
            ending = start + num_docs_per_worker
        if not osp.exists(self.current_embedding_path):
            if self.args.fixdocenc and osp.exists(self.args.embedding_path):
                raw_embeddings = np.memmap(
                    self.args.embedding_path, dtype=np.float32, mode='r', shape=(num_docs, dim))
            else:
                raw_embeddings = None
            print(
                f'Generating embeddings with unique time string {self.args.time_str}...')
            print(
                f'Generate embedding from {start} to {ending} in {num_docs} docs...')
            part_embeddings = np.memmap(
                part_embedding_path, dtype=np.float32, mode='w+', shape=(ending - start, dim))
            for ind, i in enumerate(tqdm(range(start, ending, batch_size), desc='Generate Embedding')):
                cur_start = i
                cur_ending = min(i+batch_size, ending)
                if raw_embeddings is None:
                    if self.all_docs.dtype is None:
                        # need tokenization
                        contents = self.all_docs[cur_start:cur_ending].tolist()
                        add_special_tokens = self.args.dataset in (
                            'marco', 'nq_dpr') and self.args.document_encoder in ('ance', 'ar2')
                        output_ = self.passage_tokenizer.batch_encode_plus(
                            contents,
                            max_length=128,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=add_special_tokens,
                            return_tensors='pt',
                        )
                        doc_token_ids = output_['input_ids']
                        doc_token_mask = output_['attention_mask']
                    else:
                        doc_token_ids, doc_token_mask = self.all_docs[cur_start:cur_ending]
                    doc_token_ids = doc_token_ids.cuda()
                    doc_token_mask = doc_token_mask.cuda()
                    output = self.document_encoder.generate(None, {'input_ids': doc_token_ids.view(
                        -1, self.args.co_doc_length), 'attention_mask': doc_token_mask.view(-1, self.args.co_doc_length)})
                    embedding = output.p_reps
                else:
                    embedding = raw_embeddings[cur_start:cur_ending]
                if self.unified_projection is not None:
                    embedding = self.unified_projection.encode_passage(
                        embedding,
                        None,
                        [self.pq_mapping[did]
                            for did in range(cur_start, cur_ending)],
                    )
                embedding = embedding.detach().cpu().numpy()
                part_embeddings[cur_start -
                                start:cur_ending-start, :] = embedding
            part_embeddings.flush()
            del part_embeddings
            self.barrier()
            if rank == 0:
                all_embeddings = np.memmap(
                    self.current_embedding_path, dtype=np.float32, mode='w+', shape=(num_docs, dim))
                row_offset = 0
                for i in range(nrank):
                    part_embeddings = np.memmap(
                        f'{embedpath_prefix}_{i}.bin', dtype=np.float32, mode='r')
                    part_embeddings = part_embeddings.reshape(-1, dim)
                    row_offset_ending = row_offset + part_embeddings.shape[0]
                    all_embeddings[row_offset:row_offset_ending,
                                   :] = part_embeddings
                    row_offset = row_offset_ending
                all_embeddings.flush()
                del all_embeddings
                del part_embeddings
                for i in range(nrank):
                    self.safe_rm(f'{embedpath_prefix}_{i}.bin')
            self.barrier()
        if get_all:
            all_embeddings = np.memmap(
                self.current_embedding_path, dtype=np.float32, mode='r', shape=(num_docs, dim))
            self.all_embeddings = IndexedData(
                content=all_embeddings, torch_dtype=torch.float32)
        else:
            assert False, 'Bug may exists.'
            part_embeddings = np.memmap(
                self.current_embedding_path, dtype=np.float32, mode='r', shape=(ending-start, dim), offset=start*dim)
            self.all_embeddings = IndexedData(
                content=part_embeddings, torch_dtype=torch.float32)

    def gen_pq_doc_cluster(self, rank=None, nrank=None):
        args = self.args
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        current_pq_cluster_path = self.get_current_path(
            'current_pq_cluster_path', 'clus')
        if current_pq_cluster_path is not None:
            self.current_pq_cluster_path = current_pq_cluster_path

        pq_mapping_path = self.current_pq_cluster_path.replace(
            'clus', 'mapping')
        if not osp.exists(self.current_pq_cluster_path) or not osp.exists(pq_mapping_path):
            if self.pq.get_preds:
                self.barrier()
                if self.rank == 0:
                    pq_doc_cluster, new_mapping = self.pq.get_document_cluster_simple(
                        return_mapping=True)
                    with open(self.current_pq_cluster_path, 'wb') as fw:
                        pickle.dump(pq_doc_cluster, fw)
                    with open(pq_mapping_path, 'wb') as fw:
                        pickle.dump(new_mapping, fw)
                self.pq.get_preds = False
                self.barrier()
            else:
                pq_doc_cluster, new_mapping = self.pq.get_document_cluster(
                    self.doc_emb, rank, nrank, 128, return_mapping=True)
                LogPklFile.log_file(
                    pq_doc_cluster, self.current_pq_cluster_path, rank, nrank, self.barrier, 'cluster')
                LogPklFile.log_file(
                    new_mapping, pq_mapping_path, rank, nrank, self.barrier, 'dict')

        with open(self.current_pq_cluster_path, 'rb') as fr:
            all_pq_cluster = pickle.load(fr)
        with open(pq_mapping_path, 'rb') as fr:
            all_pq_mapping = pickle.load(fr)
        self.pq_doc_cluster = all_pq_cluster
        self.pq_mapping = all_pq_mapping
        print('Number of all pq document clusters:', len(all_pq_cluster))

    def gen_pq_doc_topk(self, rank=None, nrank=None, return_cluster=False):
        args = self.args
        num_return_sequences = self.nrets
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        current_pq_doc_topk_path = self.get_current_path(
            'current_pq_doc_topk_path', f'topk{num_return_sequences}')
        if current_pq_doc_topk_path is not None:
            self.current_pq_doc_topk_path = current_pq_doc_topk_path

        multicluster_path = self.current_pq_doc_topk_path.replace(
            'topk', 'multiclus').replace('.pt', '.pkl')
        multiclusterindex_path = self.current_pq_doc_topk_path.replace(
            'topk', 'multiclusindex').replace('.pt', '.pkl')

        if not osp.exists(self.current_pq_doc_topk_path):
            pq_doc_topk_labels = self.pq.get_topk_document_mapping(
                self.doc_emb, rank, nrank, num_return_sequences, 64)
            LogTorchFile.log_file(pq_doc_topk_labels.cpu(
            ), self.current_pq_doc_topk_path, rank, nrank, self.barrier)

        all_pq_doc_topk = torch.load(self.current_pq_doc_topk_path)
        self.pq_doc_topk_mapping = all_pq_doc_topk
        if return_cluster:
            if rank == 0:
                multicluster = defaultdict(list)
                inclusindex = defaultdict(list)
                for i, labels in enumerate(all_pq_doc_topk):
                    for j, lab in enumerate(labels):
                        lab = tuple(lab.tolist())
                        multicluster[lab].append(i)
                        inclusindex[lab].append(j)
                multicluster = dict(multicluster)
                with open(multicluster_path, 'wb') as fw:
                    pickle.dump(multicluster, fw)
                if args.doc_multiclus > 1 and args.use_topic_model:
                    for k in inclusindex:
                        inclusindex[k] = torch.LongTensor(inclusindex[k])
                    with open(multiclusterindex_path, 'wb') as fw:
                        pickle.dump(inclusindex, fw)
            self.barrier()
            with open(multicluster_path, 'rb') as fr:
                multicluster = pickle.load(fr)
            self.pq_doc_cluster = multicluster
            if args.doc_multiclus > 1 and args.use_topic_model:
                with open(multiclusterindex_path, 'rb') as fr:
                    inclusindex = pickle.load(fr)
                self.pq_inclus_index = inclusindex

    def gen_all_reconstruct(self, rank=None, nrank=None):
        args = self.args
        assert args.codebook
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        current_reconstruct_path = self.get_current_path(
            'current_reconstruct_path', 'reconstruct')
        if current_reconstruct_path is not None:
            self.current_reconstruct_path = current_reconstruct_path

        if rank == 0 and not osp.exists(self.current_reconstruct_path):
            print('Generating all reconstruct centroid embeddings.')
            indices = []
            arange = torch.arange(args.kary)
            num_choices = args.kary ** (args.subvector_num - 1)
            for i in range(args.subvector_num):
                cur_indices = arange.repeat(num_choices)
                indices.append(cur_indices)
                if i != args.subvector_num - 1:
                    num_choices //= args.kary
                    arange = arange.unsqueeze(-1).repeat(1, args.kary).view(-1)
            indices = indices[::-1]
            codebook = self.pq.get_codebook().cpu()
            centroids = [codebook[i].index_select(
                0, indices[i]) for i in range(args.subvector_num)]
            if args.pq_type == 'pq':
                all_reconstruct = torch.cat(centroids, dim=-1)
            elif args.pq_type == 'rq':
                all_reconstruct = sum(centroids)
            else:
                assert False, 'Not support OPQ.'
            torch.save(all_reconstruct, self.current_reconstruct_path)
        self.barrier()
        all_reconstruct = torch.load(self.current_reconstruct_path)
        self.all_reconstruct = all_reconstruct
        if self.unified_projection is not None:
            self.unified_projection.all_reconstruct = self.all_reconstruct

    def gen_doc2index_mapping(self, rank=None, nrank=None):
        args = self.args
        assert args.codebook
        if rank is None:
            rank = self.rank
            nrank = self.nrank
        current_doc2index_path = self.get_current_path(
            'current_doc2index_path', f'docindex{args.doc_multiclus}')
        if current_doc2index_path is not None:
            self.current_doc2index_path = current_doc2index_path

        if not hasattr(self, 'all_reconstruct'):
            self.gen_all_reconstruct(rank, nrank)
        all_reconstruct = self.all_reconstruct

        current_doc_proba_path = self.current_doc2index_path.replace(
            f'docindex{args.doc_multiclus}', f'docproba{args.doc_multiclus}')

        batch_size = self.args.encode_batch_size
        num_docs = self.doc_emb.shape[0]
        num_docs_per_worker = num_docs // nrank
        start = num_docs_per_worker * rank
        if rank + 1 == nrank:
            ending = num_docs
        else:
            ending = start + num_docs_per_worker
        if not osp.exists(self.current_doc2index_path) or not osp.exists(current_doc_proba_path):
            if args.doc_multiclus > 1:
                mapping = self.pq_doc_topk_mapping
                local_mapping = torch.stack(
                    [mapping[docid] for docid in range(start, ending)]).to(torch.long)
            else:
                mapping = self.pq_mapping
                local_mapping = torch.LongTensor(
                    [mapping[docid] for docid in range(start, ending)]).unsqueeze(1)
            result_index = torch.zeros(
                (ending-start, args.doc_multiclus), dtype=torch.int64)
            result_proba = torch.empty(
                (ending-start, args.doc_multiclus), dtype=torch.float)
            for st in tqdm(range(start, ending, batch_size), desc='Generating doc prob'):
                en = min(st + batch_size, ending)
                cur_slice = slice(st-start, en-start)
                cur_index = result_index[cur_slice]
                cur_mapping = local_mapping[cur_slice]
                for i in range(args.subvector_num):
                    cur_index *= args.kary
                    cur_index += cur_mapping[..., i]
                result_index[cur_slice] = cur_index
                centroid_embeddings = all_reconstruct[cur_index]
                last_dim = centroid_embeddings.shape[-1]
                doc_outputs = self.document_encoder.generate(
                    centroid_embeddings.view(-1, last_dim), p_reps=self.all_embeddings[st:en].unsqueeze(1).repeat(1, args.doc_multiclus, 1).view(-1, last_dim), bmm=True)
                result_proba[cur_slice] = doc_outputs.scores.view(
                    -1, args.doc_multiclus)
            LogTorchFile.log_file(
                result_index, self.current_doc2index_path, rank, nrank, self.barrier)
            LogTorchFile.log_file(
                result_proba, current_doc_proba_path, rank, nrank, self.barrier)

        self.doc2index_mapping = torch.load(self.current_doc2index_path)
        self.all_doc_proba = torch.load(current_doc_proba_path)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        if self.args.fixnci:
            optimizer_grouped_parameters = []
        elif self.args.reserve_decoder and self.args.fixncit5:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.", "ori_decoder.")))],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.decoder_learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.", "ori_decoder")))],
                    "weight_decay": 0.0,
                    "lr": self.args.decoder_learning_rate,
                },
            ]
            if self.attenpool_weight is not None:
                optimizer_grouped_parameters[0]['params'].append(
                    self.attenpool_weight.weight)
                optimizer_grouped_parameters[1]['params'].append(
                    self.attenpool_weight.bias)
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.decoder_learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                    "weight_decay": 0.0,
                    "lr": self.args.decoder_learning_rate,
                },
            ]
            if self.attenpool_weight is not None:
                optimizer_grouped_parameters[1]['params'].append(
                    self.attenpool_weight.weight)
                optimizer_grouped_parameters[3]['params'].append(
                    self.attenpool_weight.bias)
            if self.args.fixncienc:
                optimizer_grouped_parameters = [
                    optimizer_grouped_parameters[1], optimizer_grouped_parameters[3]]
        if self.document_encoder:
            if not self.args.fixdocenc:
                if not (self.args.tie_encoders and self.args.query_encoder == 'nci'):
                    assert not (self.args.fixlmp and self.args.fixlmq)
                    if self.args.fixlmp:
                        def filter(x): return not x.startswith('lm_p.')
                    elif self.args.fixlmq:
                        def filter(x): return not x.startswith('lm_q.')
                    else:
                        def filter(x): return True
                    optimizer_grouped_parameters.append({
                        "params": [p for n, p in self.document_encoder.named_parameters() if filter(n)],
                        "weight_decay": 0.0,
                        "lr": self.args.document_encoder_learning_rate,
                    })
            if not self.args.fixproj:
                proj_param_group = {
                    "params": [],
                    "weight_decay": 0.0,
                    "lr": self.args.projection_learning_rate,
                }
                if self.pemb_projection is not None:
                    proj_param_group['params'].extend(
                        self.pemb_projection.parameters())
                if self.qemb_projection is not None:
                    proj_param_group['params'].extend(
                        self.qemb_projection.parameters())
                if self.unified_projection is not None:
                    proj_param_group['params'].extend(
                        self.unified_projection.parameters())
                if len(proj_param_group['params']) > 0:
                    optimizer_grouped_parameters.append(proj_param_group)
        if self.args.codebook and not self.args.fixpq:
            pq_params = []
            if not self.args.tie_nci_pq_centroid:
                pq_params.append(self.pq.codebook)
            else:
                pq_params.extend(
                    [p for n, p in self.pq.weight_layers.named_parameters()])
            if self.args.pq_type == 'opq':
                pq_params.append(self.pq.rotate)
            optimizer_grouped_parameters.append({
                "params": pq_params,
                "weight_decay": 0.0,
                "lr": self.args.document_encoder_learning_rate,
            })
        optimizer = AdamW(optimizer_grouped_parameters,
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        print('load training data and create training loader.')
        train_dataset = self.train_dataset
        self.prefix_embedding, self.prefix2idx_dict, self.prefix_mask = \
            train_dataset.prefix_embedding, train_dataset.prefix2idx_dict, train_dataset.prefix_mask
        if self.args.split_data:
            sampler = RandomSampler(train_dataset)
        else:
            sampler = DistributedSampler(train_dataset)
        num_workers = 4
        if self.args.nci_twin_train_ratio is not None:
            if self.args.alt_granularity == 'batch':
                batch_sampler = VariableBatchSizeSamplerWithinEpoch(
                    sampler, self.train_ratios, drop_last=self.args.drop_last)
            else:
                batch_sampler = VariableBatchNumberSamplerCrossEpoch(
                    sampler, self.train_ratios, drop_last=self.args.drop_last)
            dataloader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        else:
            dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                    drop_last=self.args.drop_last, shuffle=False, num_workers=num_workers)
        return dataloader


class T5FineTunerWithValidation(T5FineTuner):
    def __init__(self, args, train=True):
        super().__init__(args, train=train)
        if args.document_encoder and args.recall_level in ('fine', 'both') and not args.fixdocenc and args.save_top_k > 0:
            self.topk_embeddings = (
                [tuple(0 for _ in self.args.recall_num)
                 for _ in range(args.save_top_k)],  # recall@1
                [None for _ in range(args.save_top_k)],  # embedding file name
            )

    def val_dataloader(self):
        print('load validation data and create validation loader.')
        val_dataset = self.val_dataset
        if self.args.split_data:
            sampler = None
        else:
            sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(val_dataset, sampler=sampler, batch_size=self.args.eval_batch_size,
                                drop_last=self.args.drop_last, shuffle=False, num_workers=4, collate_fn=custom_collate, pin_memory=True)
        return dataloader

    def get_inference_scores(self, query_proba, doc_proba, qd_scores, eval_all=False):
        if self.args.use_topic_model:
            if eval_all:
                qd_scores = qd_scores.unsqueeze(-1)
            result = query_proba * (self.args.topic_score_ratio *
                                    doc_proba + (1 - self.args.topic_score_ratio) * qd_scores)
            if eval_all:
                if self.args.multiclus_score_aggr == 'max':
                    result = torch.max(result, dim=-1)
                else:
                    result = torch.sum(result, dim=-1)
        else:
            result = qd_scores
        return result

    @torch.no_grad()
    def infer(self, batch, validation=False):
        args = self.args

        if self.timer is not None and args.timing_infer_step > 0:
            start_time = time()

        input_ids = batch["source_ids"].cuda()
        attention_mask = batch["source_mask"].cuda()
        texts = batch['query']

        generate_all = args.use_topic_model and args.eval_all_documents

        topic_proba = None
        clusters = None
        pemb_dec = None
        if not args.eval_all_documents or args.use_topic_model or args.cat_cluster_centroid > 0 or args.cluster_position_topk > 0:

            # if args.decode_embedding:
            #     if args.position:
            #         expand_scale = args.max_output_length if not args.hierarchic_decode else 1
            #         decode_vocab_size = args.output_vocab_size * expand_scale + 2
            #     else:
            #         decode_vocab_size = 12
            # else:
            #     decode_vocab_size = None
            decode_vocab_size = self.decode_vocab_size

            assert not validation or (
                not self.args.softmax and self.args.gen_method == "greedy")

            outputs = []
            if args.multiple_decoder:
                assert not args.document_encoder
                indices = range(args.decoder_num)
            else:
                indices = [None]
            for i in indices:
                if args.softmax:
                    raise NotImplementedError
                    lm_labels = batch["target_id"].cpu().numpy().copy()
                    lm_labels[lm_labels[:, :] ==
                              self.tokenizer.pad_token_id] = -100
                    lm_labels = torch.from_numpy(lm_labels).cuda()
                    outs = self(
                        input_ids=batch["source_ids"].cuda(),
                        attention_mask=batch["source_mask"].cuda(),
                        decoder_input_ids=None,
                        lm_labels=lm_labels,
                        decoder_attention_mask=batch['target_mask'].cuda(),
                    )
                    loss = self.ce(
                        outs[0], batch["softmax_index"].squeeze(dim=1).cuda())
                    losses.append(loss.detach().cpu().numpy())
                    pred_index = torch.argmax(
                        outs[0], dim=1).detach().cpu().numpy()
                    dec = pred_index.tolist()
                else:
                    kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "use_cache": False,
                        "max_length": args.max_output_length,
                        "length_penalty": args.length_penalty,
                        "num_return_sequences": args.num_return_sequences,
                        "early_stopping": False,
                        "decode_embedding": args.decode_embedding,
                        "decode_vocab_size": decode_vocab_size,
                        "decode_tree": self.root,
                        "output_hidden_states": True,
                        "output_scores": True,
                        "decoder_integration": args.decoder_integration,
                    }
                    if args.multiple_decoder and args.decode_embedding == 2:
                        target_mask = batch['target_mask'][i]
                    else:
                        target_mask = batch['target_mask']
                    kwargs["decoder_attention_mask"] = target_mask.cuda()
                    if validation:
                        kwargs['num_beams'] = args.num_return_sequences
                        kwargs['timer'] = self.timer
                        if args.decode_embedding == 1:
                            kwargs.pop('decode_tree')
                        elif args.decode_embedding == 2 and args.multiple_decoder:
                            kwargs['decoder_index'] = i
                    else:
                        if args.gen_method == "greedy":
                            kwargs['num_beams'] = args.num_return_sequences
                        elif args.gen_method == "top_k":
                            kwargs['do_sample'] = True
                            kwargs['top_k'] = 1000
                            kwargs['top_p'] = 0.95
                            kwargs['length_penalty'] = 1.0
                            kwargs.pop('decode_tree')
                        else:
                            kwargs['length_penalty'] = 1.0
                            kwargs['num_beams'] = 2
                            kwargs['early_stopping'] = True

                    if generate_all:
                        kwargs['num_return_sequences'] = 1
                        kwargs['num_beams'] = 1
                        kwargs['eval_all_documents'] = True

                    with torch.no_grad():
                        nci_outputs = self.model.generate(**kwargs)
                    if self.args.reserve_decoder:
                        outs, scores, enc_last_hidden_state, dec_last_hidden_state, ori_decoder_last_hidden_state = nci_outputs
                    else:
                        outs, scores, enc_last_hidden_state, dec_last_hidden_state = nci_outputs
                        ori_decoder_last_hidden_state = None
                    outputs.append(
                        [outs, scores, enc_last_hidden_state, dec_last_hidden_state])

                if len(outputs) == 1:
                    outs, scores, enc_last_hidden_state, dec_last_hidden_state = outputs[0]
                else:
                    all_outs = []
                    all_scores = []
                    for outs, scores, _, _ in outputs:
                        all_outs.append(outs)
                        all_scores.append(scores)
                    outs = torch.concat(all_outs)
                    scores = torch.concat(all_scores)
                if not generate_all:
                    if args.num_return_sequences == 1:
                        nci_scores = torch.ones(
                            (input_ids.shape[0], 1), dtype=torch.float32)
                    else:
                        nci_scores = torch.tensor(scores, dtype=torch.float32).reshape(
                            input_ids.shape[0], args.num_return_sequences)
                        # nci_scores = F.softmax(nci_scores, -1)
                    assert args.decode_embedding == 2 or not args.kary
                    if args.decode_embedding == 1:
                        dec = [numerical_decoder(args, ids, output=True)
                               for ids in outs]
                    elif args.decode_embedding == 2:
                        dec, eos_idx = decode_token(args, outs)
                    else:
                        dec = [self.tokenizer.decode(ids) for ids in outs]
                else:
                    # nci_scores = F.softmax(scores, -1)
                    nci_scores = scores

            # if self.timer is not None and self.global_step >= args.timing_step:
            #     pickle.dump(self.timer, open(
            #         f'times{self.rank}.pkl', 'wb'))
            #     exit()
            if not generate_all:
                dec = dec_2d(dec, args.num_return_sequences)

                if self.document_encoder is not None or args.dataset == 'nq_dpr':
                    if self.args.codebook:
                        doc_cluster = self.pq_doc_cluster
                    else:
                        doc_cluster = self.doc_cluster

                scores_list = dec_2d(scores, args.num_return_sequences)

                torch_dec = dec
                if self.qemb_projection is not None:
                    clusters = torch_dec[:, :args.cat_cluster_centroid, :]
                if self.pemb_projection is not None:
                    pemb_dec = torch_dec[:, :args.cluster_position_topk, :]
                if self.unified_projection is not None:
                    clusters = pemb_dec = torch_dec[:,
                                                    :args.cluster_position_topk, :]
                dec = dec.tolist()
                if self.document_encoder is None:
                    eos_idx = dec_2d(
                        eos_idx, args.num_return_sequences).squeeze(2)
                    eos_idx = eos_idx.tolist()
                    dec = [[dd[0:ii]
                            for dd, ii in zip(d, i)] for d, i in zip(dec, eos_idx)]

        if self.timer is not None and args.timing_infer_step > 0:
            ending_time = time()
            self.timer['nci'].append(ending_time - start_time)
            start_time = time()

        pool_size = max(self.args.recall_num)
        results = []
        if self.document_encoder is None or args.recall_level in ('coarse', 'both'):
            assert args.eval_all_documents == 0
            if args.dataset == 'nq_dpr':
                offsets, array = self.nq_eval
                for i in range(len(texts)):
                    text = texts[i]
                    d = dec[i]
                    if not validation:
                        self.coarse_log_texts.add((text, d, scores_list[i]))
                    qind = batch['query_indices'][i]
                    ind = None
                    for j, newid in enumerate(d):
                        for oldid in doc_cluster.get(tuple(newid), []):
                            if qind in array[offsets[oldid]:offsets[oldid+1]]:
                                ind = j
                                break
                        if ind is not None:
                            break
                    length = sum([len(doc_cluster.get(tuple(dd), []))
                                  for dd in d])
                    results.append((text, length, [ind]))
            else:
                for text, d, gts, score in zip(texts, dec, batch['new_ids'], scores_list):
                    if not validation:
                        self.coarse_log_texts.add((text, d, gts, score))
                    if self.use_pq_topk_label:
                        indices = []
                        for gt in gts:
                            cur_ind = pool_size + 1
                            for g in gt:
                                if g in d:
                                    cur_ind = min(cur_ind, d.index(g))
                            if cur_ind == pool_size + 1:
                                cur_ind = None
                            indices.append(cur_ind)
                        indices = tuple(indices)
                    else:
                        indices = tuple(
                            d.index(gt) if gt in d else None for gt in gts)
                    if self.document_encoder:
                        length = sum([len(doc_cluster.get(tuple(dd), []))
                                      for dd in d])
                    else:
                        length = None
                    results.append((text, length, indices))
        if self.document_encoder and args.recall_level in ('fine', 'both'):
            # calculate recall after ranking
            # decoder_last_hidden_state (bs, args.max_output_length, dim)
            if args.eval_all_documents:
                dec_flatten = None
                enc_last_hidden_state = None
                ori_decoder_last_hidden_state = None
                dec_last_hidden_state = None
            else:
                if args.codebook:
                    dec_flatten = outs[:, -2]
                else:
                    dec_flatten = outs.gather(1, eos_idx).squeeze(-1)
                if ori_decoder_last_hidden_state is not None:
                    ori_decoder_last_hidden_state = ori_decoder_last_hidden_state.unsqueeze(
                        1).expand(-1, args.num_return_sequences, -1, -1).reshape(-1, ori_decoder_last_hidden_state.shape[-2], ori_decoder_last_hidden_state.shape[-1])
            if args.document_encoder == 'ance':
                qinput_ids = input_ids
                qmask = attention_mask
            else:
                qinput_ids = batch['qenc_source_ids'].cuda()
                qmask = batch['qenc_source_mask'].cuda()
            query_embedding = self.get_query_embedding(
                qinput_ids,
                qmask,
                enc_last_hidden_state,
                ori_decoder_last_hidden_state,
                dec_last_hidden_state,
                dec_flatten,
                clusters,
                flatten=True,
            )
            if args.query_encoder == 'twin' and not args.eval_all_documents:
                query_embedding = query_embedding.unsqueeze(
                    1).expand(-1, args.num_return_sequences, -1).reshape(-1, query_embedding.shape[-1])

            batch_size = self.args.encode_batch_size
            if args.eval_all_documents:
                ndoc = len(self.all_embeddings)
                nquery = len(query_embedding)
                device = query_embedding.device
                stack_scores = torch.empty(
                    [nquery, 0, ], dtype=torch.float, device=device)
                sorted_docs = torch.empty(
                    [nquery, 0, ], dtype=torch.int32, device=device)
                for start in range(0, ndoc, batch_size):
                    ending = min(start+batch_size, ndoc)
                    cur_range = slice(start, ending)
                    cur_embedding = self.all_embeddings[cur_range]
                    if args.use_topic_model:
                        cur_index = self.doc2index_mapping[cur_range]
                        new_shape = [nci_scores.shape[0], ] + \
                            list(cur_index.shape)
                        topic_proba = nci_scores[:,
                                                 cur_index.view(-1)].view(*new_shape)
                    if args.infer_reconstruct_vector == 0:
                        cur_embedding = cur_embedding.cuda()
                    if self.pemb_projection is not None:
                        cur_embedding = self.pemb_projection(
                            cur_embedding,
                            pred_clusters=pemb_dec,
                            gt_clusters=[self.pq_mapping[did]
                                         for did in range(start, ending)],
                        )
                    # elif self.unified_projection is not None:
                    #     cur_embedding = self.unified_projection.encode_passage(
                    #         cur_embedding,
                    #         pemb_dec,
                    #         [self.pq_mapping[did]
                    #             for did in range(start, ending)],
                    #     )
                    if self.pemb_projection is not None or self.unified_projection is not None:
                        output = self.document_encoder.generate(
                            query_embedding.unsqueeze(1),
                            p_reps=cur_embedding,
                            bmm=True,
                        )
                    else:
                        output = self.document_encoder.generate(
                            query_embedding,
                            p_reps=cur_embedding,
                            bmm=False,
                        )
                    doc_proba = 0
                    if self.additional_reconstruct:
                        doc_proba = self.all_doc_proba[cur_range].cuda()
                    new_scores = self.get_inference_scores(
                        topic_proba, doc_proba, output.scores, eval_all=True)
                    new_docs = torch.arange(
                        start, ending, device=sorted_docs.device, dtype=sorted_docs.dtype).unsqueeze(0).expand(nquery, -1)
                    scores = torch.cat([stack_scores, new_scores], dim=-1)
                    docs = torch.cat([sorted_docs, new_docs], dim=-1)
                    assert scores.shape == docs.shape
                    stack_scores, doc_indices = torch.topk(
                        scores, k=min(scores.shape[-1], pool_size), dim=-1)
                    sorted_docs = docs.gather(-1, doc_indices)
                if args.save_hard_neg:
                    # only calculate simple interact score!
                    for didx in range(nquery):
                        if args.dataset != 'nq_dpr':
                            cur_did = batch['doc_ids'][didx]
                            doc_embedding = self.all_embeddings[cur_did].cuda()
                            if self.pemb_projection is not None:
                                doc_embedding = self.pemb_projection(
                                    doc_embedding,
                                    pred_clusters=pemb_dec[didx],
                                    gt_clusters=self.pq_mapping[cur_did[0]],
                                )
                            # elif self.unified_projection is not None:
                            #     doc_embedding = self.unified_projection.encode_passage(
                            #         doc_embedding,
                            #         pemb_dec[didx],
                            #         self.pq_mapping[cur_did[0]],
                            #     )
                            gt_output = self.document_encoder.generate(
                                query_embedding[didx], p_reps=doc_embedding)
                            gt_output = ','.join(
                                [str(sc.item()) for sc in gt_output.scores])
                        else:
                            gt_output = ''
                        self.hn_log_texts.add((
                            texts[didx],
                            gt_output,
                            ','.join([str(sd.item())
                                      for sd in sorted_docs[didx][:args.save_hard_neg]]),
                            ','.join([str(ss.item())
                                      for ss in scores[didx][:args.save_hard_neg]]),
                        ))
                result_docs = sorted_docs.tolist()
                ndoc = [ndoc for _ in query_embedding]
            else:
                ndoc = []
                result_docs = []
                q_ind = 0
                for didx, cur_dec in enumerate(dec):
                    cur_ndoc = 0
                    scores = []
                    docs = []
                    if args.knn_topk_by_step:
                        stack_scores = torch.empty(
                            [0, ], dtype=torch.float, device=query_embedding.device)
                    sorted_docs = np.empty([0, ], dtype=np.int32)
                    for i, d in enumerate(cur_dec):
                        if args.codebook:
                            d = tuple(d)
                        else:
                            d = tuple(d[:eos_idx[i]])
                        cur_docs = doc_cluster.get(d, None)
                        if self.additional_reconstruct and args.doc_multiclus > 1:
                            cur_inclus_index = self.pq_inclus_index.get(
                                d, None)
                            if cur_inclus_index is not None:
                                cur_inclus_index = cur_inclus_index.unsqueeze(
                                    -1)
                        if cur_docs is not None:
                            # here add the length of cluster into ndoc
                            cur_ndoc += len(cur_docs)
                            if args.infer_reconstruct_vector == 1:
                                reconstruct_doc_embedding = self.pq.get_reconstruct_vector(
                                    torch_dec[didx][i])
                                doc_embedding = reconstruct_doc_embedding.unsqueeze(
                                    0).expand(len(cur_docs), -1)
                            else:
                                doc_embedding = self.all_embeddings[cur_docs]
                            if self.additional_reconstruct:
                                all_doc_proba = self.all_doc_proba[cur_docs]
                            topic_proba = nci_scores[didx][i].item()
                            for start in range(0, len(doc_embedding), batch_size):
                                cur_range = slice(start, start+batch_size)
                                cur_embedding = doc_embedding[cur_range]
                                if args.infer_reconstruct_vector == 0:
                                    cur_embedding = cur_embedding.cuda()
                                if self.pemb_projection is not None:
                                    cur_embedding = self.pemb_projection(
                                        cur_embedding,
                                        ranks=i,
                                        logprobs=topic_proba,
                                    )
                                elif self.unified_projection is not None and args.infer_reconstruct_vector == 1:
                                    cur_embedding = self.unified_projection.encode_passage(
                                        cur_embedding,
                                        pemb_dec[didx],
                                        [self.pq_mapping[did]
                                            for did in cur_docs[cur_range]],
                                        ranks=i,
                                    )
                                output = self.document_encoder.generate(
                                    query_embedding[q_ind], p_reps=cur_embedding)
                                doc_proba = 0
                                if self.additional_reconstruct:
                                    if args.doc_multiclus > 1:
                                        doc_proba = all_doc_proba[cur_range].gather(
                                            1, cur_inclus_index[cur_range])
                                    else:
                                        doc_proba = all_doc_proba[cur_range]
                                    doc_proba = doc_proba.squeeze(1).cuda()
                                new_scores = self.get_inference_scores(
                                    topic_proba, doc_proba, output.scores)
                                if self.args.knn_topk_by_step:
                                    new_docs = cur_docs[cur_range]
                                    scores = torch.cat(
                                        [stack_scores, new_scores])
                                    docs = np.concatenate(
                                        [sorted_docs, new_docs])
                                    assert len(scores) == len(docs)
                                    stack_scores, doc_indices = torch.topk(
                                        scores, k=min(len(scores), pool_size))
                                    sorted_docs = docs[doc_indices.cpu(
                                    ).numpy()]
                                else:
                                    scores.append(new_scores)
                            if not self.args.knn_topk_by_step:
                                docs.append(cur_docs)
                        q_ind += 1
                    if not self.args.knn_topk_by_step:
                        if len(scores) > 0:
                            scores = torch.cat(scores)
                            docs = np.concatenate(docs, dtype=int)
                            if self.args.doc_multiclus > 1:
                                udocs, uindices = np.unique(
                                    docs, return_inverse=True)
                                uscores = scores.new_zeros(*udocs.shape)
                                if self.args.multiclus_score_aggr == 'max':
                                    uscores[:] = -float('inf')
                                for ui, s in zip(uindices, scores):
                                    if self.args.multiclus_score_aggr == 'add':
                                        uscores[ui] += s
                                    else:
                                        uscores[ui] = torch.max(uscores[ui], s)
                                docs = udocs
                                scores = uscores
                            scores, index = torch.sort(
                                scores, descending=True)
                            sorted_docs = docs[index.cpu().numpy()].tolist()
                        else:
                            sorted_docs = []
                    else:
                        sorted_docs = sorted_docs.tolist()
                    result_docs.append(sorted_docs)
                    ndoc.append(cur_ndoc)
                    if args.save_hard_neg:
                        if args.dataset != 'nq_dpr':
                            # only calculate simple interact score!
                            cur_did = batch['doc_ids'][didx]
                            doc_embedding = self.all_embeddings[cur_did].cuda()
                            if self.pemb_projection is not None:
                                doc_embedding = self.pemb_projection(
                                    doc_embedding,
                                    pred_clusters=pemb_dec[didx],
                                    gt_clusters=self.pq_mapping[cur_did[0]],
                                )
                            # elif self.unified_projection is not None:
                            #     doc_embedding = self.unified_projection.encode_passage(
                            #         doc_embedding,
                            #         pemb_dec[didx],
                            #         self.pq_mapping[cur_did[0]],
                            #     )
                            cur_qind = q_ind - \
                                len(cur_dec) if args.eval_all_documents == 0 else didx
                            gt_output = self.document_encoder.generate(
                                query_embedding[cur_qind], p_reps=doc_embedding)
                            gt_output = ','.join(
                                [str(sc.item()) for sc in gt_output.scores])
                        else:
                            gt_output = ''
                        self.hn_log_texts.add((
                            texts[didx],
                            gt_output,
                            ','.join([str(sd)
                                      for sd in sorted_docs[:args.save_hard_neg]]),
                            ','.join([str(ss.item())
                                      for ss in scores[:args.save_hard_neg]]),
                        ))

                assert q_ind == len(query_embedding)

            if self.timer is not None and args.timing_infer_step > 0:
                ending_time = time()
                self.timer['knn'].append(ending_time - start_time)

            if args.dataset == 'nq_dpr':
                offsets, array = self.nq_eval
                for i in range(len(result_docs)):
                    qind = batch['query_indices'][i]
                    res_doc = result_docs[i]
                    if not validation:
                        self.fine_log_texts.add((texts[i], res_doc))
                    ind = None
                    for j, res in enumerate(res_doc):
                        if qind in array[offsets[res]:offsets[res+1]]:
                            ind = j
                            break
                    if args.recall_level == 'fine':
                        results.append((texts[i], ndoc[i], [ind]))
                    else:
                        assert ndoc[i] == results[i][1]
                        results[i] += ([ind],)
            else:
                for i, gts in enumerate(batch['doc_ids']):
                    res_doc = result_docs[i]
                    if not validation:
                        self.fine_log_texts.add((texts[i], res_doc, gts))
                    indices = tuple(res_doc.index(
                        gt) if gt in res_doc else None for gt in gts)
                    if args.recall_level == 'fine':
                        results.append((texts[i], ndoc[i], indices))
                    else:
                        assert ndoc[i] == results[i][1]
                        results[i] += (indices,)

        if self.timer is not None:
            if self.timing_step_for_infer >= args.timing_infer_step:
                pickle.dump(self.timer, open(
                    f'times{self.args.num_return_sequences}.pkl', 'wb'))
                exit()
            self.timing_step_for_infer += 1

        return results

    def handle_infer_results(self, inf_result_cache, validation=False):
        rank = self.rank
        nrank = self.nrank
        get_both = (self.args.recall_level == 'both')
        if not validation:
            if self.args.recall_level in ('coarse', 'both'):
                self.coarse_log_texts.wrapped_merge()
                del self.coarse_log_texts
            if self.args.recall_level in ('fine', 'both'):
                self.fine_log_texts.wrapped_merge()
                del self.fine_log_texts
        pkl_inf_save_path = '.'.join(
            self.args.inf_save_path.split('.')[:-1]) + '.pkl'
        LogPklFile.log_file(inf_result_cache, pkl_inf_save_path,
                            rank, nrank, self.barrier, 'list')
        with open(pkl_inf_save_path, 'rb') as fr:
            inf_result_cache = pickle.load(fr)
        queries = {}
        # the `length` below is the ndoc@cluster-k for each query
        if get_both:
            for (q, length, cindex, findex) in inf_result_cache:
                new_value = [length, findex, cindex]
                queries[q] = new_value
        else:
            for (q, length, index) in inf_result_cache:
                new_value = [length, index]
                queries[q] = new_value
        recalls = {r: 0 for r in self.args.recall_num}
        mrrs = {m: 0 for m in self.args.recall_num}
        hitrates = {h: 0 for h in self.args.recall_num}
        if get_both:
            # coarse recall
            recnums = sorted([
                rn for rn in self.args.recall_num if rn <= self.args.num_return_sequences])
            if recnums[-1] != self.args.num_return_sequences:
                recnums.append(self.args.num_return_sequences)
            crecalls = {r: 0 for r in recnums}
            cmrrs = {m: 0 for m in recnums}
            chitrates = {h: 0 for h in recnums}
        coarse_recall = 0
        coarse_mrr = 0
        coarse_hit = 0
        nsamples = 0

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

        for q, v in queries.items():
            v_valid, min_valid = get_metric(v[1], recalls, mrrs, hitrates)
            if self.document_encoder is not None:
                if get_both:
                    get_metric(v[2], crecalls, cmrrs, chitrates)
                elif self.args.recall_level == 'fine':
                    coarse_recall += len(v_valid) / len(v[1])
                    coarse_mrr += 1 / \
                        (min_valid + 1) if min_valid is not None else 0
                    coarse_hit += (len(v_valid) > 0)
                nsamples += v[0]
        nqueries = len(queries)
        if self.document_encoder is not None:
            if self.args.recall_level == 'fine':
                ckey = f'cluster{self.args.num_return_sequences}'
                recalls[ckey] = coarse_recall
                mrrs[ckey] = coarse_mrr
                hitrates[ckey] = coarse_hit
            # finally get the average ndoc@cluster100 here
            ndoc = nsamples / nqueries
        else:
            ndoc = None
        for recnum in recalls:
            recalls[recnum] /= nqueries
            mrrs[recnum] /= nqueries
            hitrates[recnum] /= nqueries
        if get_both:
            for recnum in crecalls:
                crecalls[recnum] /= nqueries
                cmrrs[recnum] /= nqueries
                chitrates[recnum] /= nqueries
        self.barrier()
        if rank == 0:
            for i in range(nrank):
                self.safe_rm(pkl_inf_save_path[:-2] + f'_{i}')
        if get_both:
            return (recalls, crecalls), (mrrs, cmrrs), (hitrates, chitrates), ndoc
        else:
            return recalls, mrrs, hitrates, ndoc

    def handle_validate_sampleloss(self, outputs):
        avg_val_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", avg_val_loss)
        print('Average evaluation loss:', avg_val_loss.item())

    @staticmethod
    def output_results(func, metrices):
        for name, metric in metrices.items():
            for k, v in metric.items():
                func(f"{name}{k}", v)

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.model.eval()
        if not self.synced:
            self.on_fit_start()
        print('vaildation epoch start')
        args = self.args
        in_train = args.mode == 'train'
        if not in_train and args.recall_level != 'finesampleloss':
            if args.custom_save_path is not None:
                path_prefix = args.custom_save_path[:-4]
            else:
                path_prefix = args.inf_save_path[:-4]
            if args.recall_level in ('coarse', 'both'):
                self.coarse_log_texts = LogTxtFile(
                    f'{path_prefix}_coarse.tsv', self.rank, self.nrank, self.barrier, maxline=50000)
            if args.recall_level in ('fine', 'both'):
                self.fine_log_texts = LogTxtFile(
                    f'{path_prefix}_fine.tsv', self.rank, self.nrank, self.barrier, maxline=50000)
            if args.save_hard_neg:
                self.hn_log_texts = LogTxtFile(
                    f'{path_prefix}_hn{args.save_hard_neg}.tsv', self.rank, self.nrank, self.barrier, maxline=50000)
        if args.validation_gen_val:

            # query_file = osp.join(args.data_dir, 'dev_mevi_dedup.tsv')
            # output_path = osp.join(args.data_dir, '...')
            # self.gen_all_query_embedding(query_file, output_path)

            self.gen_doc_embedding()
            # exit() # only generate embedding
            if args.codebook:
                if in_train and not args.tie_nci_pq_centroid:
                    if self.trainer is not None and self.trainer.sanity_checking:
                        self.pq.initialize(
                            args.pq_path, self.doc_emb, self.rank, args.seed - 1, args.pq_cluster_path, args.encode_batch_size)
                    elif not args.pq_update_after_eval and args.pq_update_method in ('kmeans', 'faiss') and (not self.alt_cross_epoch or self.stage == 1):
                        self.pq.unsupervised_update_codebook(
                            self.doc_emb, self.rank, args.seed + self.current_epoch, args.align_clustering)
                elif not in_train and args.infer_ckpt is None:
                    self.pq.initialize(
                        args.pq_path, self.doc_emb, self.rank, args.seed - 1, args.pq_cluster_path, args.encode_batch_size)
                self.gen_pq_doc_cluster()
                if self.use_pq_topk_label:
                    self.gen_pq_doc_topk(
                        return_cluster=args.doc_multiclus > 1)
                if self.qemb_projection is not None or (self.unified_projection is not None and not args.cluster_adaptor_trainable_token_embedding):
                    self.gen_all_reconstruct()
                if args.use_topic_model:
                    self.gen_doc2index_mapping()
        if in_train and args.validation_release_traindataset:
            self.train_dataset.release_dataset()
        if args.only_gen_rq:
            exit()
        if args.dataset == 'nq_dpr':
            offsets = np.memmap(
                osp.join(args.data_dir, 'test_inverse_offsets.bin'), mode='r', dtype=np.int32)
            array = np.memmap(
                osp.join(args.data_dir, 'test_inverse_array.bin'), mode='r', dtype=np.int32)
            self.nq_eval = (offsets, array)
        return super().on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.args.recall_level == 'finesampleloss':
            with torch.no_grad():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor)
                         else v for k, v in batch.items()}
                loss, orig_loss, dist_loss, co_loss, q_emb_distill_loss, weight_distillation, centroid_update_loss = self._step(
                    batch)
                results = loss + co_loss + centroid_update_loss
            self.log('val_loss', results)
        else:
            results = self.infer(batch, validation=True)
        return results

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        print('validation end')
        args = self.args
        if args.dataset == 'nq_dpr':
            del self.nq_eval
        in_train = args.mode == 'train'
        if (self.trainer is None or not self.trainer.sanity_checking) \
                and args.codebook \
                and in_train \
                and not args.tie_nci_pq_centroid \
                and args.pq_update_after_eval \
                and self.args.pq_update_method in ('kmeans', 'faiss') \
                and (not self.alt_cross_epoch or self.stage == 1):
            self.pq.unsupervised_update_codebook(
                self.doc_emb, self.rank, self.args.seed + self.current_epoch, self.args.align_clustering)
        self.barrier()
        if self.document_encoder and args.recall_level in ('fine', 'both') and not args.fixdocenc and not self.nci_vq_alt_epoch:
            del self.all_embeddings
        if args.validation_gen_val and args.codebook:
            npqclus = len(self.pq_doc_cluster)
            if in_train:
                self.log('npqclus', npqclus)
            if self.rank == 0:
                print(f'npqclus: {npqclus}')
        if in_train and args.recall_level == 'finesampleloss':
            self.handle_validate_sampleloss(outputs)
        else:
            inf_result_cache = [
                item for sublist in outputs for item in sublist]
            recalls, mrrs, hitrates, ndoc = self.handle_infer_results(
                inf_result_cache, validation=in_train)
            if self.args.recall_level == 'both':
                recalls, crecalls = recalls
                mrrs, cmrrs = mrrs
                hitrates, chitrates = hitrates
            if args.document_encoder and args.save_hard_neg:
                self.hn_log_texts.wrapped_merge()
                del self.hn_log_texts
            if self.rank == 0:
                self.output_results(
                    print, {'recall': recalls, 'mrr': mrrs, 'hitrate': hitrates})
                if self.args.recall_level == 'both':
                    self.output_results(
                        print, {'cluster_recall': crecalls, 'cluster_hitrate': chitrates})
                if args.document_encoder:
                    print(f"ndocs@cluster{args.num_return_sequences}: {ndoc}")

                    if in_train and args.recall_level in ('fine', 'both') and not args.fixdocenc and args.save_top_k > 0:
                        topk_recalls, topk_embedding_file = self.topk_embeddings
                        currec = tuple(recalls[k]
                                       for k in self.args.recall_num)
                        idx = None
                        for i, rec in enumerate(topk_recalls):
                            if currec >= rec:
                                idx = i
                                break
                        rm_recall = currec
                        rm_embedding_file = self.current_embedding_path
                        if idx is not None:
                            topk_recalls.insert(idx, currec)
                            rm_recall = topk_recalls.pop()
                            topk_embedding_file.insert(
                                idx, self.current_embedding_path)
                            rm_embedding_file = topk_embedding_file.pop()
                        print(
                            f'Current top{args.save_top_k} recalls', self.topk_embeddings)
                        if rm_embedding_file is args.embedding_path:
                            rm_embedding_file = None
                        if rm_embedding_file is not None:
                            print(
                                f'Remove {rm_embedding_file} with recall {rm_recall}.')
                            self.safe_rm(
                                osp.join(args.data_dir, rm_embedding_file))
                        assert len(topk_recalls) == len(
                            topk_embedding_file) == self.args.save_top_k
            if in_train:
                for k, v in recalls.items():
                    self.log(f"recall{k}", v)
                for k, v in mrrs.items():
                    self.log(f"mrr{k}", v)
                for k, v in hitrates.items():
                    self.log(f"hitrate{k}", v)
                self.output_results(
                    self.log, {'recall': recalls, 'mrr': mrrs, 'hitrate': hitrates})
                if self.args.recall_level == 'both':
                    self.output_results(
                        self.log, {'cluster_recall': crecalls, 'cluster_hitrate': chitrates})
                if args.document_encoder:
                    self.log(f"ndocs@cluster{args.num_return_sequences}", ndoc)
            else:
                if self.rank == 0:
                    with open(args.metric_path, 'w') as fw:

                        def func(*args):
                            print(*args, file=fw)

                        self.output_results(
                            func, {'recall': recalls, 'mrr': mrrs, 'hitrate': hitrates})
                        if self.args.recall_level == 'both':
                            self.output_results(
                                func, {'cluster_recall': crecalls, 'cluster_hitrate': chitrates})
                        if args.document_encoder:
                            print(
                                f"ndocs@cluster{args.num_return_sequences}: {ndoc}", file=fw)
