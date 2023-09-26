import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
import socket
from time import time
import pickle
from transformers import AutoTokenizer


def move_to_tmp(filepath):
    dirpath, filename = osp.split(filepath)
    assert osp.isdir(dirpath)
    return osp.join('/tmp', filename)


def safe_rm(file_path):
    if osp.isfile(file_path):
        os.remove(file_path)


def get_document_encoder(model_path):
    from document_encoder import DocumentEncoder
    config_path = None
    if model_path.endswith('ar2g_nq_finetune.pkl'):
        config_path = osp.join(osp.split(model_path)[0], 'ernie-2.0-base-en')
    elif model_path.endswith('ar2g_marco_finetune.pkl'):
        config_path = osp.join(osp.split(model_path)[0], 'co-condenser-marco-retriever')
    document_encoder = DocumentEncoder.build(
        model_path,
        config_path,
        tied=True,
        negatives_x_sample=True,
    )
    return document_encoder


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def gen_query_embedding(
    rank,
    query_file,
    document_encoder,
    tokenizer,
    output_path,
    batch_size,
    dim,
    gpus,
    query_length=32,
):
    nrank = len(gpus)
    if nrank > 1:
        dist.init_process_group(
            "nccl", rank=rank, world_size=nrank, timeout=datetime.timedelta(hours=24))
    device = torch.device(f'cuda:{gpus[rank]}')
    torch.cuda.set_device(device)
    document_encoder = document_encoder.to(device)
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
    for start in tqdm(range(cur_start, cur_ending, batch_size)):
        ending = min(start+batch_size, cur_ending)
        batch_query = df[start:ending]
        output = tokenizer.batch_encode_plus(batch_query, max_length=query_length,
                                             padding='max_length', truncation=True, return_tensors="pt")
        output = output.to(device)
        qemb = document_encoder.encode_query(output)
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
                safe_rm(output_path[:-4] + f'_{i}.bin')


def gen_doc_embedding(
    document_dir,
    document_encoder,
    output_path,
    batch_size,
    dim,
    doc_length=128,
    rank=None,
    nrank=None,
):
    from main_models import IndexedData
    if rank is None:
        rank = 0
        nrank = 1
    # get document tokens
    token_file, mask_file = [
        osp.join(document_dir, suffix) for suffix in ['all_document_tokens.bin', 'all_document_masks.bin']]
    all_tokens = np.memmap(
        token_file, dtype=np.int64, mode='r').reshape(-1, doc_length)
    all_masks = np.memmap(
        mask_file, dtype=np.int64, mode='r').reshape(-1, doc_length)
    all_data = IndexedData(content=(all_tokens, all_masks))

    # generate all doc embeddings
    embedpath_prefix = move_to_tmp(output_path)[:-4]
    assert rank is not None and nrank is not None
    part_embedding_path = f'{embedpath_prefix}_{rank}.bin'
    num_docs = len(all_data)
    num_docs_per_worker = num_docs // nrank
    start = num_docs_per_worker * rank
    if rank + 1 == nrank:
        ending = num_docs
    else:
        ending = start + num_docs_per_worker
    print(
        f'Generate embedding from {start} to {ending} in {num_docs} docs...')
    part_embeddings = np.memmap(
        part_embedding_path, dtype=np.float32, mode='w+', shape=(ending - start, dim))
    for ind, i in enumerate(tqdm(range(start, ending, batch_size), desc='Generate Embedding')):
        cur_start = i
        cur_ending = min(i+batch_size, ending)
        doc_token_ids, doc_token_mask = all_data[cur_start:cur_ending]
        doc_token_ids = doc_token_ids.cuda()
        doc_token_mask = doc_token_mask.cuda()
        output = document_encoder.generate(None, {'input_ids': doc_token_ids.view(
            -1, doc_length), 'attention_mask': doc_token_mask.view(-1, doc_length)})
        embedding = output.p_reps.detach().cpu().numpy()
        part_embeddings[cur_start -
                        start:cur_ending-start, :] = embedding
    part_embeddings.flush()
    del part_embeddings
    if nrank > 1:
        dist.barrier()
    if rank == 0:
        all_embeddings = np.memmap(
            output_path, dtype=np.float32, mode='w+', shape=(num_docs, dim))
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
            safe_rm(f'{embedpath_prefix}_{i}.bin')
    if nrank > 1:
        dist.barrier()


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def generate_all_query(args):
    document_encoder = get_document_encoder(args.model_path)
    if args.ckpt_path is not None:
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new = {}
        prefix = 'document_encoder.'
        len_prefix = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                nk = k[len_prefix:]
                new[nk] = v
        document_encoder.load_state_dict(new)
    tokenizer = get_tokenizer(args.tokenizer_path)
    torch.cuda.empty_cache()
    if args.nrank > 1:
        host = 'localhost'
        port = _find_free_port()
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = str(port)
        mp.spawn(gen_query_embedding, nprocs=args.nrank, args=(
            args.query_file,
            document_encoder,
            tokenizer,
            args.query_embedding_path,
            args.batch_size,
            args.dim,
            args.gpus,
        ))
    else:
        gen_query_embedding(
            0,
            args.query_file,
            document_encoder,
            tokenizer,
            args.query_embedding_path,
            args.batch_size,
            args.dim,
            args.gpus,
        )


def generate_all_document(args):
    # here the model should be T5FineTuner
    from main_models import T5FineTunerWithValidation
    ...


def profile_generate_query(
    query_file,
    document_encoder,
    tokenizer,
    step_num,
    query_length=32,
):
    device = torch.device(f'cuda:0')
    torch.cuda.set_device(device)
    document_encoder = document_encoder.to(device)
    df = pd.read_csv(
        query_file,
        names=['query', 'oldid'],
        encoding='utf-8',
        header=None,
        sep='\t',
    )['query']
    cur_start = 0
    cur_ending = step_num
    batch_size = 1
    timer = []
    for start in tqdm(range(cur_start, cur_ending)):
        ending = min(start+batch_size, cur_ending)
        batch_query = df[start:ending]
        start_time = time()
        output = tokenizer.batch_encode_plus(batch_query, max_length=query_length,
                                             padding='max_length', truncation=True, return_tensors="pt")
        output = output.to(device)
        qemb = document_encoder.encode_query(output).detach().cpu().numpy()
        ending_time = time()
        timer.append(ending_time - start_time)

    with open('timer.pkl','wb') as fw:
        pickle.dump(timer,fw)


if __name__ == '__main__':
    # simply generate embeddings for further knn or ann
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--query_embedding_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--gen_query', action='store_true', default=False)
    parser.add_argument('--timing_infer_step', type=int, default=0)
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(
        ',')] if args.gpus is not None else [0]
    args.nrank = len(args.gpus)
    if args.gen_query:
        assert args.query_file is not None and args.query_embedding_path is not None, 'Need to specify source path and target path!'
        generate_all_query(args)
    elif args.timing_infer_step > 0:
        document_encoder = get_document_encoder(args.model_path)
        tokenizer = get_tokenizer(args.tokenizer_path)
        profile_generate_query(
            args.query_file,
            document_encoder,
            tokenizer,
            args.timing_infer_step,
        )
