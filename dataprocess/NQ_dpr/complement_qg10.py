import os
import os.path as osp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12362)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, args):
    setup(rank, world_size)
    device_id = rank % torch.cuda.device_count()
    print(
        f"Start running DDP on rank {rank}/{world_size} on device {device_id}.")

    bads_path = osp.join(args.data_dir, 'bads_qg10.pkl')
    with open(bads_path, 'rb') as fr:
        bad_lines = pickle.load(fr)
    bad_docs = list(bad_lines)
    nline = len(bad_docs) // world_size
    offset = nline * rank
    left = offset
    right = offset + nline
    if rank == world_size - 1:
        right = len(bad_docs)
    bad_docs = np.array(bad_docs[left:right])
    bad_lines = {k: set(bad_lines[k]) for k in bad_docs}

    model_path = osp.join(args.ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = DDP(model.to(device_id), device_ids=[
                device_id], broadcast_buffers=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path
    )

    # format: {did, title, content}
    entries = []
    with open(args.document_path, 'r') as fr:
        for line in fr:
            ent = line.rstrip('\n').split('\t')
            entries.append(ent)
    need_qg_file = pd.DataFrame(entries, columns=['did', 'title', 'content'])

    bs = 20
    # finally_bads = []
    sample_pool = np.array(list(range(bs)))
    count_pool = np.array([len(bad_lines[bad_docs[k]])
                           for k in sample_pool])
    t = tqdm(total=len(bad_docs))
    cur_min = 0
    cur_max = 0

    with open(osp.join(args.data_dir, f"qg10_{rank}.tsv"), "w", newline='') as fw:
        while len(sample_pool) > 0:
            t.update(sample_pool[0] - cur_min)
            cur_min = sample_pool[0]
            cur_docs = bad_docs[sample_pool]
            content = need_qg_file['content'][cur_docs].tolist()
            batch_input_ids = tokenizer(content, add_special_tokens=True, max_length=args.doc_max_len,
                                        padding='max_length', truncation=True, return_tensors='pt')
            input_ids = batch_input_ids['input_ids']
            mask = batch_input_ids['attention_mask']
            num_return_sequences = int(10 - min(count_pool))
            assert num_return_sequences > 0
            generated = model.module.generate(
                input_ids.to(device_id),
                attention_mask=mask.to(device_id),
                max_length=args.query_max_len,
                do_sample=True,
                num_return_sequences=num_return_sequences,
            )
            generated = list(tokenizer.batch_decode(
                sequences=generated.tolist(), skip_special_tokens=True))
            cur_goods = []
            cur_bads = []
            for k, j in enumerate(range(0, len(generated), num_return_sequences)):
                cur_set = bad_lines[bad_docs[sample_pool[k]]]
                cur_set.update(generated[j:j+num_return_sequences])
                cur_set.discard('')
                if len(cur_set) >= 10:
                    cur_goods.append(k)
                else:
                    cur_bads.append(k)
            cur_good_docs = sample_pool[cur_goods]
            for jg in cur_good_docs:
                fw.write(f'{bad_docs[jg]}\t' +
                         '\t'.join(bad_lines.pop(bad_docs[jg])) + '\n')
            cur_bad_docs = sample_pool[cur_bads]
            cur_max = max(max(sample_pool) + 1, cur_max)
            new_docs = np.array(range(cur_max, min(
                cur_max + bs - len(cur_bad_docs), len(bad_docs))), dtype=int)
            sample_pool = np.concatenate((cur_bad_docs, new_docs))
            count_pool = np.array([len(bad_lines[bad_docs[k]])
                                   for k in sample_pool])
            fw.flush()

    cleanup()


def detect_bad():
    qg_path = osp.join(args.data_dir, 'qg10.tsv')
    bad_lines = {}
    cur_bad = False
    prev = -1
    cur_queries = set()
    with open(qg_path, 'r') as fr:
        for i, line in tqdm(enumerate(fr)):
            ents = line.split('\t')
            did = int(ents[1])
            if ents[0] == '':
                cur_bad = True
            else:
                curq = ents[0]
                if not cur_bad:
                    cur_bad = curq in cur_queries
                cur_queries.add(curq)
                if i % 10 == 9:
                    if cur_bad:
                        bad_lines[did] = list(cur_queries)
                    cur_queries = set()
                    cur_bad = False
                    for bad_did in range(prev+1, did):
                        bad_lines[bad_did] = []
                    prev = did
    print(f"Number of bad documents: {len(bad_lines)}.")
    return bad_lines


def write_to_qg_file(args, nrank):
    qg_file = osp.join(args.data_dir, 'qg10.tsv')
    new_file = osp.join(args.data_dir, 'new_qg10.tsv')
    prev = -1
    with open(qg_file, 'r') as fr, open(new_file, 'w') as fw:
        for i in range(nrank):
            fname = osp.join(args.data_dir, f'qg10_{i}.tsv')
            cur_map = {}
            with open(fname, 'r') as nfr:
                for nline in tqdm(nfr):
                    ents = nline.rstrip('\n').split('\t')[:11]
                    cur_map[int(ents[0])] = ents[1:]
            for k in sorted(cur_map.keys()):
                v = cur_map[k]
                assert len(v) == 10
                fid = int(k)
                while fid > prev + 1:
                    fline = fr.readline()
                    fents = fline.split('\t')
                    cur_fid = int(fents[1])
                    if cur_fid <= prev:
                        for _ in range(9):
                            temp = fr.readline()
                            assert int(temp.split('\t')[1]) == cur_fid
                    else:
                        if cur_fid >= fid:
                            print(fid, cur_fid, v, fents)
                        assert cur_fid < fid
                        fw.write(fline)
                        for _ in range(9):
                            fw.write(fr.readline())
                    if cur_fid == fid - 1:
                        break
                prev = fid
                for en in v:
                    fw.write(f'{en}\t{fid}\n')
        for line in fr:
            cur_fid = int(line.split('\t')[1])
            if cur_fid > prev:
                fw.write(line)


def main(fn, world_size, args):
    mp.spawn(fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    # environment variables set in command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--document_path', type=str, required=True)
    parser.add_argument('--doc_max_len', type=int, default=512)
    parser.add_argument('--query_max_len', type=int, default=64)
    parser.add_argument('--n_gen_query', type=int, default=10)
    args = parser.parse_args()
    assert osp.isdir(
        args.data_dir), f'Data directory {args.data_dir} not exists!'

    ngpu = torch.cuda.device_count()
    print(f"Use {ngpu} GPUs!")

    bads_path = osp.join(args.data_dir, 'bads_qg10.pkl')

    if not osp.exists(bads_path):
        bad_lines = detect_bad()
        with open(bads_path, 'wb') as fw:
            pickle.dump(bad_lines, fw)
    main(run, ngpu, args)

    # postprocess
    write_to_qg_file(args, ngpu)
