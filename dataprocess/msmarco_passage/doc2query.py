import os
import os.path as osp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import trange
from argparse import ArgumentParser
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, args):
    setup(rank, world_size)
    device_id = rank % torch.cuda.device_count()
    print(
        f"Start running DDP on rank {rank}/{world_size} on device {device_id}.")

    model_path = osp.join(args.data_dir, "ckpts/doc2query-t5-base-msmarco")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = DDP(model.to(device_id), device_ids=[
                device_id], broadcast_buffers=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path
    )

    ## format: {did, title, content}
    need_qg = osp.join(args.data_dir, 'raw/corpus.tsv')

    # format: {did, qg, qid(qg_{did}_i)}
    qg_save = osp.join(
        args.data_dir, f'origin/qg{args.n_gen_query}.tsv')
    bs = 20
    # with open(f"{qg_save}_{rank}", "a", newline='') as fw:
    with open(f"{qg_save}_{rank}", "w", newline='') as fw:

        need_qg_file = pd.read_csv(need_qg, names=['did', 'title', 'content'], header=None, sep='\t', dtype={
                                   'did': int, 'title': str, 'content': str})
        nline = len(need_qg_file) // world_size
        offset = nline * rank
        # finished = {
        #     0: 2947274,
        #     1: 2947274 * 2 - nline,
        #     2: 2947274 * 3 - nline - nline,
        # }
        left = offset
        right = offset + nline
        if rank == world_size - 1:
            right = len(need_qg_file)
        need_qg_file = need_qg_file[left:right]

        for i in trange(0, len(need_qg_file), bs):
            # if i + bs < finished[rank]:
            #     continue
            next_n_lines = need_qg_file['content'][i:i+bs].tolist()
            batch_input_ids = tokenizer(next_n_lines, add_special_tokens=True, max_length=args.doc_max_len,
                                        padding='max_length', truncation=True, return_tensors='pt')

            generated = model.module.generate(
                batch_input_ids['input_ids'].to(device_id),
                attention_mask=batch_input_ids['attention_mask'].to(device_id),
                max_length=args.query_max_len,
                # min_length=args.query_max_len,
                do_sample=True,
                # top_k=2,
                num_return_sequences=args.n_gen_query,
                # top_k=8,
                # top_p=0.9
            )

            generated = tokenizer.batch_decode(
                sequences=generated.tolist(), skip_special_tokens=True)

            cur_offset = offset + i
            for index, g in enumerate(generated):
                doc_ind = index // args.n_gen_query + cur_offset
                fw.write(
                    '\t'.join([str(g), str(need_qg_file['did'][doc_ind])]) + '\n')
                fw.flush()

    cleanup()


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
    parser.add_argument('--doc_max_len', type=int, default=512)
    parser.add_argument('--query_max_len', type=int, default=64)
    parser.add_argument('--n_gen_query', type=int, default=10)
    args = parser.parse_args()
    assert osp.isdir(
        args.data_dir), f'Data directory {args.data_dir} not exists!'

    ngpu = torch.cuda.device_count()
    print(f"Use {ngpu} GPUs!")
    main(run, ngpu, args)

    # concatinate
    qg_save = osp.join(args.data_dir, f'origin/qg{args.n_gen_query}.tsv')
    with open(qg_save, 'w') as fw:
        for i in range(ngpu):
            with open(f'{qg_save}_{i}', 'r') as fr:
                fw.writelines(fr.readlines())
