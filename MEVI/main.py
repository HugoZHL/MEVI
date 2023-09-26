import datetime
import os
import os.path as osp
import argparse
import nltk
import time
import socket
import torch
import pytorch_lightning as pl

from main_models import T5FineTuner, T5FineTunerWithValidation, l1_query_eval, custom_collate
from main_utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
if pl.__version__ < '1.6':
    # please set large timeout in init_process_group in pytorch-lightning
    # .../pytorch_lightning/plugins/training_type/ddp.py
    # .../pytorch_lightning/plugins/training_type/ddp_spawn.py
    from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
else:
    from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from tqdm import tqdm
from transformers import T5Tokenizer

nltk.download('punkt')
print(torch.__version__)  # 1.10.0+cu113
print(pl.__version__)  # 1.4.9
time_str = time.strftime("%Y%m%d-%H%M%S")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(args):
    if args.resume_from_checkpoint:
        try_modify_ckpt(args)
    if args.no_validation:
        model = T5FineTuner(args)
    else:
        model = T5FineTunerWithValidation(args)

    if args.infer_ckpt or args.nci_ckpt:
        try_load_ckpt(model, args.infer_ckpt, args.nci_ckpt, args)
    if args.penc_ckpt or args.qenc_ckpt:
        try_load_encoder_ckpt(model, args.qenc_ckpt, args.penc_ckpt)

    if args.timing_step > 0:
        num_sanity_val_steps = args.timing_step + 1
    else:
        num_sanity_val_steps = args.num_sanity_val_steps
    pl_version = pl.__version__
    if args.document_encoder is None:
        find_unused_parameters = False
    else:
        find_unused_parameters = True
    if pl_version < '1.6':
        if args.accelerator is None:
            args.accelerator = 'ddp'
        if args.use_deepspeed:
            plugins = DeepSpeedPlugin(config="ds.json")
        else:
            plugins = DDPPlugin(find_unused_parameters=find_unused_parameters)
        configs_in_version = {
            'checkpoint_callback': True,
            'plugins': plugins,
        }
    elif pl_version < '1.7':
        if args.accelerator is None:
            args.accelerator = 'cuda'
        configs_in_version = {
            'enable_checkpointing': True,
            'strategy': DDPStrategy(find_unused_parameters=find_unused_parameters),
        }
    else:
        if args.accelerator is None:
            args.accelerator = 'cuda'
        if args.use_deepspeed:
            strategy = DeepSpeedStrategy(config="ds.json")
        else:
            strategy = DDPStrategy(
                find_unused_parameters=find_unused_parameters, timeout=datetime.timedelta(hours=24))
        configs_in_version = {
            'enable_checkpointing': True,
            'strategy': strategy,
        }
    configs_in_version['accelerator'] = args.accelerator
    if args.no_validation or args.ckpt_monitor == 'train_loss' or args.recall_level == 'finesampleloss':
        if args.no_validation:
            period = args.check_val_every_n_epoch
            model_kwargs = {"save_top_k": args.save_top_k}
            if pl_version.startswith('1.4.'):
                model_kwargs["period"] = period
            elif pl_version.startswith('1.5.'):
                model_kwargs["every_n_epochs"] = period
        else:
            model_kwargs = {
                "every_n_epochs": args.check_val_every_n_epoch, "save_top_k": args.save_top_k}
        if args.monitor_name is None:
            if args.recall_level == 'finesampleloss':
                monitor = "avg_val_loss"
            else:
                monitor = "avg_train_loss"
        else:
            monitor = args.monitor_name
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename=args.tag_info + f'_{{epoch}}-{{{monitor}:.6f}}',
            save_on_train_epoch_end=True,
            monitor=monitor,
            mode="min",
            **model_kwargs,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        if args.no_validation:
            model_kwargs = {"limit_val_batches": 0, "num_sanity_val_steps": 0}
        else:
            model_kwargs = {"val_check_interval": args.val_check_interval,
                            "limit_val_batches": args.limit_val_batches, "num_sanity_val_steps": num_sanity_val_steps}
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            # amp_backend='apex',
            resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_clip_val=args.max_grad_norm,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
            replace_sampler_ddp=False,
            **configs_in_version,
            **model_kwargs,
        )
    elif args.ckpt_monitor == 'recall':
        if args.monitor_name is None:
            monitor = 'recall1'
        else:
            monitor = args.monitor_name
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename=args.tag_info + f'_{{epoch}}-{{{monitor}:.6f}}',
            monitor=monitor,
            save_on_train_epoch_end=False,
            mode="max",
            save_top_k=args.save_top_k,
            every_n_epochs=args.check_val_every_n_epoch,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            # amp_backend='apex',
            resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_clip_val=args.max_grad_norm,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
            replace_sampler_ddp=False,
            num_sanity_val_steps=num_sanity_val_steps,
            **configs_in_version,
        )
    else:
        NotImplementedError("This monitor is not implemented!")

    trainer = pl.Trainer(**train_params)

    trainer.fit(model)


def try_modify_ckpt(args):
    # only in training
    path = args.resume_from_checkpoint
    state_dict = torch.load(path, map_location='cpu')
    ori_args = state_dict['hyper_parameters']
    new_args = vars(args)
    if ori_args != new_args:
        state_dict['hyper_parameters'] = new_args
        parts = path.split('.')
        new_path = '.'.join(parts[:-1]) + '_new.' + parts[-1]
        torch.save(state_dict, new_path)
        args.resume_from_checkpoint = new_path


@torch.no_grad()
def try_load_ckpt(model=None, whole_path=None, nci_path=None, args=None):
    if args is None:
        load_encoder_only = False
        reserve_decoder = False
        not_load_document_encoder = False
    else:
        load_encoder_only = args.load_encoder_only
        reserve_decoder = args.reserve_decoder
        not_load_document_encoder = args.not_load_document_encoder
    # only in inference, only load nci part parameters
    assert whole_path is not None or nci_path is not None
    if whole_path is not None:
        state_dict = torch.load(whole_path, map_location='cpu')
        # print(state_dict)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        bad_params = [
            "model.decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
            "model.ori_decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
            "document_encoder.lm_q.decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
            "document_encoder.lm_p.decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
        ]
        state_dict = {k: v for k, v in state_dict.items()
                      if k not in bad_params}
        if not_load_document_encoder:
            params = model.state_dict()
            for k, v in state_dict.items():
                if not k.startswith('document_encoder.'):
                    params[k].copy_(v)
        else:
            try:
                model.load_state_dict(state_dict)
            except:
                print(
                    '[Warning] Fail to load_state_dict; load each parameter instead.')
                params = model.state_dict()
                for k, v in state_dict.items():
                    params[k].copy_(v)
        del state_dict
    else:
        state_dict = torch.load(nci_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # ori_vocab_size = None
        params = model.model.state_dict()
        for k, v in state_dict.items():
            if k.startswith('model.'):
                k = k[6:]
            isdecoder = k.startswith('decoder.')
            if load_encoder_only and isdecoder:
                continue
            if reserve_decoder and isdecoder:
                k = 'ori_' + k
            if k not in params or params[k].shape != v.shape:
                print(f'Bad parameter {k}.')
                continue
            params[k].copy_(v)
        del state_dict


@torch.no_grad()
def try_load_encoder_ckpt(model, qenc_path=None, penc_path=None):
    params = model.model.state_dict()
    if qenc_path is not None:
        new_state_dict = torch.load(qenc_path, map_location='cpu')
        for k in new_state_dict:
            if k.startswith('document_encoder.lm_q.'):
                params[k].copy_(new_state_dict[k])
    if penc_path is not None:
        new_state_dict = torch.load(penc_path, map_location='cpu')
        for k in new_state_dict:
            if k.startswith('document_encoder.lm_p.'):
                params[k].copy_(new_state_dict[k])


def inference(args):
    model = T5FineTunerWithValidation(args, train=False)
    try_load_ckpt(
        model,
        whole_path=args.infer_ckpt,
        nci_path=args.nci_ckpt,
        args=args,
    )
    if args.penc_ckpt or args.qenc_ckpt:
        try_load_encoder_ckpt(model, args.qenc_ckpt, args.penc_ckpt)
    model = model.cpu()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    num_samples = args.n_test if args.n_test >= 0 else None

    dataset = l1_query_eval(
        model, args, tokenizer, model.passage_tokenizer, num_samples, model.mapping, task='test', all_docs=model.all_docs, doc_cluster=model.doc_cluster)
    torch.cuda.empty_cache()
    model.eval()
    model.share_memory()

    nrank = len(args.n_gpu)
    torch.cuda.empty_cache()
    if nrank > 1:
        host = 'localhost'
        port = _find_free_port()
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = str(port)
        mp.spawn(partial_inference, nprocs=nrank, args=(
            args, model, dataset))
    else:
        assert nrank == 1
        partial_inference(0, args, model, dataset)


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def partial_inference(rank, args, model, dataset):
    nrank = len(args.n_gpu)
    if nrank > 1:
        dist.init_process_group(
            "nccl", rank=rank, world_size=nrank, timeout=datetime.timedelta(hours=24))
    device = torch.device(f'cuda:{args.n_gpu[rank]}')
    torch.cuda.set_device(device)
    model = model.to(device)

    sampler = DistributedSampler(
        dataset, num_replicas=nrank, rank=rank, shuffle=False)
    num_workers = 0
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)

    model.on_validation_epoch_start()
    print('Inference start...')
    # begin
    # time_begin_infer = time.time()
    nearly_last_batch = (len(dataset) // nrank - 1) // args.eval_batch_size + 1

    inf_result_cache = []
    for i, batch in enumerate(tqdm(loader, desc='Inference')):
        results = model.infer(batch)
        inf_result_cache.append(results)

    model.validation_epoch_end(inf_result_cache)
    if nrank > 1:
        dist.destroy_process_group()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def parsers_parser():
    parser = argparse.ArgumentParser()
    # wandb token, please get yours from wandb portal
    parser.add_argument('--wandb_token', type=str, default='')
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default="t5-")
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-")
    parser.add_argument('--dataset', type=str, default='marco')
    parser.add_argument('--freeze_encoder', type=int,
                        default=0, choices=[0, 1])
    parser.add_argument('--freeze_embeds', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--n_val', type=int, default=-1)
    parser.add_argument('--n_train', type=int, default=-1)
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--early_stop_callback', type=int,
                        default=0, choices=[0, 1])
    parser.add_argument('--fp_16', type=int, default=0, choices=[0, 1])
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_encoder', type=int,
                        default=1, choices=[0, 1])
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--softmax', type=int, default=0, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_cls', type=int, default=1000)
    parser.add_argument('--decode_embedding', type=int,
                        default=2, choices=[0, 1, 2])
    parser.add_argument('--output_vocab_size', type=int, default=10)
    parser.add_argument('--hierarchic_decode', type=int,
                        default=0, choices=[0, 1])
    parser.add_argument('--tie_word_embedding', type=int,
                        default=0, choices=[0, 1])
    parser.add_argument('--tie_decode_embedding', type=int,
                        default=1, choices=[0, 1])
    parser.add_argument('--gen_method', type=str, default="greedy")
    parser.add_argument('--length_penalty', type=int, default=0.8)

    parser.add_argument('--recall_num', type=str,
                        default='1,5,10,20,50,100,1000', help='[1,5,10,20,50,100]')
    parser.add_argument('--random_gen', type=int, default=0, choices=[0, 1])
    parser.add_argument('--label_length_cutoff', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    parser.add_argument('--test_set', type=str, default="dev")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=2)

    parser.add_argument('--max_input_length', type=int, default=40)
    parser.add_argument('--inf_max_input_length', type=int, default=40)
    parser.add_argument('--max_output_length', type=int, default=10)
    parser.add_argument('--doc_length', type=int, default=64)
    parser.add_argument('--contrastive_variant', type=str,
                        default="", help='E_CL, ED_CL, doc_Reweight')
    parser.add_argument('--num_return_sequences', type=int,
                        default=100, help='generated id num (include invalid)')
    parser.add_argument('--n_gpu', type=str, default='1')
    parser.add_argument('--mode', type=str, default="train",
                        choices=['train', 'eval'])
    parser.add_argument('--query_type', type=str, default='gtq_qg',
                        help='gtq -- use ground turth query;'
                             'qg -- use qg; '
                             'doc -- just use top64 doc token; '
                             'doc_aug -- use random doc token. ')
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--decoder_learning_rate', type=float, default=1e-4)
    parser.add_argument('--document_encoder_learning_rate',
                        type=float, default=5e-6)
    parser.add_argument('--projection_learning_rate', type=float, default=5e-6)
    parser.add_argument('--certain_epoch', type=int, default=None)
    parser.add_argument('--given_ckpt', type=str, default='')
    parser.add_argument('--infer_ckpt', type=str, default=None)
    parser.add_argument('--nci_ckpt', type=str, default=None)
    parser.add_argument('--qenc_ckpt', type=str, default=None)
    parser.add_argument('--penc_ckpt', type=str, default=None)
    parser.add_argument('--load_encoder_only', type=int, default=0)
    parser.add_argument('--not_load_document_encoder', type=int, default=0)
    parser.add_argument('--model_info', type=str, default='base',
                        choices=['small', 'large', 'base', '3b', '11b'])
    parser.add_argument('--id_class', type=str, default='k10_c10')
    parser.add_argument('--ckpt_monitor', type=str,
                        default='recall', choices=['recall', 'train_loss'])
    parser.add_argument('--monitor_name', type=str, default=None)
    parser.add_argument('--Rdrop', type=float,
                        default=0.15, help='default to 0-0.3')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--Rdrop_only_decoder', type=int, default=0,
                        help='1-RDrop only for decoder, 0-RDrop only for all model', choices=[0, 1])
    parser.add_argument('--Rdrop_loss', type=str,
                        default='KL', choices=['KL', 'L2'])
    parser.add_argument('--adaptor_decode', type=int,
                        default=1, help='default to 0,1')
    parser.add_argument('--adaptor_efficient', type=int,
                        default=1, help='default to 0,1')
    parser.add_argument('--adaptor_layer_num', type=int, default=4)
    parser.add_argument('--test1000', type=int,
                        default=0, help='default to 0,1')
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--contrastive', type=int, default=0)
    parser.add_argument('--embedding_distillation', type=float, default=0.0)
    parser.add_argument('--weight_distillation', type=float, default=0.0)
    parser.add_argument('--hard_negative', type=int, default=0)
    parser.add_argument('--aug_query', type=int, default=0)
    parser.add_argument('--aug_query_type', type=str,
                        default='aug_query', help='aug_query, corrupted_query')
    parser.add_argument('--sample_neg_num', type=int, default=0)
    parser.add_argument('--query_tloss', type=int, default=0)
    parser.add_argument('--weight_tloss', type=int, default=0)
    parser.add_argument('--ranking_loss', type=int, default=0)
    parser.add_argument('--disc_loss', type=int, default=0)
    parser.add_argument('--input_dropout', type=int, default=0)
    parser.add_argument('--denoising', type=int, default=0)
    parser.add_argument('--multiple_decoder', type=int, default=0)
    parser.add_argument('--decoder_num', type=int, default=1)
    parser.add_argument('--loss_weight', type=int, default=0)
    parser.add_argument('--kary', type=int, default=0)
    parser.add_argument('--tree', type=int, default=1)

    # data path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--newid_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--document_path', type=str, default=None)
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--pq_cluster_path', type=str, default=None)
    parser.add_argument('--logs_dir', type=str, default=None)
    parser.add_argument('--mapping_path', type=str, default=None)
    parser.add_argument('--cluster_path', type=str, default=None)
    parser.add_argument('--tree_path', type=str, default=None)
    parser.add_argument('--eval_train_data', type=int, default=0)
    parser.add_argument('--drop_data_rate', type=float, default=0)
    # nci
    parser.add_argument('--time_str', type=str, default=None)
    parser.add_argument('--split_data', action='store_true', default=False)
    parser.add_argument('--validation_release_traindataset',
                        action='store_true', default=False)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--timing_step', type=int, default=0)
    parser.add_argument('--timing_infer_step', type=int, default=0)
    parser.add_argument('--no_validation', action='store_true', default=False)
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--reserve_decoder', type=int, default=0)
    parser.add_argument('--decoder_integration', type=str, default='series')
    parser.add_argument('--tie_encoders', type=int, default=1)
    # mevi
    parser.add_argument('--document_encoder', type=str, default=None)
    parser.add_argument(
        '--document_encoder_from_pretrained', type=int, default=0)
    parser.add_argument('--negatives_x_sample', type=int, default=1)
    parser.add_argument('--query_encoder', type=str, default='nci')
    parser.add_argument('--eval_all_documents', type=int, default=0)
    parser.add_argument('--fixnci', action='store_true', default=False)
    parser.add_argument('--fixdocenc', action='store_true', default=False)
    parser.add_argument('--fixncienc', action='store_true', default=False)
    parser.add_argument('--fixncit5', action='store_true', default=False)
    parser.add_argument('--fixpq', action='store_true', default=False)
    parser.add_argument('--fixlmq', action='store_true', default=False)
    parser.add_argument('--fixlmp', action='store_true', default=False)
    parser.add_argument('--fixproj', action='store_true', default=False)
    parser.add_argument('--alt_granularity', type=str, default='batch')
    parser.add_argument('--alt_train', type=str, default=None)
    parser.add_argument('--nci_twin_train_ratio', type=str, default=None)
    parser.add_argument('--qtower', type=str, default='enc_dec')
    parser.add_argument('--query_embed_accum', type=str, default='maxpool')
    parser.add_argument('--co_doc_length', type=int, default=128)
    parser.add_argument('--recall_level', type=str, default='coarse')
    parser.add_argument('--co_neg_num', type=int, default=7)
    parser.add_argument('--co_neg_from', type=str, default='clus')
    parser.add_argument('--co_neg_file', type=str, default=None)
    parser.add_argument('--co_neg_clus_file', type=str, default=None)
    parser.add_argument('--save_hard_neg', type=int, default=0)
    parser.add_argument('--simans_hyper_a', type=float, default=0.5)
    parser.add_argument('--simans_hyper_b', type=float, default=0)
    parser.add_argument('--encode_batch_size', type=int, default=None)
    parser.add_argument('--knn_topk_by_step', type=int, default=0)
    parser.add_argument('--co_loss_scale', type=str, default=None)
    parser.add_argument('--no_nci_loss', type=int, default=0)
    parser.add_argument('--no_twin_loss', type=int, default=0)
    # pq
    parser.add_argument('--codebook', type=int, default=0)
    parser.add_argument('--pq_type', type=str, default='pq')
    parser.add_argument('--pq_path', type=str, default=None)
    parser.add_argument('--pq_update_method', type=str, default='grad')
    parser.add_argument('--pq_update_after_eval', type=int, default=0)
    parser.add_argument('--pq_init_method', type=str, default='kmeans')
    parser.add_argument('--pq_dist_mode', type=str, default='l2')
    parser.add_argument('--subvector_num', type=int, default=4)
    parser.add_argument('--subvector_bits', type=int, default=4)
    parser.add_argument('--pq_loss', type=str, default='mse')
    parser.add_argument('--pq_twin_loss', type=str, default='co')
    parser.add_argument('--pq_runtime_label', type=int, default=1)
    parser.add_argument('--pq_runtime_update_cluster', type=int, default=0)
    parser.add_argument('--use_gumbel_softmax', type=int, default=0)
    parser.add_argument('--pq_softmax_tau', type=float, default=1)
    parser.add_argument('--pq_hard_softmax_topk', type=int, default=0)
    parser.add_argument('--topk_sequence', type=int, default=0)
    parser.add_argument('--pq_negative', type=str, default='sample')
    parser.add_argument('--pq_negative_margin', type=float, default=1.0)
    parser.add_argument('--pq_negative_loss', type=str, default='cont')
    parser.add_argument('--tie_nci_pq_centroid', type=int, default=0)
    parser.add_argument('--aug_topk_clus', type=int, default=50)
    parser.add_argument('--aug_find_topk_from', type=str, default='vq')
    parser.add_argument('--aug_sample_topk', type=int, default=0)
    parser.add_argument('--reconstruct_for_embeddings', type=int, default=0)
    parser.add_argument('--centroid_update_loss', type=str, default='none')
    parser.add_argument('--centroid_loss_scale', type=float, default=1.)
    parser.add_argument('--infer_reconstruct_vector', type=int, default=0)
    parser.add_argument('--align_clustering', type=int, default=0)
    parser.add_argument('--query_vq_label', type=int, default=0)
    parser.add_argument('--nci_twin_alt_epoch', type=str, default=None)
    parser.add_argument('--nci_vq_alt_epoch', type=str, default=None)
    parser.add_argument('--doc_multiclus', type=int, default=1)
    parser.add_argument('--rq_topk_score', type=str, default='prod')
    parser.add_argument('--multiclus_label', type=str, default='top1')
    parser.add_argument('--multiclus_score_aggr', type=str, default='add')
    parser.add_argument('--use_topic_model', type=int, default=0)
    parser.add_argument('--topic_score_ratio', type=float, default=0.)
    parser.add_argument('--cat_cluster_centroid', type=int, default=0)
    parser.add_argument('--cluster_position_topk', type=int, default=0)
    parser.add_argument('--cluster_position_embedding',
                        type=str, default='rank')
    parser.add_argument(
        '--cluster_position_rank_reciprocal', type=int, default=1)
    parser.add_argument('--cluster_position_proj_style',
                        type=str, default='dense')
    parser.add_argument('--use_cluster_adaptor', type=int, default=0)
    parser.add_argument('--cluster_adaptor_decouple', type=int, default=0)
    parser.add_argument(
        '--cluster_adaptor_trainable_token_embedding', type=int, default=0)
    parser.add_argument(
        '--cluster_adaptor_trainable_position_embedding', type=int, default=0)
    parser.add_argument('--cluster_adaptor_head_num', type=int, default=8)
    parser.add_argument('--cluster_adaptor_layer_num', type=int, default=4)
    parser.add_argument('--only_gen_rq', type=int, default=0)
    parser.add_argument('--custom_save_path', type=str, default=None)
    # ort and deepspeed
    parser.add_argument('--use_ort', type=int, default=0, choices=[0, 1])
    parser.add_argument('--fp16_opt',
                        action='store_true', default=False)
    parser.add_argument('--use_deepspeed', type=int, default=0, choices=[0, 1])
    # ads related
    parser.add_argument('--ads_info', type=str, default='')

    parser_args = parser.parse_args()
    parser_args.recall_num = sorted(
        [int(rec) for rec in parser_args.recall_num.split(',')])
    if parser_args.reserve_decoder:
        assert parser_args.decoder_integration in ('series', 'parallel')
    elif parser_args.query_encoder == 'nci':
        parser_args.tie_encoders = 0
    assert parser_args.document_encoder in (None, 'ance', 'cocondenser', 'ar2')
    if not parser_args.document_encoder:
        parser_args.fixnci = False
        parser_args.fixdocenc = False
        parser_args.nci_twin_train_ratio = None
        parser_args.nci_twin_alt_epoch = None
        parser_args.nci_vq_alt_epoch = None
        parser_args.codebook = 0
        parser_args.tie_encoders = 0
        parser_args.save_hard_neg = 0
    else:
        assert parser_args.recall_level in (
            'both', 'coarse', 'fine', 'finesampleloss')
        assert parser_args.co_neg_from in (
            'clus', 'file', 'inter', 'interhalf', 'union', 'simans', 'simansinter', 'notclus', 'clusfile', 'interclusfile')
        if parser_args.co_neg_from not in ('clus', 'notclus'):
            assert osp.isfile(parser_args.co_neg_file) and parser_args.co_neg_file.endswith(
                ('.tsv', '.pkl'))
        if parser_args.co_neg_from.endswith('clusfile'):
            assert osp.isfile(
                parser_args.co_neg_clus_file) and parser_args.co_neg_clus_file.endswith(('.tsv', '.pkl'))
        if parser_args.save_hard_neg:
            assert parser_args.mode == 'eval'
            assert parser_args.recall_level in ('fine', 'both')
            # assert parser_args.save_hard_neg <= parser_args.recall_num[-1]
            assert parser_args.query_encoder == 'twin' and parser_args.infer_reconstruct_vector != 1 and (
                not parser_args.use_topic_model or parser_args.topic_score_ratio == 0)
        if parser_args.fixnci or parser_args.fixdocenc:
            parser_args.alt_train = None
        assert not (
            parser_args.nci_twin_train_ratio and parser_args.nci_vq_alt_epoch)
        if parser_args.nci_twin_train_ratio:
            parser_args.drop_last = True
            assert parser_args.alt_granularity in ('batch', 'epoch')
            assert parser_args.alt_train in (None, 'fix', 'loss', 'all')
        assert parser_args.query_encoder in ('nci', 'twin')
        if parser_args.eval_all_documents:
            assert parser_args.query_encoder == 'twin' and parser_args.recall_level == 'fine' and parser_args.knn_topk_by_step == 1
        # assert not parser_args.query_vq_label or parser_args.query_encoder == 'twin'

    parser_args.original_fixdocenc = parser_args.fixdocenc

    parser_args.validation_gen_val = parser_args.document_encoder and (parser_args.recall_level in ('fine', 'both') or (
        parser_args.mode == 'eval' and parser_args.recall_level == 'finesampleloss') or (
        parser_args.codebook and parser_args.recall_level != 'finesampleloss'))
    if parser_args.codebook:
        assert parser_args.pq_type in ('pq', 'opq', 'rq')
        assert parser_args.pq_update_method in (
            'grad', 'balancekmeans', 'kmeans', 'faiss', 'ema')
        assert parser_args.pq_init_method in (
            'none', 'avg', 'balancekmeans', 'kmeans', 'faiss')
        assert parser_args.pq_dist_mode in ('l2', 'ip', 'iptol2')
        assert parser_args.doc_multiclus >= 1
        assert parser_args.multiclus_label in ('top1', 'all', 'minpool')
        assert parser_args.multiclus_score_aggr in ('add', 'max')
        assert parser_args.rq_topk_score in ('prod', 'last')
        if parser_args.cluster_position_topk > 0:
            assert parser_args.cluster_position_embedding in (
                'rank', 'emb', 'score', 'scorerank')
            assert not parser_args.eval_all_documents and not (
                parser_args.cluster_position_embedding.startswith('score') and parser_args.save_hard_neg)
        if parser_args.use_topic_model:
            assert 0 <= parser_args.topic_score_ratio <= 1
            assert not parser_args.infer_reconstruct_vector
        if parser_args.doc_multiclus > 1:
            parser_args.pq_runtime_label = False
            assert parser_args.eval_all_documents or parser_args.knn_topk_by_step == 0
            assert parser_args.topk_sequence <= 0
            assert parser_args.pq_loss not in ('emdr2', 'adist')
        if parser_args.mode == 'train' and (parser_args.resume_from_checkpoint or parser_args.infer_ckpt):
            parser_args.pq_init_method = 'none'
        if parser_args.pq_init_method == 'avg':
            assert osp.isfile(
                parser_args.pq_cluster_path) and parser_args.pq_type in ('pq', 'rq')
        if parser_args.tie_nci_pq_centroid:
            assert parser_args.adaptor_efficient
            parser_args.pq_update_method = 'grad'
            print("Tie embedding requires pq not fixed.")
            parser_args.fixpq = False
        if parser_args.pq_update_method in ('kmeans', 'faiss'):
            parser_args.fixpq = True
            assert parser_args.validation_gen_val and not parser_args.no_validation
        else:
            parser_args.align_clustering = 0
        parser_args.kary = 2 ** parser_args.subvector_bits
        assert (parser_args.co_neg_from == 'file' and parser_args.pq_runtime_label) or (
            not parser_args.no_validation and parser_args.num_sanity_val_steps)
        parser_args.label_length_cutoff = 0
        parser_args.pq_loss = parser_args.pq_loss.lower()
        parser_args.pq_twin_loss = parser_args.pq_twin_loss.lower()
        assert parser_args.pq_loss in (
            'label', 'bce', 'kl', 'mse', 'cosine', 'dot', 'emdr2', 'adist', 'ce')
        assert parser_args.pq_twin_loss in ('co', 'quant')
        parser_args.centroid_update_loss = parser_args.centroid_update_loss.lower()
        assert parser_args.centroid_update_loss in ('none', 'reconstruct')
        assert not parser_args.reconstruct_for_embeddings or parser_args.centroid_update_loss == 'none'
        if parser_args.pq_loss in ('emdr2', 'adist'):
            parser_args.pq_negative = 'none'
            parser_args.use_gumbel_softmax = 0
            assert parser_args.pq_runtime_label == 1
        if parser_args.pq_loss != 'label':
            parser_args.topk_sequence = 0
        parser_args.pq_negative = parser_args.pq_negative.lower()
        parser_args.pq_negative_loss = parser_args.pq_negative_loss.lower()
        assert parser_args.pq_negative in ('none', 'sample', 'batch')
        assert parser_args.pq_negative_loss in ('cont', 'marg')
    if parser_args.kary:
        parser_args.output_vocab_size = parser_args.kary

    # args post process
    parser_args.tokenizer_name_or_path += parser_args.model_info
    parser_args.model_name_or_path += parser_args.model_info

    parser_args.n_gpu = eval(parser_args.n_gpu)
    if isinstance(parser_args.n_gpu, int):
        parser_args.n_gpu = list(range(parser_args.n_gpu))
    assert isinstance(parser_args.n_gpu, list)

    parser_args.gradient_accumulation_steps = max(
        int(8 / len(parser_args.n_gpu)), 1)

    if parser_args.dataset in ('marco', 'nq_dpr'):
        parser_args.max_input_length = parser_args.inf_max_input_length = 32
        print("change max input length to", 32)

    if parser_args.mode == 'train' and 'doc' in parser_args.query_type:
        assert parser_args.contrastive_variant == ''
        parser_args.max_input_length = parser_args.doc_length
        print("change max input length to", parser_args.doc_length)

    if not parser_args.document_encoder or parser_args.recall_level == 'coarse':
        parser_args.recall_num = [
            rn for rn in parser_args.recall_num if rn <= parser_args.num_return_sequences]

    if parser_args.model_info == 'base':
        parser_args.num_layers = 12
        parser_args.num_decoder_layers = 6
        parser_args.d_ff = 3072
        parser_args.d_model = 768
        parser_args.num_heads = 12
        parser_args.d_kv = 64
    elif parser_args.model_info == 'large':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_ff = 4096
        parser_args.d_model = 1024
        parser_args.num_heads = 16
        parser_args.d_kv = 64
    elif parser_args.model_info == 'small':
        parser_args.num_layers = 6
        parser_args.num_decoder_layers = 3
        parser_args.d_ff = 2048
        parser_args.d_model = 512
        parser_args.num_heads = 8
        parser_args.d_kv = 64

    if parser_args.document_encoder:
        if parser_args.codebook:
            parser_args.max_output_length = parser_args.subvector_num + 2
        if parser_args.recall_level in ('both', 'fine', 'finesampleloss') and not parser_args.codebook:
            assert parser_args.label_length_cutoff > 0
        else:
            if parser_args.label_length_cutoff <= 0:
                parser_args.label_length_cutoff = parser_args.max_output_length - 2
    if parser_args.label_length_cutoff > 0:
        parser_args.max_output_length = min(
            parser_args.max_output_length, parser_args.label_length_cutoff + 2)

    if parser_args.test1000:
        parser_args.n_val = 1000
        parser_args.n_train = 1000
        parser_args.n_test = 1000

    return parser_args


if __name__ == "__main__":
    args = parsers_parser()
    set_seed(args.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if args.document_encoder and args.encode_batch_size is None:
        args.encode_batch_size = 64
    if args.newid_dir is None:
        if args.dataset == 'marco':
            args.newid_dir = osp.join(args.data_dir, '../processed')
        else:
            args.newid_dir = args.data_dir
    if args.ckpt_dir is None:
        args.ckpt_dir = osp.join(args.data_dir, '../ckpts')
    if args.reserve_decoder and args.nci_ckpt is None:
        args.nci_ckpt = osp.join(args.ckpt_dir, 't5-ance/pytorch_model.bin')
    if not args.document_encoder or not args.codebook:
        if args.mapping_path is None:
            args.mapping_path = osp.join(args.newid_dir, 'old_newid.pkl')
        assert osp.isfile(args.mapping_path)
    if args.document_encoder or args.dataset == 'trivia':
        if args.document_path is None:
            temp_path = osp.join(args.data_dir, 'all_document')
            suffixes = ['_indices.pkl', '_tokens.bin', '_masks.bin']
            if args.drop_data_rate > 0:
                suffixes = [
                    '_indices.pkl', f'_tokens_drop{args.drop_data_rate}.bin', '_masks_drop{args.drop_data_rate}.bin']
            if [osp.exists(temp_path + suffix) for suffix in suffixes]:
                args.document_path = temp_path
            else:
                if args.dataset == 'marco':
                    if args.drop_data_rate > 0:
                        args.document_path = osp.join(
                            args.data_dir, f'corpus_drop{args.drop_data_rate}.tsv')
                    else:
                        args.document_path = osp.join(
                            args.data_dir, '../raw/corpus.tsv')
                elif args.dataset == 'nq_dpr':
                    args.document_path = osp.join(args.data_dir, 'corpus.tsv')
    if args.document_encoder:
        if args.codebook and args.pq_path is None:
            # if args.pq_type != 'rq':
            #     args.pq_path = osp.join(
            #         args.data_dir, f'{args.pq_type}{args.subvector_num}_{args.subvector_bits}_{args.document_encoder}.index')
            #     assert osp.isfile(args.pq_path)
            # else:
            assert not args.no_validation and args.num_sanity_val_steps
        elif not args.codebook:
            if args.cluster_path is None:
                args.cluster_path = osp.join(
                    args.newid_dir, f'doc_cluster_layer{args.label_length_cutoff}.pkl')
            assert osp.isfile(args.cluster_path)
    if args.logs_dir is None:
        args.logs_dir = osp.join(args.data_dir, 'logs')
    os.makedirs(args.logs_dir, exist_ok=True)
    # this is model pkl save dir
    args.output_dir = args.logs_dir
    args.newid_suffix = osp.split(args.newid_dir)[1]

    ###########################
    if args.time_str is None:
        args.time_str = time_str.replace('-', '')
    else:
        time_str = args.time_str[:8] + '-' + args.time_str[8:]
    # Note -- you can put important info into here, then it will appear to the name of saved ckpt
    if args.document_encoder:
        if args.codebook:
            important_info_list = [
                'mevi',
                args.dataset,
                args.query_type,
                args.model_info,
                f'k{args.kary}',
                f'dem{args.decode_embedding}',
                f'ada{args.adaptor_decode}_{args.adaptor_efficient}_{args.adaptor_layer_num}',
                f'rdrop{args.dropout_rate}_{args.Rdrop}_{args.Rdrop_only_decoder}',
                f'denc{args.document_encoder[:4]}',
                f'neg{args.co_neg_from}_{args.co_neg_num}',
                f'q{args.qtower}_{args.query_embed_accum[:4]}',
                f'rec{args.recall_level[:4]}_{args.num_return_sequences}',
                f'fix{int(args.fixnci)}_{int(args.fixdocenc)}_{int(args.fixpq)}',
                f'{args.pq_type}{args.subvector_num}_{args.subvector_bits}_tie{args.tie_nci_pq_centroid}',
                f'i{args.pq_init_method}_u{args.pq_update_method}',
                f'l{args.pq_loss}_{args.pq_negative}_{args.pq_dist_mode}_{args.centroid_update_loss}',
                f'rt{args.pq_runtime_label}_{args.pq_runtime_update_cluster}',
                f'sm{args.use_gumbel_softmax}_{args.pq_hard_softmax_topk}_{args.pq_softmax_tau}',
                f'ctopk{args.topk_sequence}',
                args.newid_suffix,
            ]
            if args.pq_loss in ('emdr2', 'adist'):
                important_info_list[-1] = f'aug{args.aug_topk_clus}{args.aug_find_topk_from}{args.aug_sample_topk}'
                important_info_list.append(args.newid_suffix)
            project_name = f'mevi_{args.dataset}_pq'
        else:
            if args.nci_twin_train_ratio:
                altratio = "_".join([x.split(":")[0]
                                    for x in args.nci_twin_train_ratio.split(",")])
            else:
                altratio = ''
            important_info_list = [
                'mevi',
                args.dataset,
                args.query_type,
                args.model_info,
                f'k{args.kary}',
                f'dem{args.decode_embedding}',
                f'ada{args.adaptor_decode}_{args.adaptor_efficient}_{args.adaptor_layer_num}',
                f'rdrop{args.dropout_rate}_{args.Rdrop}_{args.Rdrop_only_decoder}',
                f'cut{args.label_length_cutoff}',
                f'denc{args.document_encoder[:4]}',
                f'neg{args.co_neg_from}_{args.co_neg_num}',
                f'q{args.qtower}_{args.query_embed_accum[:4]}',
                f'rec{args.recall_level[:4]}_{args.num_return_sequences}',
                # f'alt{altratio}_{args.alt_train}_{args.alt_granularity}',
                f'fix{int(args.fixnci)}_{int(args.fixdocenc)}',
                args.newid_suffix,
            ]
            project_name = f'mevi_{args.dataset}'
    else:
        important_info_list = [
            'nci',
            args.dataset,
            args.query_type,
            args.model_info,
            f'k{args.kary}',
            f'dem{args.decode_embedding}',
            f'ada{args.adaptor_decode}_{args.adaptor_efficient}_{args.adaptor_layer_num}',
            f'rdrop{args.dropout_rate}_{args.Rdrop}_{args.Rdrop_only_decoder}',
            f'cut{args.label_length_cutoff}',
            args.newid_suffix,
        ]
        project_name = f'nci_{args.dataset}'

    args.query_info = '_'.join(important_info_list)
    if args.wandb_token != '':
        os.environ["WANDB_API_KEY"] = args.wandb_token
        name = '{}-{}'.format(time_str, args.query_info)
        if args.use_deepspeed:
            name = 'ds-' + name
        logger = WandbLogger(name=name, project=project_name, id=args.wandb_id)
    else:
        logger = TensorBoardLogger("logs/")
    ###########################

    args.tag_info = '{}_lre{}d{}'.format(args.query_info, str(float(args.learning_rate * 1e4)),
                                         str(float(args.decoder_learning_rate * 1e4)))
    if args.document_encoder:
        args.tag_info += 'de{}'.format(
            str(float(args.document_encoder_learning_rate * 1e4)))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        inference(args)
