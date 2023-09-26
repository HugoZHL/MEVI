
from dataclasses import dataclass
from typing import Dict, Optional
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (AutoModel, BatchEncoding,
                          PreTrainedModel, AutoConfig)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_bert import BertModel
from transformers.modeling_t5 import T5Model
from transformers.modeling_ernie import ErnieModel


@dataclass
class DocEncOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    pos_p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DocumentEncoder(nn.Module):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        normalize: bool = False,
        negatives_x_sample: bool = True,
        negatives_x_device: bool = False,
    ):
        # support T5 and Bert, including T5-ANCE, ANCE-Tele, coCondenser
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        if isinstance(lm_p, T5Model):
            self.mtype = 't5'
        elif isinstance(lm_p, (BertModel, ErnieModel)):
            self.mtype = 'bert'
        else:
            raise NotImplementedError

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.normalize = normalize
        self.negatives_x_sample = negatives_x_sample
        self.negatives_x_device = negatives_x_device

        if negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
        self,
        q_reps: Dict[str, Tensor] = None,
        p_reps: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        if p_reps is None:
            p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            assert False

        # for training
        if self.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        if self.negatives_x_sample:
            scores = self.compute_similarity(q_reps, p_reps)
            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))
        else:
            p_reps = p_reps.view(q_reps.shape[0], -1, p_reps.shape[-1])
            scores = self.compute_similarity(q_reps.unsqueeze(1), p_reps, True)
            target = torch.zeros(
                scores.shape[0], device=scores.device, dtype=torch.long)

        loss = self.loss_fn(scores, target)

        if self.training and self.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DocEncOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            pos_p_reps=p_reps[target],
        )

    def encode(self, items, model):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        if self.mtype == 't5':
            decoder_input_ids = torch.zeros(
                (items.input_ids.shape[0], 1), dtype=torch.long).to(items.input_ids.device)
            items_out = model(
                **items, decoder_input_ids=decoder_input_ids, return_dict=True)
        elif self.mtype == 'bert':
            items_out = model(
                **items, return_dict=True)
        hidden = items_out.last_hidden_state
        reps = hidden[:, 0, :]
        if self.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps

    def encode_query(self, qry):
        return self.encode(qry, self.lm_q)[1]

    def encode_passage(self, psg):
        return self.encode(psg, self.lm_p)[1]

    def compute_similarity(self, q_reps, p_reps, bmm=False):
        if bmm:
            return torch.sum(q_reps * p_reps, dim=-1)
        else:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @classmethod
    def build(
        cls,
        model_name_or_path,
        config_path=None,
        normalize=False,
        tied=True,
        negatives_x_sample=True,
        dropout=None,
        **hf_kwargs,
    ):
        if config_path is not None:
            # only for ar2
            config = AutoConfig.from_pretrained(
                config_path,
            )
            if dropout is not None:
                config.attention_probs_dropout_prob = dropout
                config.hidden_dropout_prob = dropout
            lm_p = AutoModel.from_pretrained(
                config_path, config=config)
            lm_q = deepcopy(lm_p)
            from torch.serialization import default_restore_location
            params = torch.load(
                model_name_or_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            params = params['model_dict']
            p_prefix = 'ctx_model.'
            q_prefix = 'question_model.'
            p_st = lm_p.state_dict()
            q_st = lm_q.state_dict()
            for k, v in params.items():
                if k.startswith(p_prefix):
                    kk = k[len(p_prefix):]
                    if kk in p_st:
                        p_st[kk].copy_(v)
                    else:
                        print(f'{kk} not in state dict !')
                elif k.startswith(q_prefix):
                    kk = k[len(q_prefix):]
                    if kk in q_st:
                        q_st[kk].copy_(v)
                    else:
                        print(f'{kk} not in state dict !')
                else:
                    assert False
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=1,
                cache_dir=None,
            )
            lm_p = AutoModel.from_pretrained(
                model_name_or_path, config=config, **hf_kwargs)
            if tied:
                lm_q = lm_p
            else:
                lm_q = deepcopy(lm_p)

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            normalize=normalize,
            negatives_x_sample=negatives_x_sample,
        )
        return model

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def generate(
        self,
        q_reps: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        p_reps: Dict[str, Tensor] = None,
        bmm: bool = False,
    ):
        if p_reps is None:
            p_reps = self.encode_passage(passage)
        if q_reps is None:
            scores = None
        else:
            scores = self.compute_similarity(q_reps, p_reps, bmm)
        return DocEncOutput(scores=scores, q_reps=q_reps, p_reps=p_reps)
