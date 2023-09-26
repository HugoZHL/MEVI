# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """

import copy
import math
import os
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from .configuration_t5 import T5Config
from .file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, Seq2SeqLMOutput, Seq2SeqModelOutput
from .modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from .utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    # logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Construct a layernorm module in the T5 style
        No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            input,
            mask=None,
            kv=None,
            position_bias=None,
            past_key_value=None,
            head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                    len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_qlen = qlen + past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value is not None:
            if kv is None:
                k_, v_ = past_key_value
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(
            q, k.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", q, k), compatible with onnx op>9

        if position_bias is None:
            if mask.shape[-1] == scores.shape[-1] + 1:
                # original decoder
                mask = mask[..., :-1]
            elif mask.shape[-1] == klen + 1:
                # pawa decoder
                klen += 1
                real_qlen = klen
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_qlen, klen), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -qlen:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state + (position_bias,)

        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    # remove has_relative_attention_bias: T5 has no relative attention bias; see below
    # https://github.com/huggingface/transformers/pull/8518
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            kv,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
                decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        # print(f"extended_attention_mask: {extended_attention_mask.shape}")

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            T5 is a model with relative position embeddings so you should be able to pad the inputs on both the right
            and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for detail.

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a
            `T5 Training <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for sequence to sequence training. T5 uses the :obj:`pad_token_id` as the starting token for
            :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at
            `T5 Training <./t5.html#training>`__. If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_input_ids` takes the value of :obj:`input_ids`.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`: `attentions`)
            :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds` have to be input
            (see :obj:`past_key_values`).
            This is useful if you want more control over how to convert :obj:`decoder_input_ids` indices into
            associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_inputs_embeds` takes the value of :obj:`inputs_embeds`.

        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            input_mask=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            only_encoder=False,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if only_encoder:
            return encoder_outputs

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class HierarchicT5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, depth=1):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.depth = depth

        self.stacks = nn.ModuleList(
            [T5Stack(config, embed_tokens) for _ in range(depth)]
        )

        for stack in self.stacks:
            stack.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        is_train = input_ids.shape[1] > 1 and past_key_values is None
        if is_train:
            assert input_ids.shape[1] <= self.depth
            max_depth = input_ids.shape[1]
            outputs = [
                self.stacks[i](
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                for i in range(max_depth)
            ]
            final_output = outputs[max_depth - 1]
            for i in range(max_depth - 1):
                final_output[0][:, i, :] = outputs[i][0][:, i, :]
        else:
            cur_depth = 0
            if past_key_values is not None:
                cur_depth = past_key_values[0][0].shape[2]
            # print(f"generate for {cur_depth}th token")
            final_output = self.stacks[cur_depth](
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return final_output


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        if 'Rdrop' in config.__dict__:
            self.Rdrop = config.Rdrop
        if 'Rdrop_only_decoder' in config.__dict__:
            self.Rdrop_only_decoder = config.Rdrop_only_decoder
        if 'Rdrop_loss' in config.__dict__:
            self.Rdrop_loss = config.Rdrop_loss
        if 'embedding_distillation' in config.__dict__:
            self.embedding_distillation = config.embedding_distillation
        if 'weight_distillation' in config.__dict__:
            self.weight_distillation = config.weight_distillation
        self.Rdrop_loss = "Contrast"

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        decode_embedding = getattr(config, "decode_embedding", None)
        hierarchic_decode = getattr(config, "hierarchic_decode", None)
        self.decode_vocab_size = getattr(config, "decode_vocab_size", None)
        tie_decode_embedding = getattr(config, "tie_decode_embedding", None)
        # decode_depth = getattr(config, "decode_depth", None)
        self.adaptor_decode = getattr(config, "adaptor_decode", None)
        self.adaptor_efficient = getattr(config, "adaptor_efficient", None)
        self.denoising = getattr(config, "denoising", None)
        self.multiple_decoder = getattr(config, "multiple_decoder", None)
        self.decoder_num = getattr(config, "decoder_num", None)
        self.max_output_length = getattr(config, "max_output_length", None)
        self.output_vocab_size = getattr(config, "output_vocab_size", None)
        self.use_codebook = getattr(config, 'use_codebook', 0)
        self.pq_loss = getattr(config, 'pq_loss', 'mse')
        self.pq_twin_loss = getattr(config, 'pq_twin_loss', 'co')
        self.reserve_decoder = getattr(config, 'reserve_decoder', 0)
        self.decoder_integration = getattr(config, 'decoder_integration', 'series')
        self.topk_minpooling = getattr(config, 'topk_minpooling', None)

        if decode_embedding:
            assert config.decode_vocab_size is not None
            if self.multiple_decoder:
                self.decode_embeddings_list = []
                for i in range(self.decoder_num):
                    self.decode_embeddings_list.append(nn.Embedding(config.decode_vocab_size, config.d_model))
            else:
                self.decode_embeddings = nn.Embedding(config.decode_vocab_size, config.d_model)
        else:
            self.decode_embeddings = self.shared

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        if self.multiple_decoder:
            decoder_config.num_layers = config.num_decoder_layers
            #decoder_config.num_layers = int(config.num_decoder_layers / config.decoder_num)
            print("num_decoder_layers", config.num_decoder_layers)
        else:
            decoder_config.num_layers = config.num_decoder_layers
        if self.multiple_decoder:
            self.decoder_list = []
            for i in range(self.decoder_num):
                self.decoder_list.append(T5Stack(decoder_config, self.decode_embeddings_list[i]))
        else:
            if decode_embedding and hierarchic_decode:
                assert config.decode_depth is not None
                self.decoder = HierarchicT5Stack(decoder_config, self.decode_embeddings, depth=config.decode_depth)
            else:
                self.decoder = T5Stack(decoder_config, self.decode_embeddings)
        if self.reserve_decoder:
            ori_decoder_config = copy.deepcopy(decoder_config)
            # fixed, align with t5-ance
            ori_decoder_config.num_decoder_layers = 12
            ori_decoder_config.num_layers = 12
            self.ori_decoder = T5Stack(ori_decoder_config, self.shared)

        print("adaptor_decode", self.adaptor_decode)
        print("adaptor_efficient", self.adaptor_efficient)
        if self.adaptor_decode and not self.adaptor_efficient:
            ## TODO: Need separate embedding, or reuse decoder embedding?
            print('10')
            self.adaptor_embeddings = nn.Embedding(config.decode_vocab_size, config.d_model)
            self.adaptor = T5Stack(decoder_config, self.adaptor_embeddings)  # [batch_size, seq_len, emb_dim]
            self.adaptor_linear = nn.Linear(config.d_model, config.d_model ** 2, bias=False)
        elif self.adaptor_efficient:
            print('11')
            if self.multiple_decoder:
                self.adaptor_embeddings_list = []
                self.adaptor_list = []
                self.adaptor_linear_list = []
                for i in range(self.decoder_num):
                    self.adaptor_embeddings_list.append(nn.Parameter(torch.rand(1, 1, config.d_model)))
                    decoder_layer = nn.TransformerDecoderLayer(d_model=config.d_model, nhead=8)
                    self.adaptor_list.append(nn.TransformerDecoder(decoder_layer, num_layers=config.adaptor_layer_num))
                    self.adaptor_linear_list.append(nn.Linear(config.d_model, config.d_model * config.decode_vocab_size, bias=False))
            else:
                self.adaptor_embeddings = nn.Parameter(torch.rand(1, 1, config.d_model))
                decoder_layer = nn.TransformerDecoderLayer(d_model=config.d_model, nhead=8)
                self.adaptor = nn.TransformerDecoder(decoder_layer, num_layers=config.adaptor_layer_num)
                self.adaptor_linear = nn.Linear(config.d_model, config.d_model * config.decode_vocab_size, bias=False)
        else:
            print('00')
            self.adaptor_embeddings = None
            self.adaptor = None
            self.adaptor_linear = None


        if decode_embedding:
            if self.multiple_decoder:
                self.lm_head_list = []
                for i in range(self.decoder_num):
                    self.lm_head_list.append(nn.Linear(config.d_model, config.decode_vocab_size, bias=False))
            else:
                self.lm_head = nn.Linear(config.d_model, config.decode_vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # [decode_vocab_size, emb_dim]
        if self.denoising:
           self.denoising_head = nn.Linear(config.d_model, 2, bias=False)
           self.denoising_prediction_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # [encoder_vocab_size, emb_dim]
           self._tie_or_clone_weights(self.denoising_prediction_head, self.shared)

        if tie_decode_embedding:
            if self.adaptor_decode or self.adaptor_efficient:
                if self.multiple_decoder:
                    for i in range(self.decoder_num):
                        self._tie_or_clone_weights(self.lm_head_list[i], self.decode_embeddings_list[i])
                else:
                    self._tie_or_clone_weights(self.lm_head, self.decode_embeddings)
            elif self.adaptor_decode:
                self._tie_or_clone_weights(self.lm_head, self.decode_embeddings)
                # self._tie_or_clone_weights(self.lm_head, self.adaptor_embeddings)
            else:
                self._tie_or_clone_weights(self.lm_head, self.decode_embeddings)

        if decode_embedding:
            #init decoder valid mask
            bz=1
            seq_length = config.max_output_length
            vocab_size = config.decode_vocab_size

            output_vocab_size = config.output_vocab_size
            print(bz, seq_length, vocab_size, output_vocab_size)
            valid_indices = torch.arange(output_vocab_size).view(1, -1)
            pos_indices = torch.arange(seq_length).view(-1, 1) * output_vocab_size

            # valid_indices = torch.arange(10).view(1, -1)
            # pos_indices = torch.arange(seq_length).view(-1, 1) * 10
            valid_indices = valid_indices + pos_indices + 2  #[seq_length, 10]
            ones_indices = torch.ones(seq_length, 1).to(valid_indices.device)
            valid_indices = torch.cat((valid_indices, ones_indices), dim=-1).long()
            valid_indices[-1,:] = torch.ones(1, output_vocab_size+1)
            # valid_indices[-1,:] = torch.ones(1, 11)
            valid_indices = valid_indices.unsqueeze(0).repeat([1, 1, 1]) #[bz, sl,10]
            zero_mask = torch.zeros(1, seq_length, vocab_size)
            mask = zero_mask - 1e9
            self.logit_mask = mask.scatter_(-1, valid_indices, zero_mask)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        if self.multiple_decoder:
            return self.lm_head_list
        else:
            return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            input_mask=None,
            logit_mask=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            only_encoder=False,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            lm_weights=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            query_embedding=None,
            prefix_embedding=None,
            prefix_mask=None,
            decoder_index=-1,
            loss_weight=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.topk_minpooling is None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
        else:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        # print(f"decoder_input_ids: {decoder_input_ids}")
        # if past_key_values is not None:
        #     print(f"past_key_values: {len(past_key_values)}")
        #     print(f"past_key_values[0]: {len(past_key_values[0])}")
        #     print(f"past_key_values[0][0].shape: {past_key_values[0][0].shape}")
        #     print(f"past_key_values[0][1].shape: {past_key_values[0][1].shape}")
        # else:
        #     print(f"past_key_values: None")

        # print(f"decoder_attention_mask: {decoder_attention_mask}")
        var_list = [decoder_input_ids, decoder_attention_mask, decoder_inputs_embeds,
                    past_key_values, hidden_states, attention_mask, head_mask, use_cache, output_attentions,
                    output_hidden_states]
        # #encoder constrastive
        # if self.training and self.Rdrop > 0:
        #     logits_1 = hidden_states[:,0,:]   #[bz, query_embed]
        #     logits_2 = hidden_states[:,0,:].transpose(0, 1)  #[query_embed, bz]
        #     query_logits = torch.matmul(logits_1, logits_2) #[bz, bz_logits]
        #     bz = query_logits.shape[0]
        #     query_mask = -1e9 * torch.eye(bz).to(query_logits.device)
        #     query_logits = query_logits + query_mask.unsqueeze(0)
        #     query_logits = F.softmax(query_logits.view(-1, bz), dim=-1) #[bz, bz_logits]
        #     contrast_labels = torch.cat([torch.arange(bz//2, bz), torch.arange(0, bz//2)], dim=-1)
        #     contrast_labels = contrast_labels.to(query_logits.device).long()
        #     query_dist_loss = loss_fct(query_logits, contrast_labels)
        if self.training and self.denoising and input_ids is not None:
            masked_input_ids = torch.where(input_mask==True, input_ids, torch.zeros_like(input_ids))
            predict_outputs = self.encoder(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            predict_input_logits = self.denoising_prediction_head(predict_outputs[0]) #[bz, seq_length, vocab_size]
            bz = predict_input_logits.shape[0]
            seq_length = predict_input_logits.shape[1]
            vocab_size = predict_input_logits.shape[2]
            predict_input_probs = F.softmax(predict_input_logits, dim=-1)
            predict_input_ids = torch.multinomial(predict_input_probs.view(-1, vocab_size), 1)
            predict_input_ids = predict_input_ids.view(bz, seq_length)
            generation_loss = loss_fct(predict_input_logits.transpose(1, 2), input_ids)
            predict_input_ids = torch.where(input_mask==True, input_ids, predict_input_ids)
            denoising_outputs = self.encoder(
                input_ids=predict_input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            denoising_logits = self.denoising_head(denoising_outputs[0])
            #if np.random.rand() < 0.2:
            #    encoder_outputs = denoising_outputs
            denoising_loss = loss_fct(denoising_logits.transpose(1, 2), input_mask.long())
        else:
            generation_loss = None
            denoising_loss = None

        if only_encoder:
            return encoder_outputs, input_mask, generation_loss, denoising_loss

        if self.reserve_decoder:
            ori_decoder_input_ids = torch.zeros(
                (attention_mask.shape[0], 1), dtype=torch.long).to(attention_mask.device)
            ori_decoder_outputs = self.ori_decoder(
                input_ids=ori_decoder_input_ids,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if self.decoder_integration == 'series':
                hidden_states = torch.cat((hidden_states, ori_decoder_outputs.last_hidden_state), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
                var_list[4] = hidden_states
                var_list[5] = attention_mask

        if self.training and self.Rdrop > 0 and self.Rdrop_only_decoder:
            for i in range(len(var_list)):
                if torch.is_tensor(var_list[i]):
                    var_list[i] = torch.cat([var_list[i], var_list[i].clone().detach()], dim=0)
            pass

        if self.multiple_decoder:
            #print(self.training, decoder_index)
            self_decoder = self.decoder_list[decoder_index].cuda()
            self_decode_embeddings = self.decode_embeddings_list[decoder_index].cuda()
            self_lm_head = self.lm_head_list[decoder_index].cuda()
            if self.adaptor_decode and self.adaptor_efficient:
                self_adaptor = self.adaptor_list[decoder_index].cuda()
                self_adaptor_embeddings = self.adaptor_embeddings_list[decoder_index].cuda()
                self_adaptor_linear = self.adaptor_linear_list[decoder_index].cuda()
        else:
            self_decoder = self.decoder
            self_decode_embeddings = self.decode_embeddings
            self_lm_head = self.lm_head
            self_adaptor = self.adaptor
            self_adaptor_embeddings = self.adaptor_embeddings
            self_adaptor_linear = self.adaptor_linear

        decoder_outputs = self_decoder(
            input_ids=var_list[0],
            attention_mask=var_list[1],
            inputs_embeds=var_list[2],
            past_key_values=var_list[3],
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=var_list[5],
            head_mask=var_list[6],
            use_cache=var_list[7],
            output_attentions=var_list[8],
            output_hidden_states=var_list[9],
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        #print("sequence_output", sequence_output.shape)

        def select_valid_embedding(sequence):
            bz = sequence.shape[0]
            seq_length = sequence.shape[1]
            vocab_size = sequence.shape[2]
            # valid_indices = torch.arange(10).view(1, -1).to(sequence.device)
            # pos_indices = torch.arange(seq_length).view(-1, 1).to(sequence.device) * 10

            valid_indices = torch.arange(self.output_vocab_size).view(1, -1).to(sequence.device)
            pos_indices = torch.arange(seq_length).view(-1, 1).to(sequence.device) * self.output_vocab_size

            valid_indices = valid_indices + pos_indices + 2  #[seq_length, 10]
            ones_indices = torch.ones(seq_length, 1).to(valid_indices.device)
            valid_indices = torch.cat((valid_indices, ones_indices), dim=-1).long()
            #print("valid_indices", valid_indices)
            # if seq_length == self.max_output_length - 1:
            #     valid_indices[-1,:] = torch.ones(1, 11)
            valid_indices = valid_indices.unsqueeze(0).repeat([bz, 1, 1]) #[bz, sl,10]
            #print("sequence", sequence.shape, sequence)
            #valid_sequence = torch.gather(sequence.view(-1, vocab_size), -1, valid_indices)
            #valid_sequence = valid_sequence.view(bz, seq_length, 10)
            mask = torch.zeros_like(sequence) - 1e9
            mask = mask.scatter_(-1, valid_indices, torch.zeros_like(valid_indices, dtype=torch.float32))
            #print("mask", mask.shape, mask)
            sequence = sequence + mask
            #print("result sequence", sequence.shape. sequence)
            return sequence

        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (
                    self.model_dim ** -0.5)  # shape(batch_size, sequence_length, config.d_model)
        # lm_logits = self.lm_head(sequence_output)

        if self.adaptor_decode and not self.adaptor_efficient:
            ## TODO: for inference, we have two option with past_key_value:
            ## 1. Do the same as decode, return past value as cache, then handle the cache during beam search
            ## 2. Ignore past_key_value, do full ataptor decode, then need to feed full historical ids intead of only the last token.
            adaptor_output = self_adaptor(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=torch.zeros_like(hidden_states),
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            adaptor_output = adaptor_output[0]  # shape(batch_size, sequence_length, config.d_model)
            adaptor_output = adaptor_output * (self.model_dim ** -0.5)  # TODO: What scale should be used
            adaptor_weight = self.adaptor_linear(adaptor_output).reshape(adaptor_output.shape[0], adaptor_output.shape[1], self.model_dim, self.model_dim)  # shape(batch_size, sequence_length, config.d_model ** 2)
            ## TODO: add activation?
            lm_head = torch.matmul(adaptor_weight, self.lm_head.weight.T)  # shape(batch_size, sequence_length, config.d_model, vocab_size)
            lm_logits = torch.matmul(sequence_output.unsqueeze(-2), lm_head)  # batch matmul, [batch, seq, 1, d_model] * [batch, seq, d_model, vocab_size] = [batch, seq, 1, vocab_size]
            lm_logits = lm_logits.squeeze(-2)
            """
            adaptor_output = adaptor_output[0]  # shape(batch_size, sequence_length, config.d_model)
            adaptor_output = adaptor_output * (self.model_dim ** -0.5)  # TODO: What scale should be used
            adaptor_weight = self.adaptor_linear(adaptor_output).reshape(adaptor_output.shape[0], adaptor_output.shape[1], self.model_dim, -1)  # shape(batch_size, sequence_length, config.d_model, vocab_size)
            ## TODO: add activation?
            # print("adaptor_weight shape:{}".format(adaptor_weight.shape))
            lm_head_weight = self.lm_head.weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, config.d_model, vocab_size]
            lm_head_weight = adaptor_weight * lm_head_weight  # [batch_size, sequence_length, config.d_model, vocab_size]
            lm_logits = torch.matmul(sequence_output.unsqueeze(-2),
                                     lm_head_weight)  # batch matmul, [batch, seq, 1, d_model] * [batch, seq, d_model, vocab_size] = [batch, seq, 1, vocab_size]
            lm_logits = lm_logits.squeeze(-2)
        """
        elif self.adaptor_efficient:
            lm_head_weight = self_lm_head.weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, config.d_model, vocab_size]
            # print("decoder_input_ids",decoder_input_ids)
            decoder_input_embedding = self_decode_embeddings(decoder_input_ids)  # [batch_size, seq_length, config.d_model]
            batch_size = decoder_input_ids.shape[0]
            seq_length = decoder_input_embedding.shape[1]

            def generate_square_subsequent_mask(sz):
                mask = (torch.triu(torch.ones(sz, sz)) == 1). \
                    transpose(0, 1)
                mask = mask.float(). \
                    masked_fill(mask == 0, float('-inf')). \
                    masked_fill(mask == 1, float(0.0))
                return mask

            mask = generate_square_subsequent_mask(seq_length).to(decoder_input_embedding.device, decoder_input_embedding.dtype)
            encode_embedding = self_adaptor_embeddings + torch.zeros(batch_size, 1, 1).to(decoder_input_embedding.device, decoder_input_embedding.dtype)
            decoder_input_embedding = self_adaptor(decoder_input_embedding.transpose(0, 1),
                                                   encode_embedding.transpose(0, 1), tgt_mask=mask).transpose(0, 1)
            if os.environ.get("FP16_OPT", "false").lower() == "true":
                decoder_input_embedding = decoder_input_embedding.half()
                linear_weight = list(self_adaptor_linear.parameters())[0].half().transpose(0, 1)
                linear_result = torch.matmul(decoder_input_embedding, linear_weight)
                adaptor_weight = linear_result.reshape(decoder_input_embedding.shape[0],
                                                       decoder_input_embedding.shape[1],
                                                       self.model_dim, -1)
                lm_head_weight = adaptor_weight + lm_head_weight.half()
                lm_logits = torch.matmul(sequence_output.half().unsqueeze(-2), lm_head_weight)
                lm_logits = lm_logits.squeeze(-2).float()
            else:
                adaptor_weight = self_adaptor_linear(decoder_input_embedding).reshape(decoder_input_embedding.shape[0],
                                                                                      decoder_input_embedding.shape[1],
                                                                                      self.model_dim, -1)
                lm_head_weight = adaptor_weight + lm_head_weight
                lm_logits = torch.matmul(sequence_output.unsqueeze(-2), lm_head_weight)
                lm_logits = lm_logits.squeeze(-2)
        else:
            lm_logits = self_lm_head(sequence_output)  # shape(batch_size, sequence_length, config.vocab_size)

        if self.training:
            lm_logits = lm_logits + self.logit_mask.to(lm_logits.device)
        else:
            lm_logits = select_valid_embedding(lm_logits)

        loss = None
        if labels is not None:
            use_codebook = self.use_codebook and self.pq_loss != 'label'
            if use_codebook or (self.use_codebook and self.pq_twin_loss == 'quant'):
                if self.output_vocab_size + 2 == self.decode_vocab_size:
                    nci_logits = lm_logits[:, :-2, 2:]
                else:
                    seqlen = labels.size(1) - 2
                    index = torch.arange(seqlen * self.output_vocab_size, device=lm_logits.device) + 2
                    index = index.view(1, seqlen, self.output_vocab_size).expand(lm_logits.size(0), -1, -1)
                    nci_logits = torch.gather(lm_logits[:, :-2], dim=2, index=index)

            if self.Rdrop > 0 and self.training:
                bz = lm_logits.shape[0]
                sl = lm_logits.shape[1]
                lm_logits_half = lm_logits[:bz//2]
                lm_logits_another = lm_logits[bz//2:]
                #lm_logits_half = sequence_output[:bz//2,-1,:]
                #lm_logits_another = sequence_output[bz//2:,-1,:]
                #label_half = labels[:bz//2]
                if loss_weight is not None:
                    assert not use_codebook
                    loss_fct_2 = CrossEntropyLoss(ignore_index=-100, reduction="none")
                    orig_loss = loss_fct_2(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                    while loss_weight.dim() < 2:
                        loss_weight = loss_weight.unsqueeze(-1)
                    orig_loss = orig_loss.view(lm_logits.shape[0], -1) * loss_weight
                    mask = (labels.view(-1) != -100)
                    orig_loss = torch.sum(orig_loss) / torch.sum(mask)
                else:
                    if use_codebook:
                        orig_loss = 0
                    else:
                        orig_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                        if self.topk_minpooling is not None:
                            orig_loss = orig_loss.view(-1, self.topk_minpooling, labels.shape[-1])
                            orig_loss = torch.mean(orig_loss, [0, 2])
                            orig_loss = torch.min(orig_loss, 0)[0]
                            orig_loss = torch.mean(loss)
                if self.Rdrop_loss == 'KL':
                    p_loss = F.kl_div(F.log_softmax(lm_logits_half.view(-1, lm_logits.size(-1)), dim=-1), F.softmax(lm_logits_another.view(-1, lm_logits.size(-1)), dim=-1), reduction='none')
                    q_loss = F.kl_div(F.log_softmax(lm_logits_another.view(-1, lm_logits.size(-1)), dim=-1), F.softmax(lm_logits_half.view(-1, lm_logits.size(-1)), dim=-1), reduction='none')
                    p_loss = p_loss.sum()
                    q_loss = q_loss.sum()
                    dist_loss = (p_loss + q_loss) / 2 / sl
                elif self.Rdrop_loss == "Contrast":
                    neg_logits_1 = sequence_output.transpose(0, 1)  #[sl, bz, vocab_size]
                    neg_logits_2 = neg_logits_1.transpose(1, 2)  #[sl, vocab_size, bz]
                    neg_logits = torch.bmm(neg_logits_1, neg_logits_2) #[sl, bz, bz_logits]
                    neg_mask = -1e9 * torch.eye(bz).to(neg_logits.device)
                    neg_logits = neg_logits + neg_mask.unsqueeze(0)
                    neg_logits = F.softmax(neg_logits.view(-1, bz), dim=-1) #[sl*bz, bz_logits]
                    contrast_labels = torch.cat([torch.arange(bz//2, bz), torch.arange(0, bz//2)], dim=-1)
                    contrast_labels = contrast_labels.to(neg_logits.device).long()
                    contrast_labels = contrast_labels.unsqueeze(0).repeat(sl, 1).view(-1)
                    dist_loss = loss_fct(neg_logits, contrast_labels)
                else:
                    prob_half = F.softmax(lm_logits_half.view(-1, lm_logits.size(-1)), dim=-1)
                    prob_another = F.softmax(lm_logits_another.view(-1, lm_logits.size(-1)), dim=-1)
                    dist_loss = torch.mean((prob_half-prob_another)**2)
                loss = orig_loss + self.Rdrop*dist_loss
                #print("orig_loss", orig_loss)
            else:
                #if use_codebook:
                if False:
                    loss = 0
                else:
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                    if self.topk_minpooling is not None:
                        loss = loss.view(-1, self.topk_minpooling, labels.shape[-1])
                        loss = torch.mean(loss, [0, 2])
                        loss = torch.min(loss, 0)[0]
                        loss = torch.mean(loss)
            #print(labels.shape, lm_logits.shape)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if self.embedding_distillation > 0:
                # cut_idx = hidden_states.shape[0]//2 if self.Rdrop > 0 else hidden_states.shape[0]
                # embedding_distillation_loss = torch.mean((hidden_states[:cut_idx, 0, :] - query_embedding) ** 2)
                end_token_emb = torch.cat(
                    [hidden_states[idx, i, :].unsqueeze(-2) for idx, i in enumerate(torch.where(input_ids == 1)[1])],
                    dim=0)
                embedding_distillation_loss = torch.mean((end_token_emb - query_embedding) ** 2)
                loss = loss + self.embedding_distillation * embedding_distillation_loss
            if self.weight_distillation > 0:
                # TODO: check lm_head_weight shape; get prefix string; according to the prefix string get current_prefix_embed
                # prefix = labels[:,:-1]
                # weight_distill_loss = torch.mean((lm_head_weight - current_prefix_embed)**2)
                weight_mask = decoder_attention_mask.unsqueeze(-1) * prefix_mask
                _temp_weight = torch.transpose(lm_head_weight, -1, -2)
                _temp_weight = _temp_weight[:,:,2:,:]
                _temp_weight = _temp_weight.reshape(_temp_weight.shape[0], _temp_weight.shape[1],
                                                    _temp_weight.shape[1], -1, _temp_weight.shape[-1])
                _temp_weight = torch.cat([_temp_weight[:, i, i, :, :].unsqueeze(dim=1) for i in range(_temp_weight.shape[1])], dim=1)
                _weight_distill_loss = torch.mean((_temp_weight - prefix_embedding)**2, dim=-1)
                weight_distill_loss = _weight_distill_loss[weight_mask > 0].sum() / weight_mask.sum()
                loss = loss + self.weight_distillation * weight_distill_loss
        else:
            loss = None

        if loss is not None and self.denoising:
            loss += generation_loss + denoising_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        if self.training and self.Rdrop > 0 and labels is not None:
            return_result = Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                orig_loss=orig_loss,
                dist_loss=self.Rdrop*dist_loss,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            return_result = Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        return_result.labels = labels
        if self.training and self.use_codebook and (self.pq_loss != 'label' or self.pq_twin_loss == 'quant'):
            return_result.nci_logits = nci_logits
        if self.training and self.embedding_distillation > 0 and labels is not None:
            return_result.emb_distill_loss = self.embedding_distillation * embedding_distillation_loss

        if self.training and self.weight_distillation > 0 and labels is not None:
            return_result.weight_distill_loss = self.weight_distillation * weight_distill_loss
        # print('!')
        if self.adaptor_decode and self.adaptor_efficient:
            return_result.lm_head_weight = lm_head_weight
        return_result.encoder_outputs = encoder_outputs
        return_result.lm_logits = lm_logits
        return_result.decoder_last_hidden_state=decoder_outputs.last_hidden_state
        if self.reserve_decoder:
            return_result.ori_decoder_last_hidden_state = ori_decoder_outputs.last_hidden_state
        else:
            return_result.ori_decoder_last_hidden_state = None
        return return_result

    def get_cluster_embedding(
        self,
        decoder_input_ids,
        decoder_attention_mask,
        accum,
        attenpool_weight,
        decoder_index=-1
    ):
        assert self.adaptor_decode
        if self.multiple_decoder:
            #print(self.training, decoder_index)
            self_decoder = self.decoder_list[decoder_index].cuda()
            self_decode_embeddings = self.decode_embeddings_list[decoder_index].cuda()
            self_lm_head = self.lm_head_list[decoder_index].cuda()
            if self.adaptor_decode and self.adaptor_efficient:
                self_adaptor = self.adaptor_list[decoder_index].cuda()
                self_adaptor_embeddings = self.adaptor_embeddings_list[decoder_index].cuda()
                self_adaptor_linear = self.adaptor_linear_list[decoder_index].cuda()
        else:
            self_decoder = self.decoder
            self_decode_embeddings = self.decode_embeddings
            self_lm_head = self.lm_head
            self_adaptor = self.adaptor
            self_adaptor_embeddings = self.adaptor_embeddings
            self_adaptor_linear = self.adaptor_linear

        if not self.adaptor_efficient:
            raise NotImplementedError
            adaptor_output = self_adaptor(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=torch.zeros_like(hidden_states),
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            adaptor_output = adaptor_output[0]  # shape(batch_size, sequence_length, config.d_model)
            adaptor_output = adaptor_output * (self.model_dim ** -0.5)  # TODO: What scale should be used
            adaptor_weight = self.adaptor_linear(adaptor_output).reshape(adaptor_output.shape[0], adaptor_output.shape[1], self.model_dim, self.model_dim)  # shape(batch_size, sequence_length, config.d_model ** 2)
            ## TODO: add activation?
            lm_head = torch.matmul(adaptor_weight, self.lm_head.weight.T)  # shape(batch_size, sequence_length, config.d_model, vocab_size)
            lm_logits = torch.matmul(sequence_output.unsqueeze(-2), lm_head)  # batch matmul, [batch, seq, 1, d_model] * [batch, seq, d_model, vocab_size] = [batch, seq, 1, vocab_size]
            lm_logits = lm_logits.squeeze(-2)
            return lm_head
        else:
            lm_head_weight = self_lm_head.weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, config.d_model, vocab_size]
            # print("decoder_input_ids",decoder_input_ids)
            decoder_input_embedding = self_decode_embeddings(decoder_input_ids)  # [batch_size, seq_length, config.d_model]
            batch_size = decoder_input_ids.shape[0]
            seq_length = decoder_input_embedding.shape[1]

            def generate_square_subsequent_mask(sz):
                mask = (torch.triu(torch.ones(sz, sz)) == 1). \
                    transpose(0, 1)
                mask = mask.float(). \
                    masked_fill(mask == 0, float('-inf')). \
                    masked_fill(mask == 1, float(0.0))
                return mask

            mask = generate_square_subsequent_mask(seq_length).to(decoder_input_embedding.device, decoder_input_embedding.dtype)
            encode_embedding = self_adaptor_embeddings + torch.zeros(batch_size, 1, 1).to(decoder_input_embedding.device, decoder_input_embedding.dtype)
            decoder_input_embedding = self_adaptor(decoder_input_embedding.transpose(0, 1),
                                                   encode_embedding.transpose(0, 1), tgt_mask=mask).transpose(0, 1)
            adaptor_weight = self_adaptor_linear(decoder_input_embedding).reshape(decoder_input_embedding.shape[0],
                                                                                  decoder_input_embedding.shape[1],
                                                                                  self.model_dim, -1)
            lm_head_weight = adaptor_weight + lm_head_weight
            cluster_embeddings = []
            for weight, ids in zip(lm_head_weight, decoder_input_ids):
                # (seqlen, dim, vocabsize), (seqlen,)
                embeds = torch.stack([w[:,i] for w, i in zip(weight, ids)])
                # (seqlen, dim)
                if accum == 'maxpool':
                    clus_emb = torch.max(embeds, 0)[0]
                elif accum == 'avgpool':
                    clus_emb = torch.mean(embeds, 0)
                elif accum == 'attenpool':
                    atten_w = torch.nn.functional.softmax(
                        attenpool_weight(embeds), dim=0)
                    clus_emb = torch.sum(embeds * atten_w, dim=0)
                # (dim,)
                cluster_embeddings.append(clus_emb)
            return torch.stack(cluster_embeddings)


    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
