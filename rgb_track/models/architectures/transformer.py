import copy
import math
import torch
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.
    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + \
            (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask=None):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        # and perform attention probs computation in float32
        attention_scores = torch.matmul(query, key.transpose(-1, -2)).float()
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.float()
        attention_probs = torch.softmax(attention_scores, dim=-1).to(key.dtype)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size, )
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        output_states = self.layer_norm(queries + output_states)

        return output_states


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.
    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = torch.relu(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        output_states = self.layer_norm(hidden_states + output_states)
        return output_states


class TransformerEncoderBlock(nn.Module):
    """
    Building block of Transformer encoder.
    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(self, hidden_size, inner_size=None,
                 attn_score_dropout=0.1, attn_layer_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        num_attention_heads = hidden_size // 32
        
        inner_size = inner_size or hidden_size * 4
        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads,
            attn_score_dropout, attn_layer_dropout)
        self.second_sub_layer = PositionWiseFF(
            hidden_size, inner_size, ffn_dropout)

    def forward(self, hidden_states):
        self_attn_output = self.first_sub_layer(
           hidden_states, hidden_states, hidden_states, None)
        output_states = self.second_sub_layer(self_attn_output)
        return output_states


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, **kwargs):
        super().__init__()

        layer = TransformerEncoderBlock(hidden_size, **kwargs)
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, hidden_states):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
        """

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states