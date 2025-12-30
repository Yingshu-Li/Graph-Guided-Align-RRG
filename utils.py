import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.rouge.rouge import Rouge
from evalcap.meteor.meteor import Meteor
import numpy as np

factory = {'Bleu_4': Bleu(), 'CIDEr': Cider()}


class GCNLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super(GCNLayer, self).__init__()
        self.condense = nn.Linear(in_size, state_size, bias=False)
        self.condense_norm = nn.LayerNorm(state_size)
        self.fw_trans = nn.Linear(in_size, state_size, bias=False)
        self.fw_norm = nn.LayerNorm(state_size)
        self.bw_trans = nn.Linear(in_size, state_size, bias=False)
        self.bw_norm = nn.LayerNorm(state_size)
        self.update = nn.Linear(state_size, in_size, bias=False)
        self.update_norm = nn.LayerNorm(in_size)
        self.gelu = nn.GELU() 
        # v2:
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, states, fw_A, bw_A):
        # states: batch_size x feat size x nodes
        condensed = self.gelu(self.condense_norm(self.condense(states)))
        fw_msg = self.fw_trans(states).permute(0, 2, 1).bmm(fw_A)
        fw_msg = self.gelu(self.fw_norm(fw_msg.permute(0, 2, 1)))
        bw_msg = self.bw_trans(states).permute(0, 2, 1).bmm(bw_A)
        bw_msg = self.gelu(self.bw_norm(bw_msg.permute(0, 2, 1)))
        updated = self.update_norm(self.update(condensed+fw_msg+bw_msg))
        updated = self.gelu(self.dropout(updated) + states)
        return updated

class GCN(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.layer1 = GCNLayer(in_size, state_size)
        self.layer2 = GCNLayer(in_size, state_size)
        # self.layer3 = GCNLayer(in_size, state_size)

    def forward(self, x, img_feats, fw_A, bw_A):
        
        states = x.float()
        states = self.layer1(states, fw_A, bw_A)
        states = self.layer2(states, fw_A, bw_A)
        # states = self.layer3(states, fw_A, bw_A)
        return states.type_as(x)
    
class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

rms_norm = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        

class TransformationBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(TransformationBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = LayerNormFp32(feature_dim)
        self.norm2 = LayerNormFp32(feature_dim)
        self.norm3 = LayerNormFp32(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, query, key, value):
        # Self-attention
        query_attended, _ = self.self_attention(query, query, query)
        query = self.norm1(query + query_attended)
        
        # Cross-attention
        query_cross_attended, _ = self.cross_attention(query, key, value)
        query = self.norm2(query + query_cross_attended)
        
        # Feed-forward network
        query = self.norm3(query + self.ffn(query))
        
        return query


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, ffn_dim, hidden_dim):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        y = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        y = self.w2(y)
        return y


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(ffn_dim, hidden_dim) \
                                      for _ in range(self.num_experts)])

    def load_balance(self, router_logits):
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts)
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss


    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        # calculate top-k experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, \
                                               self.top_k, dim=-1)
        # normlize routing weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        ## One Hot encoding
        expert_mask = torch.nn.functional.one_hot(selected_experts, \
                                          num_classes=self.num_experts).permute(2, 1, 0)
        final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), \
                    dtype=hidden_states.dtype, device=hidden_states.device)
        
        # collect output
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state)  \
                            * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        load_balance_loss = self.load_balance(router_logits)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), load_balance_loss
    


def kullback_leibler_divergence(P, Q, eps=1e-7):
    P = torch.clamp(P, min=0.0+eps, max=1.0-eps)
    Q = torch.clamp(Q, min=0.0+eps, max=1.0-eps)
    dist = (P * torch.log(P / Q) + (1 - P) *
            torch.log((1 - P) / (1 - Q))).mean()
    return dist


class JensenShannonDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_P, logits_Q):
        P = self.sigmoid(logits_P)
        Q = self.sigmoid(logits_Q)
        # P = P.float()  # 
        # Q = Q.float()
        M = 0.5 * (P + Q)

        loss = 0.5 * kullback_leibler_divergence(
            P, M, self.eps) + 0.5 * kullback_leibler_divergence(Q, M, self.eps)
        return loss
        # return loss.to(dtype=torch.bfloat16)