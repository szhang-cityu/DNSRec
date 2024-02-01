from thop import profile
import torch
import torch.nn as nn
import math
import torch.nn.functional as fn
import numpy as np


#--------------------Attention used for FLOPs Calculation----------------------
class GateNN_(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="ReLU",
                 dropout_rate=0.0,
                 batch_norm=False):
        super(GateNN_, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        gate_layers = [self.lin1]# Neural layer

        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        # gate_layers.append(get_activation(hidden_activation))
        gate_layers.append(nn.ReLU())# Relu
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))

        self.lin2 = nn.Linear(hidden_dim, output_dim)
        gate_layers.append(self.lin2)# Add another neural layer
        gate_layers.append(nn.Sigmoid())# Sigmoid
        self.gate = nn.Sequential(*gate_layers)

        #Initialize Parameters with Xavier and HE
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.kaiming_uniform_(self.lin2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, inputs):
        return self.gate(inputs) * 2

class SP_MultiHeadAttention_(nn.Module):
    """
    Scaled Product Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads=4,
        hidden_size=64,
        hidden_dropout_prob=0.2,
        attn_dropout_prob=0.2,
        layer_norm_eps=1e-12,
    ):
        super(SP_MultiHeadAttention_, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask = 0):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer,torch.matmul(normalized_key_layer,value_layer))/ self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward_(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
            self,
            n_heads = 4,
            hidden_size=[128, 64, 32],
            inner_size=[256, 128, 64],
            hidden_dropout_prob=0.2,
            attn_dropout_prob=0.2,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
    ):
        super(FeedForward_, self).__init__()
        self.dense_1 = nn.Linear(hidden_size[0], inner_size[0])
        self.gate_1 = GateNN_(hidden_size[0], output_dim=inner_size[0])
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size[0], hidden_size[0])
        self.gate_2 = GateNN_(hidden_size[0], output_dim=hidden_size[0])
        self.LayerNorm = nn.LayerNorm(hidden_size[0], eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.dense2_1 = nn.Linear(hidden_size[1], inner_size[1])
        self.gate2_1 = GateNN_(hidden_size[1], output_dim=inner_size[1])
        self.dense2_2 = nn.Linear(inner_size[1], hidden_size[1])
        self.gate2_2 = GateNN_(hidden_size[1], output_dim=hidden_size[1])
        self.LayerNorm2 = nn.LayerNorm(hidden_size[1], eps=layer_norm_eps)

        self.dense3_1 = nn.Linear(hidden_size[2], inner_size[2])
        self.gate3_1 = GateNN_(hidden_size[2], output_dim=inner_size[2])
        self.dense3_2 = nn.Linear(inner_size[2], hidden_size[2])
        self.gate3_2 = GateNN_(hidden_size[2], output_dim=hidden_size[2])
        self.LayerNorm3 = nn.LayerNorm(hidden_size[2], eps=layer_norm_eps)

        self.attention1 = SP_MultiHeadAttention_(
            n_heads, hidden_size[0], hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
        )
        self.attention2 = SP_MultiHeadAttention_(
            n_heads, hidden_size[1], hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
        )
        self.attention3 = SP_MultiHeadAttention_(
            n_heads, hidden_size[2], hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
        )

        # Initialize Parameters with Xavier and HE
        nn.init.kaiming_uniform_(self.dense_1.weight, mode='fan_in', nonlinearity='leaky_relu')

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor, k):
        if k == 1:
            hidden_states = self.attention1(input_tensor)
            hidden_states = self.dense_1(hidden_states)
            gate_1 = self.gate_1(input_tensor)
            hidden_states = self.intermediate_act_fn(hidden_states * gate_1)

            hidden_states = self.dense_2(hidden_states)
            gate_2 = self.gate_2(input_tensor)  # not used repeatedly

            hidden_states = self.dropout(hidden_states * gate_2)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        if k == 2:
            hidden_states = self.attention2(input_tensor)
            hidden_states = self.dense2_1(hidden_states)
            gate2_1 = self.gate2_1(input_tensor)
            hidden_states = self.intermediate_act_fn(hidden_states * gate2_1)

            hidden_states = self.dense2_2(hidden_states)
            gate2_2 = self.gate2_2(input_tensor)  # not used repeatedly

            hidden_states = self.dropout(hidden_states * gate2_2)
            hidden_states = self.LayerNorm2(hidden_states + input_tensor)
        if k == 3:
            hidden_states = self.attention3(input_tensor)
            hidden_states = self.dense3_1(hidden_states)
            gate3_1 = self.gate3_1(input_tensor)
            hidden_states = self.intermediate_act_fn(hidden_states * gate3_1)

            hidden_states = self.dense3_2(hidden_states)
            gate3_2 = self.gate3_2(input_tensor)  # not used repeatedly

            hidden_states = self.dropout(hidden_states * gate3_2)
            hidden_states = self.LayerNorm3(hidden_states + input_tensor)

        return hidden_states

def FLOPs_Calculation():
    device = torch.device("cpu")
    global FeedForward_
    global GateNN_
    global SP_MultiHeadAttention_
    net = FeedForward_().to(device)
    size = [128, 64, 32]
    FLOPs = []
    k=1
    for i in size:
        input_shape = (200, i)
        input_tensor = torch.randn(1, *input_shape).to(device)
        flops, _ = profile(net, inputs=(input_tensor, k))
        k += 1
        FLOPs.append(flops)
    print(FLOPs)
    # # ---------------------Drop the useless class--------------------
    # del GateNN_
    # del SP_MultiHeadAttention_
    # del FeedForward_
    return FLOPs