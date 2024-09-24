import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class IConv(nn.Module):
    """

    """
    def __init__(self, out_channels, kernel_size, mhc_len=34, stride=1, **kwargs):
        super(IConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, **kwargs):
        bs = peptide_x.shape[0]
        kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight))
        outputs = F.conv1d(peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1]),
                           kernel.contiguous().view(-1, *kernel.shape[-2:]), stride=self.stride, groups=bs)
        return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


    
class MIL_DotGateAttn(nn.Module):
    def __init__(self, *, in_channel, hidden_size, **kwargs):
        super(MIL_DotGateAttn, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channel, hidden_size))
        self.gate_weight = nn.Parameter(torch.Tensor(in_channel, hidden_size))
        self.vec = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameter()
        
    def forward(self, inter_pre, bags_size):
        inter_pre = inter_pre.squeeze()
        inter_ = torch.tanh(torch.einsum('bm,ml->bl', inter_pre, self.weight))
        inter_gate = torch.sigmoid(torch.einsum('bm,ml->bl', inter_pre, self.weight))
        inter_ = torch.mul(inter_, inter_gate)
        attn_ = torch.einsum('bl,l->b', inter_, self.vec)
        offsets = np.cumsum([0] + list(bags_size.cpu().detach().numpy())
        softmax_attn = torch.cat([F.softmax(attn_[i:j], dim=0) for i,j in zip(offsets[:-1], offsets[1:])], dim=0)
        inter_pre_bag = torch.stack([torch.einsum('b,bl->l', softmax_attn[i:j], inter_pre[i:j,:]) for i,j in zip(offsets[:-1], offsets[1:])])
        return {'debag_out': inter_pre_bag.unsqueeze(-1), 'MIL_attn': softmax_attn}
    
    def reset_parameter(self):
        truncated_normal_(self.weight, std=0.02)
        truncated_normal_(self.gate_weight, std=0.02)
        truncated_normal_(self.vec, std=0.02)
        

class SLinear_T(nn.Module):
    r"""For a input peptide embedding matrix, do linear calculation on each acid with independent weights.
    
    Using tensordot with torch.arange
    """
    def __init__(self, 
                 pep_len, 
                 in_channel, 
                 out_channel, 
                 bias : bool = True
                 ):
        super(SLinear_T, self).__init__()
        self.pep_len = pep_len
        self.bias_bool = bias
        self.weight = nn.Parameter(torch.Tensor(pep_len, in_channel, out_channel))
        if bias:    
            self.bias = nn.Parameter(torch.Tensor(pep_len, out_channel))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, peptide_x):
        peptide_x = torch.tensordot(peptide_x, self.weight, dims=((-1,), (1,)))
        peptide_x = peptide_x[:,torch.arange(self.pep_len).unsqueeze(-1), torch.arange(self.pep_len).unsqueeze(-1),:]
        if self.bias_bool:
            return peptide_x.squeeze(2) + self.bias
        else:
            return peptide_x.squeeze(2)
        
    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        if self.bias_bool:
            nn.init.zeros_(self.bias)

class SGLU(nn.Module):
    r"""Special gated linear unit, not for one acid, for a whole peptide instead.
    """
    #Gated Linear Unit
    def __init__(self, 
                 pep_len, 
                 in_channel, 
                 hidden_layer_size,
                 dropout_rate = None,
                 temperature = False,
                 SLinear = SLinear_T
                ):
        super(SGLU, self).__init__()
        self.pep_len = pep_len
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = SLinear(pep_len, in_channel, hidden_layer_size)
        self.gated_layer = SLinear(pep_len, in_channel, hidden_layer_size)

        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature
        if self.temperature:
            self.register_parameter('alpha', nn.Parameter(torch.Tensor(1)))
            self.register_parameter('beta', nn.Parameter(torch.Tensor(1)))
            self.reset_parameters()
        
    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        activation = self.activation_layer(x)
        if not self.temperature:
            gated = self.sigmoid(self.gated_layer(x))
        else:
            gated = self.sigmoid(self.gated_layer(x)/self.alpha + self.beta)
        
        return torch.mul(activation, gated), gated
    
    def reset_parameters(self):
        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.beta, 0.0)



