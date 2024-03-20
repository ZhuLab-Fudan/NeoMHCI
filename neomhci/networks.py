import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import neomhci
from neomhci.data_utils import ACIDS
from neomhci.modules import *


class Network(nn.Module):
    """

    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(Network, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, peptide_esm_x, mhc_x, **kwargs):
        peptide_out, mhc_out = self.peptide_emb(peptide_x), self.mhc_emb(mhc_x)
        peptide_x = peptide_x[:, self.peptide_pad: -self.peptide_pad]
        masks = peptide_x != self.padding_idx
        return peptide_out, masks, mhc_out

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)



class NeoMHCI(Network):
    def __init__(self, *, 
                 conv_num, 
                 conv_size, 
                 conv_off, 
                 linear_size,
                 attn_para,
                 selection_layer,
                 MIL_module,
                 MIL_module_config,
                 dropout=0.5, **kwargs):
        super(NeoMHCI, self).__init__(**kwargs)
        selection = getattr(neomhci.modules, selection_layer)
        self.glu = selection(**dict(attn_para)) 
        self.conv1 = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn1 = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv2 = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn2 = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 1)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])
        self.output_BA = nn.Conv1d(linear_size[-1], 1, 1)
        self.output_EL_main = nn.Conv1d(linear_size[-1], 1, 1)
        self.MIL_debag = getattr(neomhci.modules, MIL_module)(**MIL_module_config)  

        
        self.reset_parameters()
        
    def forward(self, peptide_x, peptide_esm_x, mhc_x, bags_size, **kwargs):
        aux_info = {}
        peptide_x, masks, mhc_x = super(NeoMHCI, self).forward(peptide_x, peptide_esm_x, mhc_x)
        peptide_x_gated, _ = self.glu(peptide_x)
        
        conv_out1 = torch.cat([conv_bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv1, self.conv_bn1, self.conv_off)], dim=1)        
        conv_out2 = torch.cat([conv_bn(F.relu(conv(peptide_x_gated[:, off: peptide_x_gated.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv2, self.conv_bn2, self.conv_off)], dim=1)
        conv_out_add = F.gelu(conv_out1 + conv_out2)
        conv_out_add = self.dropout(conv_out_add)
        
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            conv_out_add = linear_bn(F.relu(linear(conv_out_add)))  # B, l_s[0], L  
        masks = masks[:, None, -conv_out_add.shape[2]:]  # B, 1, L
        
        pool_out, _ = conv_out_add.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
        BA_pred = torch.sigmoid(self.output_BA(pool_out).flatten())
        
        MIL_output = self.MIL_debag(pool_out, bags_size)
        debag_out = MIL_output['debag_out'] if torch.sum(bags_size) > bags_size.size()[0] else pool_out
        if 'MIL_attn' in MIL_output:
            aux_info['MIL_attn'] = MIL_output['MIL_attn'].clone().detach().cpu().numpy()
        aux_info['bag_size'] = bags_size
        EL_pred = torch.sigmoid(self.output_EL_main(debag_out))
        return EL_pred.flatten(), aux_info

    def reset_parameters(self):
        super(NeoMHCI, self).reset_parameters()
        for conv, conv_bn in zip(self.conv1, self.conv_bn1):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for conv, conv_bn in zip(self.conv2, self.conv_bn2):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        truncated_normal_(self.output_BA.weight, std=0.1)
        nn.init.zeros_(self.output_BA.bias)
        truncated_normal_(self.output_EL_main.weight, std=0.1)
        nn.init.zeros_(self.output_EL_main.bias)
            
