import torch
import esm
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from neomhci.data_utils import ACIDS_VOCAB


class MHCDataset(Dataset):
    def __init__(self, mode='emb', esm_model=None, **kwargs):
        assert mode in ('emb', 'esm')
        self.mode = mode
        if mode == 'esm':
            esm_module_func = getattr(esm.pretrained, esm_model)
            _, self.esm_alphabet = esm_module_func()

    def encode_peptide(self, peptide_seq, peptide_len, peptide_pad):
        emb_peptide_x = [ACIDS_VOCAB.get(x, ACIDS_VOCAB['-']) for x in peptide_seq[:peptide_len]]
        emb_peptide_out = [ACIDS_VOCAB['0']] * peptide_pad + emb_peptide_x + [ACIDS_VOCAB['0']] * (peptide_len - len(emb_peptide_x)) + [ACIDS_VOCAB['0']] * peptide_pad
        if self.mode == 'emb':
            return ([], emb_peptide_out)
        else:
            peptide_x = self.esm_alphabet.encode(peptide_seq[:peptide_len])
            esm_peptide_x = [self.esm_alphabet.cls_idx] + peptide_x + [self.esm_alphabet.padding_idx] * (peptide_len - len(peptide_seq[:peptide_len])) + [self.esm_alphabet.eos_idx]
            assert len(esm_peptide_x) == peptide_len + 2
            assert all(isinstance(x, int) for x in esm_peptide_x)
            return (esm_peptide_x, emb_peptide_out)

    def encode_mhc(self, mhc_seq):
        return [ACIDS_VOCAB.get(x, ACIDS_VOCAB['-']) for x in mhc_seq]

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ELMHCDataset(MHCDataset):
    def __init__(self, data_list, mhc_name_seq, peptide_len=15, peptide_pad=3, mhc_len=34, is_sa=False, **kwargs):
        super(ELMHCDataset, self).__init__(**kwargs)
        self.cell_line_name, self.peptide_esm_x, self.peptide_x, self.cell_line, self.targets = [], [], [], [], []
        self.mhc_name_idx = {x: i for i, x in enumerate(mhc_name_seq)}
        for cell_line_name, peptide_seq, mhc_name, score in tqdm(data_list, leave=False):
            self.cell_line_name.append(cell_line_name)
            pep_esm_x, pep_emb_x = self.encode_peptide(peptide_seq, peptide_len, peptide_pad)
            self.peptide_esm_x.append(pep_esm_x)
            self.peptide_x.append(pep_emb_x)
            self.cell_line.append(np.asarray([self.mhc_name_idx[x] for x in mhc_name]))
            self.targets.append(score)
        self.mhc_x = [self.encode_mhc(mhc_name_seq[n_]) for n_ in self.mhc_name_idx]
        self.peptide_esm_x, self.peptide_x, self.mhc_x = np.asarray(self.peptide_esm_x), np.asarray(self.peptide_x), np.asarray(self.mhc_x)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.is_sa, self.sa_item = is_sa, [i for i in range(len(self.cell_line)) if len(self.cell_line[i]) == 1]


    def __getitem__(self, item):
        if self.is_sa:
            item = self.sa_item[item]
        return (self.peptide_x[None, item].repeat(len(c_:=self.cell_line[item]), axis=0), 
                self.peptide_esm_x[None, item].repeat(len(c_), axis=0), 
                self.mhc_x[c_],
                len(c_), 
                self.targets[item])

    def __len__(self):
        return len(self.cell_line_name) if not self.is_sa else len(self.sa_item)

    @staticmethod
    def collate_fn(batch):
        peptide_x, peptide_esm_x, mhc_x, bags_size, targets = [torch.as_tensor(np.vstack(x)) for x in zip(*batch)]
        return (peptide_x, peptide_esm_x, mhc_x, bags_size.flatten()), targets.flatten()

