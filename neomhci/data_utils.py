import numpy as np

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'
ACIDS_VOCAB = {x: i for i, x in enumerate(ACIDS)}
RE_ACIDS = {i:x for i, x in enumerate(ACIDS)}


def get_mhc_name_seq(mhc_name_seq_file):
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq


def get_seq2logo_data(data_file, mhc_name, motif_len=9, cell_line_file='./data/allelelist'):
    cell_line = get_cell_line(cell_line_file)
    assert mhc_name in cell_line
    data_list, peptide_list = [], []
    with open(data_file) as fp:
        for k, line in enumerate(fp):
            peptide_seq = line.strip()
            pad_peptide_seq = peptide_seq[:motif_len]
            data_list += [(mhc_name, pad_peptide_seq, cell_line[mhc_name], 0.0)] 
            peptide_list.append(peptide_seq[:motif_len])
    return peptide_list, data_list


def get_cell_line(cell_line_file):
    cell_line = {}
    with open(cell_line_file) as fp:
        for line in fp:
            c_, mhc_name = line.split()
            cell_line[c_] = mhc_name.split(',')
    return cell_line

def get_el_data(data_file, cell_line_file):
    cell_line = get_cell_line(cell_line_file)
    data_list = []
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            peptide_seq, score, cell_line_name, *_ = line.split()
            if len(peptide_seq) >= 8 and len(peptide_seq) <= 15:
                data_list.append((cell_line_name, peptide_seq.upper(), cell_line[cell_line_name], float(score)))
    return data_list
