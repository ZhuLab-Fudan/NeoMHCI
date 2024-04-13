import click
from pathlib import Path
from ruamel.yaml import YAML
from tqdm import trange
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from neomhci.data_utils import *
from neomhci.datasets import ELMHCDataset
from neomhci.models import Model
from neomhci.networks import NeoMHCI

def test(model, model_cnf, mhc_name_seq, test_data):
    data_loader = DataLoader(ELMHCDataset(test_data, mhc_name_seq, **model_cnf['EL_padding']),
                             collate_fn=ELMHCDataset.collate_fn,
                             batch_size=model_cnf['test']['batch_size'])
    return model.predict(data_loader)

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(('test', 'motif')), default=None)
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
@click.option('-g', '--gpu', type=int, default=0)
@click.option('-a', '--allele', default=None)
@click.option('--motif_len', 'motif_len', default=9, type=int)
def main(data_cnf, model_cnf, mode, start_id, num_models, allele, motif_len, gpu):
    device = torch.device(f'cuda:{gpu}')    
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']    
    model_path = Path(model_cnf['path'])/F'{model_name}.pt'
    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
    cell_line = get_cell_line(data_cnf['cell_line'])
    
    if mode == 'test':
        test_data = np.asarray(get_el_data(data_cnf['test'], data_cnf['cell_line']), dtype=object)
        test_file_name = Path(data_cnf['test']).stem
        scores_list, attn_list = [], []
        data_loader = DataLoader(ELMHCDataset(test_data, mhc_name_seq, **model_cnf['EL_padding']),
                                collate_fn=ELMHCDataset.collate_fn,
                                batch_size=model_cnf['test']['batch_size'])
        for model_id in trange(start_id, start_id + num_models):
            for cv_ in range(5):
                model = Model(network=NeoMHCI, device=device, output_key='MIL_attn',
                            model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'),
                            **model_cnf['model'])
                preds, attns = model.predict(data_loader)
                attn_list.append(attns)
                scores_list.append(preds)
                
        avg_scores_list = np.mean(scores_list, axis=0)
        attn_list = np.asarray(attn_list, dtype=object)
        avg_attn_list = np.mean(attn_list, axis=0)
        Path(F'./outputs').mkdir(exist_ok=True, parents=True)
        with open(Path(F'./outputs').joinpath(test_file_name).with_suffix('.csv'), 'w') as fp:
            cols = ['peptide','target','cell_line','alleles','attention_scores','neomhci-prediction']
            print(','.join(cols), file=fp)
            for i in range(len(test_data)):
                peptide = test_data[i][1]
                target = str(test_data[i][-1])
                cell_line = test_data[i][0]
                alleles = ';'.join(test_data[i][2])
                attns = ';'.join([str(x) for x in avg_attn_list[i]])
                pred = str(avg_scores_list[i])
                piece = [peptide, target, cell_line, alleles, attns, pred]
                print(','.join(piece), file=fp)
                
    elif mode == 'motif':
        assert allele in cell_line
        alleles = ','.join(cell_line[allele])
        peptide_list, data_list = get_seq2logo_data(data_cnf['rawdata'], allele,
                                motif_len=motif_len,
                                cell_line_file=data_cnf['cell_line'])
        data_loader = DataLoader(ELMHCDataset(data_list, mhc_name_seq, **model_cnf['EL_padding']),
                                collate_fn=ELMHCDataset.collate_fn,
                                batch_size=model_cnf['test']['batch_size'])
        
        scores_list, attn_list = [], []
        for model_id in trange(start_id, start_id + num_models):
            for cv_ in range(5):
                model = Model(network=NeoMHCI, device=device, output_key='MIL_attn',
                            model_path=model_path.with_name(F'{model_path.stem}-{model_id}-CV{cv_}'),
                            **model_cnf['model'])
                preds, attns = model.predict(data_loader)
                attn_list.append(attns)
                scores_list.append(preds)
                
        scores = np.mean(scores_list, axis=0).reshape(len(peptide_list), -1)
        attns = np.mean(attn_list, axis=0)
        s_ = scores.max(axis=1)
        Path(F'./motifs').mkdir(exist_ok=True, parents=True)
        motif_outpath = Path(F'./motifs/Motif_{model_path.stem}_{allele}_{motif_len}.txt')
        top_attns = []
        with open(motif_outpath, 'w') as fp:
            for k in (-s_).argsort()[:int(0.01 * len(s_))]:
                print(peptide_list[k], file=fp)
                top_attns.append(attns[k])    
        print(F'Attentions for {alleles}')
        print(np.mean(top_attns, axis=0))
        
if __name__ == '__main__':
    import shlex
    main()
              
        


