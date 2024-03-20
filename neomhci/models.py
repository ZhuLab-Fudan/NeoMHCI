import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm



class Model(object):
    def __init__(self, *, network, device, model_path, output_key='MIL_attn', **kwargs):
        super(Model, self).__init__()
        self.device = device
        self.model_path = Path(model_path)
        self.model = self.network = network(**kwargs).to(self.device)
        self.output_key = output_key

    @torch.no_grad()
    def predict_step(self, inputs, **kwargs):
        self.model.eval()
        EL_pred, aux_info  = self.model(*(x.to(self.device) for x in inputs), **kwargs)
        return EL_pred.cpu(), aux_info

    def predict(self, data_loader: DataLoader, **kwargs):
        self.load_model()
        pred_scores, tensor_to_plot = [], []
        for data_x, _ in tqdm(data_loader, ncols=80, leave=False, dynamic_ncols=False):
            out = self.predict_step(data_x, **kwargs)
            pred_scores.append(out[0])
            idx = np.cumsum(out[1]['bag_size'].cpu().numpy())
            s_idx = [0] + list(idx[:-1])
            e_idx = idx
            for s, e in zip(s_idx, e_idx):
                tensor_to_plot.append(out[1][self.output_key][s:e])
        return np.hstack(pred_scores), tensor_to_plot
    
    def load_model(self):
        paras = torch.load(self.model_path)
        for layer_name in list(paras.keys()):
            if 'module' in layer_name:
                paras[layer_name[7:]] = paras.pop(layer_name)
        self.model.load_state_dict(paras)    

    
    

        