import torch
import numpy as np
from torch_geometric.data import Data

class DrugData(Data):
    def __init__(self, *args, **kwargs):
        super(DrugData, self).__init__(*args, **kwargs)

    @staticmethod
    def from_drug_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = DrugData(**kwargs)
        if protein_dict is None:
            if ligand_dict is not None:
                for key, item in ligand_dict.items():
                    instance[key] = item
                instance['orig_keys'] = list(ligand_dict.keys())
        else:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item
            if ligand_dict is not None:
                for key, item in ligand_dict.items():
                    instance['ligand_' + key] = item            
        return instance
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bond_index':
            return len(self['node_type'])
        elif key == 'edge_index':
            return len(self['node_type'])
        elif key == 'halfedge_index':
            return len(self['node_type'])
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output
