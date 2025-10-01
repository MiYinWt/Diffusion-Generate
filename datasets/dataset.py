import sys
import os
sys.path.append('.')
import pickle
import lmdb
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from torch.utils.data import Dataset,Subset

from datasets.data import ProteinLigandData, torchify_dict
from utils.data import PDBProtein, parse_lig_file, parse_drug3d_mol

def get_dataset(config, *args, **kwargs):
    if config.name == "protein_ligand":
        dataset = ProteinLigandDataset(config.path, *args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset name: {config.name}")
    
    if "split" in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset, None

class ProteinLigandDataset(Dataset):
    def __init__(self, path, transform=None, version="final"):
        super().__init__()
        self.path = path.rstrip('/')
        self.index_path = os.path.join(self.path, "index.pkl")
        self.processed_path = os.path.join(
            os.path.dirname(self.path), os.path.basename(self.path) + f"_processed_{version}.lmdb"
        )
        self.transform = transform
        self.database = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f"{self.processed_path} does not exist, start to process data!")
            self._process()

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, _, ligand_fn, _) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_drug3d_mol(os.path.join(data_prefix, ligand_fn))
                    
                    data = ProteinLigandData.from_drug_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()

    def _build_db(self):
        assert self.database is None
        self.database = lmdb.open(
            self.processed_path, map_size=10*(1024*1024*1024), create=False, 
            subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
        )
        with self.database.begin() as db:
            self.keys = list(db.cursor().iternext(values=False))

    def __len__(self):
        if self.database is None:
            self._build_db()
        return len(self.keys)

    def get_ori_data(self, idx):
        if self.database is None:
            self._build_db()
        key = self.keys[idx]
        data = pickle.loads(self.database.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

