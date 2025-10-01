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

env = lmdb.open('./datas/crossdocked2020/pocket10_processed_final.lmdb', readonly=True, subdir=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        data = pickle.loads(value)
        if 'protein_pos' not in data:
            print(f"Key {key} missing protein_pos")