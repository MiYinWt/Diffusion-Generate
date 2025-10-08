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

data_prefix = './datas/crossdocked2020/pocket10'
ligand_fn = '3HAO_CUPMC_1_172_0/4i3p_A_rec.pdb'

ligand_dict = parse_drug3d_mol(os.path.join(data_prefix, ligand_fn))

