import sys
sys.path.append('.')
from datasets.dataset import get_dataset
from torch_geometric.transforms import Compose
import utils.transforms as transforms
from utils.misc import *


configs = './configs/train_configs/train_bondpred.yml'
config = load_config(configs)
featurizer = transforms.FeatureComplex(
        config.data.transform.ligand_atom_mode, 
        sample=config.data.transform.sample
    )
transform = Compose([
        featurizer,
    ])

dataset, subsets = get_dataset(config = config.data,transform = transform)

print('Dataset length', len(dataset))
for i in range(len(dataset)):
    try:
        d = dataset[i]
        print(i, 'ok')
    except Exception as e:
        print('Exception at', i, e)
        break