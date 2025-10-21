import os
import pickle
import argparse
from tqdm import tqdm

INDEX_FILENAME = 'index/INDEX_general_PL.2020R1.lst'

if __name__ == "__main__":
    # Useage: python ./datas/data_preparation/clean_PDBbind.py --source ./datas/PDBbind_v2020/source --dest ./datas/PDBbind_v2020/pkl
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./datas/PDBbind_v2020")
    parser.add_argument('--dest', type=str, required=True, default='./datas/PDBbind_v2020/')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=False)
    index_path = os.path.join(args.source, INDEX_FILENAME)

    index = []
    with open(index_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the first line
        for line in tqdm(lines):
            if line.startswith('#'):
                continue
            else:
                pdbid, res, year, pka = line.split('//')[0].strip().split()
                try:
                    protein_fn = pdbid + "/" + pdbid + "_protein.pdb"
                    ligand_fn = pdbid + "/" + pdbid + "_ligand.sdf"

                    index.append((protein_fn, ligand_fn, pdbid))
                except Exception as e:
                    print(pdbid, str(e))
                    continue

    index_path = os.path.join(args.dest, 'index.pkl')            
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    
    print(f'Done processing {len(index)} protein-ligand pairs in total.\n Processed files in {args.dest}.')