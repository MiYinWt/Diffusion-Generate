import os
import re
import sys
import shutil
import argparse
sys.path.append('.')

import torch
import numpy as np
from scipy import spatial
import torch.utils.tensorboard
from easydict import EasyDict
from rdkit import Chem

from torch_scatter import scatter_sum
from torch_geometric.data import Batch
from models.model import DiffGen
from models.bond_predictor import BondPredictor
from utils.sample import seperate_outputs
from torch_geometric.transforms import Compose
from utils.evaluation.atom_num_config import CONFIG
from utils.data import PDBProtein, parse_drug3d_mol, parse_lig_file
from datasets.dataset import get_dataset, ProteinLigandData
from datasets.data import torchify_dict
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *


def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))

def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])

def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)

def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    atom_num = np.random.choice(num_atom_list, p=prob_list)
    return atom_num

def pdb_to_pocket(pocket_pdb_path):
    pocket_dict = PDBProtein(pocket_pdb_path).to_dict_atom()
    ligand_dict = {
        "element": torch.empty([0, ], dtype=torch.long),
        "hybridization": torch.empty([0, ], dtype=torch.long),
        "pos": torch.empty([0, 3], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0, ], dtype=torch.long),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
        }
    # if frag_sdf_path != 'None':
    #     frag_dict = parse_lig_file(frag_sdf_path)
    #     data = ProteinLigandData.protein_ligand_dicts(
    #     protein_dict=to_torch_dict(pocket_dict),
    #     ligand_dict=to_torch_dict(ligand_dict),
    #     frag_dict=to_torch_dict(frag_dict)
    # )
    # else:
    #     data = ProteinLigandData.protein_ligand_dicts(
    #     protein_dict=to_torch_dict(pocket_dict),
    #     ligand_dict=to_torch_dict(ligand_dict)
    # )
    data = ProteinLigandData.from_drug_dicts(
    protein_dict=torchify_dict(pocket_dict),
    ligand_dict=torchify_dict(ligand_dict)
    )

    return data


if __name__ == '__main__':
    # Usage: python scripts/sample.py  --pocket ./fgfr1/7pqv_pocket.pdb --config ./configs/sample_configs/sample.yml --outdir ./outputs --device cuda:0
    parser = argparse.ArgumentParser()
    parser.add_argument('--pocket', type=str)
#   parser.add_argument('--ligand', type=str, default='None')
#   parser.add_argument('--frag', type=str, default='None')   
    parser.add_argument('--config', type=str, default='./configs/sample/sample.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))

    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    #log_dir = args.logdir
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # # Transform
    logger.info('Loading data placeholder...')
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    # if config.model.gen_mode == 'denovo':
    #     featurizer = FeatureComplex(ligand_atom_mode, sample=config.sample.sample)
    # else:
    #     featurizer = FeatureComplexWithFrag(ligand_atom_mode, sample=config.sample.sample)
    featurizer = FeatureComplex(ligand_atom_mode, sample=config.sample.sample)
    transform = Compose([
        featurizer,
    ])
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffgen':
        model = DiffGen(
                    config=train_config.model,
                    protein_node_types=featurizer.protein_feat_dim,
                    ligand_node_types=featurizer.atom_feat_dim,
                    num_edge_types=featurizer.bond_feat_dim,
                ).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()

    # # Bond predictor and guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(
            config=ckpt_bond['config']['model'],
            protein_node_types=featurizer.protein_feat_dim,
            ligand_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None

    # Load pocket or test set data
    if config.sample.mode == 'pocket':
        data = pdb_to_pocket(args.pocket)
        data = transform(data)
        data_list = []
        data_list.append(data)
    elif config.sample.mode == 'test':
        dataset, subsets = get_dataset(
            config = config.data,
            transform = transform,
        )
        data_list = subsets['test']
        logger.info(f'Test dataset: {len(data_list)}.')
    else:
        raise NotImplementedError('Sample mode should be pocket or test!')

    # prepare batch
    data_length = len(data_list)
    for i in tqdm(range(data_length), desc='Sample'):
        data = data_list[i]
        if config.sample.mode == 'pocket':
            name = args.pocket.split('/')[1].split('_')[0]
            logger.info(f'Protein Pocket: {args.pocket}')
            # logger.info(f'Reference Ligand: {config.model.ligand}')
            # logger.info(f'Optimization Fragment: {config.model.frag}')
        elif config.sample.mode == 'test':
            if config.data.dataset == 'pdbbind':
                name = data.protein_filename.split('/')[0]
            elif config.data.dataset == 'crossdocked':
                name = data.protein_filename.split('.')[0]
            logger.info(f'Protein Pocket: {data.protein_filename}.')

        pool = EasyDict({
            'failed': [],
            'finished': [],
        })
        # # generating molecules
        mol_list = []
        while len(pool.finished) < config.sample.num_mols:
            if len(pool.failed) > 3 * (config.sample.num_mols):
                logger.info('Too many failed molecules. Stop sampling.')
                break

            batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
            n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
            batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch=featurizer.follow_batch).to(args.device)

            if config.sample.sample_method == "priori":
                pocket_size = get_pocket_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)]
                ligand_batch = torch.repeat_interleave(torch.arange(n_graphs), torch.tensor(ligand_num_atoms)).to(args.device)
            elif config.sample.sample_method == "range":
                ligand_num_atoms = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
                ligand_num_atoms = ligand_num_atoms.astype('int64')
                ligand_batch = torch.repeat_interleave(torch.arange(n_graphs), torch.tensor(ligand_num_atoms)).to(args.device)
            elif config.sample.sample_method == "ref":
                ligand_batch = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(ligand_batch), ligand_batch, dim=0).tolist()
            else:
                raise ValueError
            
            logger.info(f'ligand_num_atoms: {ligand_num_atoms}')

            batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=args.device)
            batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
            
            # inference
            # if config.model.gen_mode == 'denovo':
            outputs = model.sample(
                n_graphs=n_graphs,
                protein_node=batch.protein_atom_feat.float(), 
                protein_pos=batch.protein_pos, 
                protein_batch=batch.protein_element_batch,
                ligand_batch=batch_node,
                halfedge_index=halfedge_index,
                halfedge_batch=batch_halfedge,
                bond_predictor=bond_predictor,
                guidance=guidance,
            )
            # elif config.model.gen_mode in ('frag_cond', 'frag_diff'):
            #     outputs = model.sample_frag(
            #         n_graphs=n_graphs,
            #         protein_node=batch.protein_atom_feat.float(), 
            #         protein_pos=batch.protein_pos, 
            #         protein_batch=batch.protein_element_batch,
            #         frag_node=batch.frag_atom_feat_full, 
            #         frag_pos=batch.frag_pos, 
            #         frag_batch=batch.frag_element_batch,
            #         frag_halfedge_type=batch.frag_halfedge_type,
            #         frag_halfedge_index=batch.frag_halfedge_index,
            #         frag_halfedge_batch=batch.frag_halfedge_type_batch,
            #         ligand_batch=batch_node,
            #         halfedge_index=halfedge_index,
            #         halfedge_batch=batch_halfedge,
            #         gui_strength=config.sample.gui_strength,
            #         bond_predictor=bond_predictor,
            #         guidance=guidance,
            #         gen_mode=config.model.gen_mode
            #     )

            outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
            
            # decode outputs to molecules
            batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
            try:
                output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
            except Exception as e:
                logger.info(f'Separate results error: {e}')
                continue
            gen_list = []
            for i_mol, output_mol in enumerate(output_list):
                mol_info = featurizer.decode_output(
                    pred_node=output_mol['pred'][0],
                    pred_pos=output_mol['pred'][1],
                    pred_halfedge=output_mol['pred'][2],
                    halfedge_index=output_mol['halfedge_index'],
                )  # note: traj is not used
                if add_edge == 'openbabel':
                    del mol_info['bond_index']
                    del mol_info['bond_type']
                    del mol_info['bond_prob']
                try:
                    rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                except MolReconsError:
                    pool.failed.append(mol_info)
                    logger.warning('Reconstruction error encountered.')
                    continue
                mol_info['rdmol'] = rdmol
                smiles = Chem.MolToSmiles(rdmol)
                mol_info['smiles'] = smiles
                contain_B = re.search(r'B(?![rR]\b)', smiles)
                if '.' in smiles:
                    logger.warning('Incomplete molecule: %s' % smiles)
                    pool.failed.append(mol_info)
                elif contain_B:
                    logger.warning('Element Boron in molecule: %s' % smiles)
                else:   # Pass checks!
                    try:
                        logger.info('Success: %s' % smiles)
                        chem_results = scoring_func.get_chem(rdmol)
                        if config.sample.mode == 'pocket':
                            vina_task = VinaDockingTask.from_generated_mol(rdmol, args.pocket)
                        elif config.sample.mode == 'test':
                            if config.data.dataset == 'pdbbind':
                                protein_fn = os.path.join(os.path.dirname(data.protein_filename), os.path.basename(data.protein_filename)[:4] + '_protein.pdb')
                            elif config.data.dataset == 'crossdocked':
                                protein_fn = os.path.join(os.path.dirname(data.protein_filename), os.path.basename(data.protein_filename)[:10] + '.pdb')
                            vina_task = VinaDockingTask.from_generated_mol(rdmol, os.path.join(config.data.protein_root, protein_fn))
                        vina_score = vina_task.run(mode='score_only', exhaustiveness=16)
                        mol_info['sa'] = chem_results['sa']
                        mol_info['vina_score'] = vina_score[0]['affinity']


                        p_save_traj = np.random.rand()  # save traj
                        if p_save_traj <  config.sample.save_traj_prob:
                            traj_info = [featurizer.decode_output(
                                pred_node=output_mol['traj'][0][t],
                                pred_pos=output_mol['traj'][1][t],
                                pred_halfedge=output_mol['traj'][2][t],
                                halfedge_index=output_mol['halfedge_index'],
                            ) for t in range(len(output_mol['traj'][0]))]
                            mol_traj = []
                            for t in range(len(traj_info)):
                                try:
                                    mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                                except MolReconsError:
                                    mol_traj.append(Chem.MolFromSmiles('O'))
                            mol_info['traj'] = mol_traj
                    except:
                        logger.warning('RDkit error encountered.')
                        continue    
                    gen_list.append(mol_info)
                    mol_list.append(mol_info)
                    # pool.finished.append(mol_info)
            pool.finished.extend(gen_list)
            print_pool_status(pool, logger)

        # # Save sdf mols
        sdf_dir = log_dir + '/'+ f'{name}_SDF'
        os.makedirs(sdf_dir, exist_ok=True)
        sorted_mol_list = sorted(mol_list, key=lambda mol: mol['mol_score'])
        with open(os.path.join(sdf_dir, 'log.txt'), 'a') as f:
            f.write('number, smiles, sa, vina:' + '\n')
            for i, data_finished in enumerate(sorted_mol_list):
                f.write(str(i) + ', ' + data_finished['smiles'] + ', ' + str(data_finished['sa']) + ', ' + 
                        str(data_finished['vina_score']) + '\n')
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, data_finished in enumerate(sorted_mol_list):
                smiles_f.write(data_finished['smiles'] + '\n')
                rdmol = data_finished['rdmol']
                try:
                    Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % (i)))
                except:
                    continue

                if 'traj' in data_finished:
                    writer = Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i)))
                    for m in data_finished['traj']:
                        try:
                            writer.write(m)
                        except:
                            writer.write(Chem.MolFromSmiles('O'))

        if config.data.dataset == 'pdbbind':
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))
        elif config.data.dataset == 'crossdocked':
            name = name.replace('/', '-')
            torch.save(pool, os.path.join(log_dir, f'samples_{name}.pt'))
    