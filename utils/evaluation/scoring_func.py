import os
import pickle
import torch
import itertools
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolAlign, rdfiltercatalog
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.QED import qed

from .sascorer import compute_sa_score


def is_pains(mol):
    """
    Checks if a molecule contains any PAINS (Pan-Assay Interference Compounds) substructures.
    This implementation uses the modern RDKit API.
    """
    # Initialize the filter catalog parameters
    params = rdfiltercatalog.FilterCatalogParams()
    
    # Add the PAINS filter catalog. Note: In modern RDKit, it's just PAINS, not PAINS_A/B/C.
    params.AddCatalog(rdfiltercatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    
    # Create the catalog
    catalog = rdfiltercatalog.FilterCatalog(params)
    
    mol_copy = deepcopy(mol)
    Chem.SanitizeMol(mol_copy)
    
    # Check for a match
    entry = catalog.GetFirstMatch(mol_copy)
    
    # Return True if a match is found, False otherwise
    return entry is not None

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    all_rmsd = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
            all_rmsd.append(rmsd)
        rmsd_list = np.array(rmsd_list)
        return np.min(rmsd_list)
    except:
        return []


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    tpsa = Chem.rdMolDescriptors.CalcTPSA(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'tpsa': tpsa,
        'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies

def get_rdkit_optimize_rmsd(ori_mol, addHs=False, enable_torsion=False):
    """
    Calculate the alignment of generated mol and rdkit optimized mol
    Return their rmsd
    """
    mol = deepcopy(ori_mol)
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if mp is None:
        return (None, )

    # turn off angle-related terms
    mp.SetMMFFOopTerm(enable_torsion)
    mp.SetMMFFAngleTerm(True)
    mp.SetMMFFTorsionTerm(enable_torsion)

    # optimize unrelated to angles
    mp.SetMMFFStretchBendTerm(True)
    mp.SetMMFFBondTerm(True)
    mp.SetMMFFVdWTerm(True)
    mp.SetMMFFEleTerm(True)
    
    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        energy_before_ff = ff.CalcEnergy()
        ff.Minimize()
        energy_after_ff = ff.CalcEnergy()
        energy_change = energy_before_ff - energy_after_ff
        Chem.SanitizeMol(ori_mol)
        Chem.SanitizeMol(mol)
        rmsd = rdMolAlign.GetBestRMS(ori_mol, mol)
    except:
        return []
    return [energy_change, rmsd, mol]

def get_novelty_and_uniqueness(gen_mols, ref_mols):
    train_smiles = []
    for mol in ref_mols:
        train_smile = Chem.MolToSmiles(mol)
        train_smiles.append(train_smile)
    n_in_train = 0
    smiles_list = []
    for mol in tqdm(gen_mols, desc='Calculate novelty and uniqueness'):
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
        if smiles in train_smiles:
            n_in_train += 1
    novelty = 1 - n_in_train / len(gen_mols)
    unique = len(np.unique(smiles_list)) / len(gen_mols)
    return {'novelty': novelty, 'uniqueness': unique}

def get_sim_with_ref(gen_mols, ref_mols, parallel=False):
    train_finger = []
    for mol in ref_mols:
        fg = Chem.RDKFingerprint(mol)
        train_finger.append(fg)
    mol_finger = [Chem.RDKFingerprint(mol) for mol in gen_mols]
    finger_pair = list(itertools.product(mol_finger, train_finger))
    if not parallel:
        similarity_list = []
        for fg1, fg2 in tqdm(finger_pair, desc='Calculate similarity with ref'):
            similarity_list.append(get_similarity((fg1, fg2)))
    else:
        with Pool(102) as pool:
            similarity_list = list(tqdm(pool.imap(get_similarity, finger_pair), 
                                        total=len(mol_finger)*len(train_finger)))
            
    # calculate the max similarity of each mol with train data
    if len(similarity_list) > 0:
        similarity_max = np.reshape(similarity_list, (len(gen_mols), -1)).max(axis=1)
    else:
        similarity_max = np.array([])
    return np.mean(similarity_max)

def get_diversity(gen_mols, parallel=False):
    fgs = [Chem.RDKFingerprint(mol) for mol in gen_mols]
    all_fg_pairs = list(itertools.combinations(fgs, 2))
    if not parallel:
        similarity_list = []
        for fg1, fg2 in tqdm(all_fg_pairs, desc='Calculate diversity'):
            similarity_list.append(TanimotoSimilarity(fg1, fg2))
    else:
        with Pool(102) as pool:
            similarity_list = pool.imap_unordered(TanimotoSimilarity, all_fg_pairs)
    return 1 - np.mean(similarity_list)

def get_similarity(fg_pair):
    return TanimotoSimilarity(fg_pair[0], fg_pair[1])
