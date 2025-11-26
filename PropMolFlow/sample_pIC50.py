from propmolflow.models.flowmol import FlowMol
import os
import torch
import argparse
from pathlib import Path
import math
from rdkit import Chem
# import selfies
from tqdm import tqdm
import pickle

p = argparse.ArgumentParser(description='Process geometry')
p.add_argument('--model-dir', type=Path, help='trained model path')
p.add_argument('--save-dir', type=Path, help='config file path')
p.add_argument('--method', type=str, help='property conditioning method')
p.add_argument('--sample-pIC50', type=float, help='pIC50 to sample from')
p.add_argument('--n-timesteps', type=int, default=100)
p.add_argument('--max-batch-size', type=int, default=1024)
p.add_argument('--n-mols', type=int, default=10000)

args = p.parse_args()

device = torch.device('cuda')
model = FlowMol.load_from_checkpoint(os.path.join(args.model_dir, 'checkpoints', 'last-v1.ckpt')).to(device)
model.eval()

n_batches = math.ceil(args.n_mols / args.max_batch_size)
molecules = []

print(f"Sampling {args.n_mols} molecules in {n_batches} batches for {args.method} conditioning method")
for _ in tqdm(range(n_batches)):
    # print(f"Batch {batch_idx+1}/{n_batches}")
    n_mols_needed = args.n_mols - len(molecules)
    batch_size = min(n_mols_needed, args.max_batch_size)

    batch_molecules = model.sample_random_sizes(
        batch_size, #batch_size,
        device=device,
        n_timesteps=args.n_timesteps,
        xt_traj=False, #args.xt_traj,
        ep_traj=False, #args.ep_traj,
        # stochasticity=None, #args.stochasticity,
        # high_confidence_threshold=args.hc_thresh,
        properties_for_sampling=args.sample_pIC50, #args.properties_for_sampling,
        # training_mode=args.training_mode,
        # property_name=args.property_name,
        # normalization_file_path=args.normalization_file_path,
        properties_handle_method=args.method, #args.properties_handle_method,
        # multilple_values_to_one_property=batch_property,
        # number_of_atoms=[150],
    )

    molecules.extend(batch_molecules)

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

import rdkit
# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

rdkit_quiet()

smiles_list = []

for mol in tqdm(molecules):
    rdkit_mol = mol.rdkit_mol
    mol_standardizer = rdMolStandardize.Normalize(rdkit_mol)
    smiles = Chem.MolToSmiles(mol_standardizer)
    try:
        # smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        smiles_list.append(smiles)
    except:
        pass

# smiles_list = []
# for mol in molecules:
#     rdkit_mol = mol.rdkit_mol
#     smiles_list.append(Chem.MolToSmiles(rdkit_mol))

# selfies_list = []
# for smiles in smiles_list:
#     try:
#         selfies_list.append(selfies.encoder(smiles=smiles))
#     except:
#         pass

# smiles_list = [selfies.decoder(selfies=sf) for sf in selfies_list]

os.makedirs(args.save_dir, exist_ok=True)

with open(os.path.join(args.save_dir, f'samples_method_{args.method}_sample_pIC50_{args.sample_pIC50}.pkl'), 'wb') as file:
    pickle.dump(smiles_list, file)