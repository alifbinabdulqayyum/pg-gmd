""" calculate penalized logP for all smiles in a file """

import argparse
import pickle as pkl
from tqdm.auto import tqdm
from rdkit import Chem
from weighted_retraining.chem.chem_utils import (
    pXC50,
    get_pipeline,
    rdkit_quiet,
    standardize_smiles,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    nargs="+",
    help="list of file of SMILES to calculate pIC50 for",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    help="pkl file to write properties to",
    required=True,
)
parser.add_argument(
    "-pmp",
    "--parp_model_path",
    type=str,
    help="Filepath of the PARP Model",
    required=True,
)

if __name__ == "__main__":

    rdkit_quiet()

    args = parser.parse_args()

    pipe = get_pipeline(model_path=args.parp_model_path)

    # Read input file
    print("Reading input file...")
    input_smiles = []
    for fname in args.input_file:
        with open(fname) as f:
            input_smiles += f.readlines()
    input_smiles = [s.strip() for s in input_smiles]

    print("Calculating properties...")
    prop_dict = dict()
    for smiles in tqdm(input_smiles, desc="calc pXC50", dynamic_ncols=True):
        c_smiles = standardize_smiles(smiles)
        logP = pXC50(pipe=pipe, smiles=c_smiles)
        prop_dict[smiles] = logP
        prop_dict[c_smiles] = logP

    # Output to a file
    print("Writing output file...")
    with open(args.output_file, "wb") as f:
        pkl.dump(prop_dict, f)
