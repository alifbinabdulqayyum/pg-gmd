""" calculate penalized logP for all smiles in a file """
import contextlib
import argparse
import pickle as pkl
from tqdm.auto import tqdm
from rdkit import Chem
from weighted_retraining.chem.chem_utils import (
    # therapeutic_score,
    get_estimated_therapeutic_model,
    get_pipeline,
    pXC50,
    rdkit_quiet,
    standardize_smiles, 
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    nargs="+",
    help="list of file of SMILES to calculate Therapeutic Score for",
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
    "-p",
    "--pathway_model",
    type=str,
    help="The type of Pathway Model to be used to calculate the Therapeutic Score",
    choices=["viable", "modified", "impractical"],
    required=True,
)

parser.add_argument(
    "-pmp",
    "--parp_model_path",
    type=str,
    help="Filepath of the PARP Model",
    required=True,
)

parser.add_argument(
    "-bmp",
    "--bngl_model_path",
    type=str,
    help="Filepath of the Pathway Model",
    required=True,
)

if __name__ == "__main__":

    rdkit_quiet()

    args = parser.parse_args()

    # Set kr2 value according to the specified pathway model
    if args.pathway_model == "viable":
        kr2 = 2.25e-1
    elif args.pathway_model == "modified":
        kr2 = 2.25e-3
    elif args.pathway_model == "impractical":
        kr2 = 2.25e-6
    else:
        raise NotImplementedError("Not available in the list of pathway models")
    
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

    print(args.bngl_model_path)

    therapeutic_score = get_estimated_therapeutic_model(bng_path=args.bngl_model_path, kr2=kr2)
    for smiles in tqdm(input_smiles, desc="calc Therapeutic Score", dynamic_ncols=True):
        c_smiles = standardize_smiles(smiles)
        with contextlib.redirect_stdout(None):
            # score = therapeutic_score(c_smiles)
            score = therapeutic_score(pXC50(pipe=pipe, smiles=c_smiles))
        prop_dict[smiles] = score
        prop_dict[c_smiles] = score

    # Output to a file
    print("Writing output file...")
    with open(args.output_file, "wb") as f:
        pkl.dump(prop_dict, f)
