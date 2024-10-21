import numpy as np
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pickle
import warnings
import bionetgen
import contextlib
import os
import pandas as pd

import numpy as np

import pickle

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse

from multiprocessing import Pool

import argparse

# arguments and argument checking
parser = argparse.ArgumentParser()

parser.add_argument('--parp-model-path', type=str, required=True, help='Filepath for the PARP model')
parser.add_argument('--pathway-model', type=str, required=True, choices=['viable', 'modified', 'impractical'])
parser.add_argument('--sample-path', type=str, required=True, help='Directory for the generated molecule samples')
parser.add_argument('--bngl-model-path', required=True, help='Filepath for the pathway model')
parser.add_argument('--save-filedir', type=str, required=True, help='Directory to save the results')
parser.add_argument('--ncpu', type=int, default=16)

warnings.filterwarnings("ignore")

# Parse args and run main code
args = parser.parse_args()

# xc_path = "/home/alif/JTVAE/updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"

# model_path = xc_path
smiles_col = 'smiles'
response_col = 'pXC50'
dont_standardize = True
is_featurized = False

pred_params = {'featurizer': 'computed_descriptors', 
               'result_dir': None,
               'id_col': 'compound_id', 
               'smiles_col': smiles_col,
               'response_cols': response_col}

pred_params = parse.wrapper(pred_params)
          
def get_pipeline(pred_params,
                 model_path,
                 reload_dir=None,
                 verbose=False):
    pipe = mp.create_prediction_pipeline_from_file(pred_params, 
                                               reload_dir=None, 
                                               model_path=model_path, 
                                               verbose=False)
   
    return pipe 
     
pipe = get_pipeline(pred_params=pred_params,
                    model_path=args.parp_model_path)

def pXC50(smiles):
    with contextlib.redirect_stdout(None):
        pred_df = pipe.predict_on_smiles([smiles], AD_method='z_score')
        pIC50 = pred_df['pred'][0]
    return pIC50

def calc_therapeutic_score(smiles, kr2:float):
    with contextlib.redirect_stdout(None):
        pIC50 = pXC50_dict[smiles]
        model = bionetgen.bngmodel(args.bngl_model_path, 'model')
        model.parameters.IC50 = 10 ** (6 - pIC50)
        model.kr2 = kr2
        result = bionetgen.run(model)
        therapeutic_score = result['Apopt Repair Toy Model 011823 v2.0'][-1][6]
        del model
    return therapeutic_score

def process(data):
    vocab = set()
    for smiles in data:
        with contextlib.redirect_stdout(None):
            s = calc_therapeutic_score(smiles, kr2=kr2)
        vocab.add( (smiles, s) )
    return vocab

if args.pathway_model == "viable":
    kr2 = 2.25e-1
elif args.pathway_model == "modified":
    kr2 = 2.25e-3
elif args.pathway_model == "impractical":
    kr2 = 2.25e-6
else:
    raise NotImplementedError("Not available in the list of pathway models")

os.makedirs(args.save_filedir, exist_ok=True)

all_smiles = []

# k = inf

res_file_therap_opt = os.path.join(args.sample_path, "results-uniform-weight.npz")

results_therap_opt = np.load(res_file_therap_opt, allow_pickle = True)

smiles = results_therap_opt['sample_points'].reshape((-1, ))

smiles = set(smiles)
if None in smiles:
    smiles.remove(None)

all_smiles.extend(list(smiles))

# k = {3, 4, 5, 6}

for k in [3, 4, 5, 6]:
    res_file_therap_opt = os.path.join(args.sample_path, "results-{}-k-{}.npz".format(args.pathway_model, k))

    results_therap_opt = np.load(res_file_therap_opt, allow_pickle = True)

    smiles = results_therap_opt['sample_points'].reshape((-1, ))

    smiles = set(smiles)
    if None in smiles:
        smiles.remove(None)

    all_smiles.extend(list(smiles))

all_smiles = list(set(all_smiles))

import timeit

start = timeit.default_timer()

from tqdm import tqdm
pXC50_dict = {}

print("Measuring pIC50 of the generated molecules \n")

with tqdm(total=len(all_smiles)) as pbar:
    for smiles in set(all_smiles):
        pXC50_dict[smiles] = pXC50(smiles)
        pbar.update(1)

with open(os.path.join(args.save_filedir, "gen_pXC50_{}_pathway_model.pkl".format(args.pathway_model)), "wb") as f:
    pickle.dump(pXC50_dict, f)

print("Measuring therapeutic scores of the generated molecules \n")

with open(os.path.join(args.save_filedir, "gen_pXC50_{}_pathway_model.pkl".format(args.pathway_model)), "rb") as f:
    pXC50_dict = pickle.load(f)

ncpu = args.ncpu

batch_size = len(all_smiles) // ncpu + 1
batches = [all_smiles[i : i + batch_size] for i in range(0, len(all_smiles), batch_size)]

pool = Pool(ncpu)
vocab_list = pool.map(process, batches)
vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
vocab = list(set(vocab))

print(vocab)

therap_dict = {}

for smiles, score in sorted(vocab):
    therap_dict[smiles] = score
    with open(os.path.join(args.save_filedir, "gen_therapeutic_score_{}_pathway_model.txt".format(args.pathway_model)), 'a') as f:
    	f.writelines(smiles+': '+str(score)+'\n')
    	
with open(os.path.join(args.save_filedir, "gen_therapeutic_score_{}_pathway_model.pkl".format(args.pathway_model)), "wb") as f:
    pickle.dump(therap_dict, f)
    
stop = timeit.default_timer()

print('Time: ', stop - start) 
