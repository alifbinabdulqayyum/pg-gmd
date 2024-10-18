""" Contains many chem utils codes """
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen
import networkx as nx
from rdkit.Chem import rdmolops

# My imports
from weighted_retraining.chem.chem_utils.SA_Score import sascorer

# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

# =============================== #

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

import bionetgen
import contextlib
import os
import pandas as pd

import numpy as np

import pickle

# from atomsci.ddm.pipeline import predict_from_model as pfm
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse

xc_path = "/home/alif/JTVAE/updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"

model_path = xc_path
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
          
def get_pipeline(pred_params,model_path,reload_dir=None,verbose=False):
    # this is necessary to restrict the pXC50 model to use GPU, this is weird, but it WORKS
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    pipe = mp.create_prediction_pipeline_from_file(pred_params, 
                                               reload_dir=None, 
                                               model_path=model_path, 
                                               verbose=False)
    # this is necessary to let other models to use GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"     
    return pipe 
     
pipe = get_pipeline(pred_params=pred_params,model_path=model_path)

# =============================== #

def get_mol(smiles_or_mol):                                                     
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''                                                                                                                                       
    if isinstance(smiles_or_mol, str):                                          
        if len(smiles_or_mol) == 0:                                              
            return None                                                           
        mol = Chem.MolFromSmiles(smiles_or_mol)                                 
        if mol is None:                                                          
            return None                                                           
        try:                                                                    
            Chem.SanitizeMol(mol)                                                 
        except ValueError:                                                      
            return None                                                           
        return mol                                                              
    return smiles_or_mol

def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def pXC50(smiles):
    with contextlib.redirect_stdout(None):
        pred_df = pipe.predict_on_smiles([smiles], AD_method='z_score')
        pIC50 = pred_df['pred'][0]
    return pIC50
    
def therapeutic_score(smiles):
    with contextlib.redirect_stdout(None):
        pIC50 = pXC50(smiles)
        therapeutic_score = spl_estimator(pIC50)
        #model = bionetgen.bngmodel("/home/alif/BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl", 'model')
        #model.parameters.IC50 = 10 ** (6 - pIC50)
        #result = bionetgen.run(model)
        #therapeutic_score = result['Apopt Repair Toy Model 011823 v2.0'][-1][6]
    
    return therapeutic_score
    
def actual_therapeutic_score(pIC50):
	with contextlib.redirect_stdout(None):
		model = bionetgen.bngmodel("/home/alif/BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl", 'model')
		model.parameters.IC50 = 10 ** (6 - pIC50)
		result = bionetgen.run(model)
		therapeutic_score = result['Apopt Repair Toy Model 011823 v2.0'][-1][6]
	return therapeutic_score

def penalized_logP(smiles: str, min_score=-float("inf")) -> float:
    """ calculate penalized logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    sa = SA(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767
        + (-sa + 3.05352042) / 0.83460587
        + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)


def SA(mol):
    return sascorer.calculateScore(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def QED(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.QED.qed(mol)
    
# =============================== #
## Estimated Model
from scipy.interpolate import CubicSpline
pIC50, _ = np.linspace(0.0, 10.0, num=201, retstep=True)
therap = []
#with tqdm(total=len(pIC50)) as pbar:
for pIC50_val in pIC50:
	therap.append(actual_therapeutic_score(pIC50_val))
#pbar.update(1)
spl_estimator = CubicSpline(pIC50, therap)

# =============================== #
