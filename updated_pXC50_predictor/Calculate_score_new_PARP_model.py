'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''

import warnings
# warnings.filterwarnings(action='once')
warnings.simplefilter("ignore")

import timeit

start = timeit.default_timer()

#Your statements here

import bionetgen
import contextlib
import os
import pandas as pd

import numpy as np

import pickle

# from atomsci.ddm.pipeline import predict_from_model as pfm

from multiprocessing import Pool

'''
# xc_path = "/home/grads/a/alifbinabdulqayyum/PARP1/the_final_xc50_data_model_396cdb1d-1c54-4014-b84f-07ca5ed7f506.tar.gz"
# xc_path = "/media/alif/Alif/Research(Yoon)/PARP1/the_final_xc50_data_model_396cdb1d-1c54-4014-b84f-07ca5ed7f506.tar.gz"
xc_path = "/home/alif/JTVAE/updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"

model_path = xc_path
smiles_col = 'smiles'
response_col = 'pXC50'
dont_standardize = True
is_featurized = False


if os.path.exists('test_pXC50.txt'):
    os.remove('test_pXC50.txt')
'''
if os.path.exists('test_therapeutic_score.txt'):
    os.remove('test_therapeutic_score.txt')
'''
with open('/media/alif/Alif/Research(Yoon)/weighted-retraining/data/chem/zinc/orig_model/pen_logP_all.pkl', "rb") as f:
    property_dict = pickle.load(f)
    
all_smiles = list(property_dict.keys())

# all_smiles = all_smiles[:2500]

input_df = pd.DataFrame(all_smiles, columns = ['smiles'])

pred_df = pfm.predict_from_model_file(model_path = model_path, 
                                      input_df=input_df, 
                                      smiles_col=smiles_col, 
                                      response_col=response_col,
                                      dont_standardize=dont_standardize, 
                                      is_featurized = is_featurized, 
                                      AD_method='z_score')
                                      
pXC50_dict = {}
for smiles, pXC50 in zip(pred_df['smiles'], pred_df['activity_value_pred']):
    pXC50_dict[smiles] = pXC50
    with open('test_pXC50.txt', 'a') as f:
        f.writelines(smiles+': '+str(pXC50)+'\n')
        
with open('pXC50_new_PARP.pkl', 'wb') as handle:
    pickle.dump(pXC50_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
from tqdm import tqdm

def process(data):
    vocab = set()
    for smiles in data:
        s = calc_therapeutic_score(smiles)
        vocab.add( (smiles, s) )
    return vocab

def calc_therapeutic_score(smiles):
    with contextlib.redirect_stdout(None):
        pXC50 = pXC50_dict[smiles]
        model = bionetgen.bngmodel("/home/alif/BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl", 'model')
        model.parameters.IC50 = 10 ** (6 - pXC50)
        result = bionetgen.run(model)
        therapeutic_score = result['Apopt Repair Toy Model 011823 v2.0'][-1][6]
        del model
    return therapeutic_score

with open('pXC50_new_PARP.pkl', "rb") as f:
    pXC50_dict = pickle.load(f)
    all_smiles = list(pXC50_dict.keys())
    
all_smiles = all_smiles[0:1000]

ncpu = 10

batch_size = len(all_smiles) // ncpu + 1
batches = [all_smiles[i : i + batch_size] for i in range(0, len(all_smiles), batch_size)]

pool = Pool(ncpu)
vocab_list = pool.map(process, batches)
vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
vocab = list(set(vocab))

print(vocab)

therap_dict = {}
for smiles, therapeutic_score in sorted(vocab):
    therap_dict[smiles] = therapeutic_score
    with open('therapeutic_score_new_PARP_peak_6.txt', 'a') as f:
    	f.writelines(smiles+': '+str(therapeutic_score)+'\n')
    	
with open('therapeutic_score_new_PARP_peak_6.pkl', 'wb') as handle:
    pickle.dump(therap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

stop = timeit.default_timer()

print('Time: ', stop - start) 
