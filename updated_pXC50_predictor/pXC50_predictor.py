import timeit

start = timeit.default_timer()

import glob
import numpy as np

result_dir = "/home/alif/JTVAE/sample-results"
res_files = glob.glob(result_dir+'/*.npz')

all_smiles = []

for res_file in res_files:
    results = np.load(res_file, allow_pickle = True)

    smiles = results['sample_points'].reshape((-1, ))

    smiles = set(smiles)
    if None in smiles:
        smiles.remove(None)
    print(res_file, len(smiles))
    all_smiles.extend(list(smiles))
    
import bionetgen
import contextlib
import os
import pandas as pd

import numpy as np

import pickle

from atomsci.ddm.pipeline import predict_from_model as pfm

xc_path = "/home/alif/JTVAE/updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"

model_path = xc_path
smiles_col = 'smiles'
response_col = 'pXC50'
dont_standardize = True
is_featurized = False

# all_smiles = all_smiles[:10000]
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
    
with open('gen_pXC50.pkl', 'wb') as handle:
    pickle.dump(pXC50_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
end = timeit.default_timer()

print("Total Time Taken: ", end-start)
