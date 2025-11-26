import torch
import pickle
import os
from tqdm import tqdm

os.makedirs(f'./data/qm9_updated', exist_ok=True)

data_split_list = ["train_a", "train_b", "test", "val"]

for data_split in tqdm(data_split_list):
    data_file = f'./data/qm9/{data_split}_data_processed.pt'
    data_dict = torch.load(data_file, weights_only=False)

    # # Comment this part in the original code
    # data_dict['properties'] = data_dict['properties'][:,:-1]
    # data_dict['property_names'] = data_dict['property_names'][:-1]
    # ==== #

    with open(f'./data/qm9_raw/qm9_pIC50_amanda.pkl', 'rb') as file:
        pIC50_dict = pickle.load(file)

    pIC50_list = []
    for smiles in data_dict['smiles']:
        pIC50_list.append(pIC50_dict[smiles])

    data_dict['properties'] = torch.cat([data_dict['properties'], torch.tensor(pIC50_list)[:,None]], dim=-1)#.shape
    data_dict['property_names'].append('pIC50')

    data_file = f'./data/qm9_updated/{data_split}_data_processed.pt'
    
    torch.save(data_dict, data_file)