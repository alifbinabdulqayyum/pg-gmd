# Pathway-Guided Optimization of Deep Generative Molecular Design Models for Cancer Therapy

First we have to download the [ATOM Modeling PipeLine (AMPL) for Drug Discovery](https://github.com/ATOMScience-org/AMPL) package version 1.4.2.
```
# Download AMPL package version 1.4.2
wget https://github.com/ATOMScience-org/AMPL/archive/refs/tags/1.4.2.tar.gz

# Unzip the folder
tar -xvzf 1.4.2.tar.gz
```

Then create a conda environment and install necessary packages:
```
# Create the environment
conda create -y -n pggmd --file conda_package_list.txt

# Activate the conda environment
conda activate pggmd

# Install necessary pip packages
pip install -r pip_requirements.txt
pip uninstall -y keras
pip install -U tensorflow==2.8.0 keras==2.8.0


# Install AMPL package
cd AMPL-1.4.2
./build.sh && ./install.sh system
cd ..
# Uninstall rdkit-pypi as this conflicts with weighted-retraining requirements
pip uninstall -y rdkit-pypi
conda remove --force rdkit==2020.09.5

# Then reinstall rdkit again
conda install https://conda.anaconda.org/conda-forge/linux-64/rdkit-2020.09.5-py37he53b9e1_0.tar.bz2
```

Then we need to install the project which is based on [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://github.com/cambridge-mlg/weighted-retraining) paper.
```
python -m pip install -e .
```

## Preprocess the data

First, to preprocess the dataset, run the following command: 
```
bash scripts/data/setup-chem.sh
```

## Train the unweighted model

We then train the unweighted model, by running the following command:
```
bash scripts/models/train-chem.sh
```
We provided a unweighted trained model in this repository. 

## Optimize the model through weighted retraining

To retrain the JTVAE with viable pathway model, run the following command:
```
bash scripts/opt/opt-chem-viable.sh
```

To retrain the JTVAE with modified pathway model, run the following command:
```
bash scripts/opt/opt-chem-modified.sh
```

To retrain the JTVAE with impractical pathway model, run the following command:
```
bash scripts/opt/opt-chem-impractical.sh
```

To retrain the JTVAE for pIC50 optimization, run the following command:
```
bash scripts/opt/opt-chem-pXC50.sh
```

To optimize the models with different `k` values, update the `k` value in the bash scripts:
```
k = 5 
```
```
k = 4 
```
```
k = 3
```

## Sample molecules with optimized JTVAE at consecutive retraining iterations

Run the `Sample-molecules.ipynb` script 

## Calculate the therapeutic scores of the generated molecules

To measure the therapeutic score of the generated molecules for different pathway models, run the following command:
```
bash scripts/plots/measure-chem-property.sh
```

## Plot the results

Run the `Plot-result.ipynb` script

## Training The Conditional Generative Models

Check `PropMolFlow` subfolder for training the conditional generative models.

## Acknowledgements

We sincerely thank the authors of [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://github.com/cambridge-mlg/weighted-retraining) and [PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching](https://github.com/Liu-Group-UF/PropMolFlow) for their excellent work. Our code implementations are based on these two works.