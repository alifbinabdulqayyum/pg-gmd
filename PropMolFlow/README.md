# PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching

## Install Conda Environment & Download Data
Follow the conda environment installation and data download procedure mentioned in the original [PropMolFlow](https://github.com/Liu-Group-UF/PropMolFlow) repository.

## Add the pIC50 data into the processed datafiles
Run `Update_Data.py` to add the pIC50 values into the processed datafiles.

## Train & Generate
Run `train.sh` for training the PropMolFlow model. Then run `sample.sh` for generation with the trained models. Modify the config files accordingly to tweak with the data and model hyperparameters.