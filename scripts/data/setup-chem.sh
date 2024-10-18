# Copy stored data to correct folder
chem_dir="data/chem/orig_model"
mkdir -p "$chem_dir"
cp -r assets/data/*.txt "$chem_dir"

# Normally you might make the vocab, but the vocab is already made
# (it was copied from the original repo,
# so may not be exactly reproducible with the code in this repo)
# To make vocab for another model/dataset, run a command like the following:

# Before running this, it is suggested to uninstall the rdkit-pypi, rdkit, and then reinstall rdkit only:
# ```
pip uninstall -y rdkit-pypi==2022.3.1
conda remove --force rdkit==2020.09.5

conda install https://conda.anaconda.org/conda-forge/linux-64/rdkit-2020.09.5-py37he53b9e1_0.tar.bz2
# ```
python weighted_retraining/chem/create_vocab.py \
    --input_file=data/chem/orig_model/train.txt \
    --output_file=data/chem/orig_model/vocab-CHECK.txt
# After this is done, reinstall rdkit-pypi:
# ```
pip install -U rdkit-pypi=2022.3.1
# ```