# Pathway-Guided Optimization of Deep Generative Molecular Design Models for Cancer Therapy

First we have to download the [ATOM Modeling PipeLine (AMPL) for Drug Discovery](https://github.com/ATOMScience-org/AMPL) package version 1.4.2.
```
wget https://github.com/ATOMScience-org/AMPL/archive/refs/tags/1.4.2.tar.gz
tar -xvzf 1.4.2.tar.gz
```

Then create a conda environment and install necessary packages:
```
conda create -y -n pggmd --file conda_package_list.txt
conda activate pggmd
pip install -r pip_requirements.txt
cd AMPL-1.4.2
./build.sh && ./install.sh system
cd ..
```

Then we need to install the project which is based on [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://github.com/cambridge-mlg/weighted-retraining) paper.
```
python -m pip install -e .
```