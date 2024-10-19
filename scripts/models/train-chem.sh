# Script to train all models for the paper
# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=730007773
root_dir="logs/train"
parp_model_path="updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"

# Train chem VAE
python weighted_retraining/train_scripts/train_chem.py \
    --root_dir="$root_dir" \
    --seed="$seed" $gpu \
    --beta_final=0.005 --lr=0.0007 --latent_dim=56 \
    --max_epochs=30 --batch_size=64 \
    --train_path="data/chem/orig_model/tensors_train" \
    --val_path="data/chem/orig_model/tensors_val" \
    --vocab_file="data/chem/orig_model/vocab.txt" \
    --property_file="data/chem/orig_model/pIC50.pkl" \
    --parp_model_path="$parp_model_path" \
    --property='pXC50' \
