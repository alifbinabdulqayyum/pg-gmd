gpu="--gpu"  # change to "" if no GPU is to be used
seed=730007773

pathway_model="viable"
parp_model_path="updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"
bngl_model_path="BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl"
root_dir="logs/bo/chem_therapeutic_score_$pathway_model"
start_model="data/models/chem.ckpt"
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1

r=50
weight_type="rank"
lso_strategy="opt"

k=6

echo "Property to optimize: therapeutic score"
echo "Pathway Model to be used: $pathway_model"

# Run command
python weighted_retraining/opt_scripts/opt_chem.py \
    --batch_size=32 \
    --seed="$seed" $gpu \
    --query_budget="$query_budget" \
    --retraining_frequency="$r" \
    --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
    --pretrained_model_file="$start_model" \
    --lso_strategy="$lso_strategy" \
    --train_path="data/chem/orig_model/tensors_train" \
    --val_path="data/chem/orig_model/tensors_val" \
    --vocab_file="data/chem/orig_model/vocab.txt" \
    --property_file="data/chem/orig_model/therapeutic_score_$pathway_model.pkl" \
    --property='therapeutic_score' \
    --n_retrain_epochs="$n_retrain_epochs" \
    --n_init_retrain_epochs="$n_init_retrain_epochs" \
    --n_best_points=2000 --n_rand_points=8000 \
    --n_inducing_points=500 \
    --invalid_score=0 \
    --weight_type="$weight_type" --rank_weight_k="1e-$k" \
    --samples_per_model=1000 \
    --pathway_model="$pathway_model" \
    --parp_model_path="$parp_model_path" \
    --bngl_model_path="$bngl_model_path"