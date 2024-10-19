# Copy stored data to correct folder
chem_dir="data/chem/orig_model"
mkdir -p "$chem_dir"
cp -r assets/data/*.txt "$chem_dir"

# Normally you might make the vocab, but the vocab is already made
# (it was copied from the original repo,
# so may not be exactly reproducible with the code in this repo)
# To make vocab for another model/dataset, run a command like the following:

# python weighted_retraining/chem/create_vocab.py \
#     --input_file=data/chem/orig_model/train.txt \
#     --output_file=data/chem/orig_model/vocab-CHECK.txt

# # Preprocess the data
# preprocess_script="weighted_retraining/chem/preprocess_data.py"

# # Training Set
# out_dir="$chem_dir"/tensors_train
# mkdir "$out_dir"
# python "$preprocess_script" \
#     -t "$chem_dir"/train.txt \
#     -d "$out_dir" 

# # Validation Set
# out_dir="$chem_dir"/tensors_val
# mkdir "$out_dir"
# python "$preprocess_script" \
#     -t "$chem_dir"/val.txt \
#     -d "$out_dir" 

therap_script="weighted_retraining/chem/calc_therapeutic_score.py"
pIC50_script="weighted_retraining/chem/calc_pXC50.py"
parp_model_path="updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"
bngl_model_path="BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl"

# Create a file for all molecules
all_smiles_file="$chem_dir/all.txt"
cat "$chem_dir/train.txt" "$chem_dir/val.txt" > "$all_smiles_file"

# Calculate pIC50
python "$pIC50_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/pIC50.pkl" \
    --parp_model_path="$parp_model_path" 

# Calculate Therapeutic Score for Viable Pathway Model
pathway_model="viable"
python "$therap_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/therapeutic_score_$pathway_model.pkl" \
    --pathway_model="$pathway_model" \
    --parp_model_path="$parp_model_path" \
    --bngl_model_path="$bngl_model_path"

# Calculate Therapeutic Score for Modified Pathway Model
pathway_model="modified"
python "$therap_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/therapeutic_score_$pathway_model.pkl" \
    --pathway_model="$pathway_model" \
    --parp_model_path="$parp_model_path" \
    --bngl_model_path="$bngl_model_path"

# Calculate Therapeutic Score for Modified Pathway Model
pathway_model="impractical"
python "$therap_script" \
    --input_file "$chem_dir/train.txt" "$chem_dir/val.txt" \
    --output_file="$chem_dir/therapeutic_score_$pathway_model.pkl" \
    --pathway_model="$pathway_model" \
    --parp_model_path="$parp_model_path" \
    --bngl_model_path="$bngl_model_path"