parp_model_path="updated_pXC50_predictor/PARP1_CGUAgg_2022-06_fingerprint_graphconv_model_4f296899-1e4f-4d08-a7c5-47ef64d7fec3.tar.gz"
bngl_model_path="BioNetGen/Apopt Repair Toy Model 011823 v2.0.bngl"

sample_path="sample-results"

save_filedir="gen-mols-property" 

ncpu=16

# Calculate properties of generated molecules with differnet pathway models
for pathway_model in "viable" "modified" "impractical";
do
    python scripts/plots/calculate_score_gen_molecule.py \
        --parp-model-path="$parp_model_path" \
        --bngl-model-path="$bngl_model_path" \
        --pathway-model="$pathway_model" \
        --save-filedir="$save_filedir" \
        --sample-path="$sample_path" \
        --ncpu=$ncpu
done