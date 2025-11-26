data_split='second'

for method in 'concatenate' 'concatenate_multiply' 'multiply' 'concatenate_sum' 'sum';
# for method in 'sum' 'concatenate';
do
    python sample_pIC50.py \
        --model-dir ./pIC50_amanda_"$method"_updated_"$data_split"halfdata \
        --save-dir ./sample_updated_"$data_split"halfdata-G25 \
        --method $method \
        --n-timesteps 200 \
        --sample-pIC50 8.0 #&
done

wait 
echo "DONE"