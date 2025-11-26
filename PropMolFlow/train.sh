source "/raid/alifbinabdulqayyum/anaconda3/bin/activate"
conda activate propmolflow

cd "/raid/alifbinabdulqayyum/PropMolFlow"

python train.py --config=./configs/with_gaussian_expansion/pIC50_concatenate_multiply.yaml &
python train.py --config=./configs/with_gaussian_expansion/pIC50_multiply.yaml &
python train.py --config=./configs/with_gaussian_expansion/pIC50_concatenate_sum.yaml &
python train.py --config=./configs/with_gaussian_expansion/pIC50_concatenate.yaml &
python train.py --config=./configs/with_gaussian_expansion/pIC50_sum.yaml &

wait

echo "DONE"