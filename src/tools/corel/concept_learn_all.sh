#dataset directory
dataname=$1
dataset=../../data/${dataname}/intermediate
python generate_bash.py $dataset

cd ../corel/c
chmod +x run_emb_full_tax.sh
cd ../..
