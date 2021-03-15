#dataset directory
DATA=$1
dataset=../../data/${DATA}/intermediate
python generate_bash.py $dataset

cd ../corel/c
chmod +x run_emb_full_tax.sh
cd ../..
