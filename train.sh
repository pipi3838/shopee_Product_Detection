# data preprocessing (specify the data file path)
python preprocess.py $1

# training
python -m src.main configs/train/fix_effnet.yaml