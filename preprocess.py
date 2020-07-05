import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2

data_path = '/nfs/nas-5.1/wbcheng/shopee_task2/'

data_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
data_df.columns = ['Filename', 'Category']

print(data_df.shape)
# print(data_df['Category'].value_counts())

img_list = []
img_class_list = []

cnt = 0
discard_cnt = 0

img_dir = os.path.join(data_path, 'train/train/')
for img_name, label in zip(data_df['Filename'].values, data_df['Category'].values):
    img_path = os.path.join(img_dir, str(label).zfill(2), img_name)
    img = cv2.imread(img_path)
    row, col, _ = img.shape
    if row > 300 and col > 300: 
        img_list.append(img_name)
        img_class_list.append(label)
    else: discard_cnt += 1
    print('procseeing {}th img'.format(cnt+1), end='\r')
    cnt += 1

print()
print('Modified Count: ', cnt, discard_cnt)
modified_df = pd.DataFrame({'Filename': img_list, 'Category': img_class_list})
modified_df.to_csv(os.path.join(data_path, 'modified_train.csv'), index=False, header=False)

# modified_df = pd.read_csv(os.path.join(data_path, 'modified_train.csv'))
train_df, valid_df = train_test_split(modified_df, test_size=0.2)

train_df.to_csv(os.path.join(data_path, 'modified_split_train.csv'), index=False, header=False)
valid_df.to_csv(os.path.join(data_path, 'modified_split_valid.csv'), index=False, header=False)

print(train_df.shape)
print(valid_df.shape)

# train_df.to_csv(os.path.join(data_path, 'split_train.csv'), index=False, header=False)
# valid_df.to_csv(os.path.join(data_path, 'split_valid.csv'), index=False, header=False)

# tmp_df = pd.read_csv(os.path.join(data_path, 'split_valid.csv'), names=['filename', 'category'])
# print(tmp_df.head())