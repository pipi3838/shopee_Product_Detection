import numpy as np
import pandas as pd
from torch import nn

# ResNet_modified_aug
# EfficientNet_b2_aug
# Effnet_b4_modified

data_path = '/nfs/nas-5.1/wbcheng/shopee_task2/test.csv'
df = pd.read_csv(data_path, names=['filename', 'category'], header=None)
filename = df['filename'].values[1:]

effnet_b2 = np.load('./effnet_b2_aug.npy', allow_pickle=True)
effnet_b3 = np.load('./effnet_b3fc_aug.npy', allow_pickle=True)
resnet_aug = np.load('./resnet_aug.npy', allow_pickle=True)

# print(resnet_aug.shape)
# print(resnet_aug[0].shape)

final_ans = []
softmax = nn.Softmax(dim=-1)

for b2_logit, b3_logit, resnet_logit in zip(effnet_b2, effnet_b3, resnet_aug):
    b2_logit, b3_logit, resnet_logit = softmax(b2_logit), softmax(b3_logit), softmax(resnet_logit)
    tmp = b2_logit + 3 * b3_logit + 1.5 * resnet_logit
    pred = tmp.argmax(dim=-1, keepdim=False)
    final_ans.append(pred)

output = open('ensemble3_ver3_ans.csv', 'w')
output.write('filename,category\n')

for f, p in zip(filename, final_ans):
    output.write('{},{}\n'.format(f, str(p.item()).zfill(2)))