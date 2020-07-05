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
print(effnet_b2.shape)
print(effnet_b2[0].shape)

softmax = nn.Softmax(dim=-1)

effnet_b2 = softmax(effnet_b2)
# for idx, b in enumerate(effnet_b2):


