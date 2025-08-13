```python
# 先查看当前 PyTorch 和 CUDA 版本
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
```


```python
# !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
#   -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```


```python
import json
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import MultiLabelBinarizer

# 使用英文字体（默认是 DejaVu Sans，适合英文）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 如果你的分隔符是制表符，sep='|'
df = pd.read_csv('/content/drive/My Drive/familyinfo/dr_feature.csv', sep=',', quotechar='"')

def parse_eci_list(x):
    if pd.isna(x):
        return []
    # 去掉引号和空格
    s = str(x).strip('"').strip()
    # 按 | 分割（如果没有 | 就会返回一个只有一个元素的列表）
    parts = s.split('|')
    # 转成整数
    return [int(p) for p in parts if p.strip().isdigit()]

df['NIGHT_ECI_LIST'] = df['NIGHT_ECI_LIST'].apply(parse_eci_list)

df['row_id'] = df['row_id'].astype(str).str.replace(',', '')  # 去掉逗号
df['row_id'] = df['row_id'].astype(int)                      # 转成整数

# 转成字典：MSISDN（转成字符串） -> 特征列表
data_dict = dict(zip(df['row_id'].astype(int), df['NIGHT_ECI_LIST']))

edges=pd.read_csv("/content/drive/My Drive/familyinfo/dr_call_rowid.csv")

target_df=pd.read_csv("/content/drive/My Drive/familyinfo/msisdn_label.csv")
target_df['row_id'] = target_df['row_id'].astype(str).str.replace(',', '')  # 去掉逗号
target_df['row_id'] = target_df['row_id'].astype(int)

train_df = target_df[target_df['label'].notna()]

max_row = target_df.loc[target_df['label'].idxmax()]
print("label 最大值所在行：")
print(max_row)

print("50 top nodes features")
print(dict(list(data_dict.items())[:50]))
print("5 top nodes labels")
print(target_df.head(5).to_markdown())
print("5 last nodes")
print(target_df.tail(5).to_markdown())
```

    label 最大值所在行：
    row_id    1503.0
    label      490.0
    Name: 1502, dtype: float64
    50 top nodes features
    {1: [38886203395], 2: [38886203393, 38886203396], 3: [244082723, 38715588609], 4: [98865282], 5: [195909249, 38860619782, 38860619783], 6: [38886203395], 7: [258772104], 8: [203427780, 203427785], 9: [38822645761], 10: [38886203393], 11: [244072449, 54092882], 12: [138699523], 13: [258770308], 14: [54097489], 15: [244082723, 38777249793, 98868609], 16: [195907969, 195907973], 17: [38860779529], 18: [203435721], 19: [54094656], 20: [38822645761], 21: [138678069], 22: [38860640259], 23: [244031245, 38822526977], 24: [38860582922], 25: [258772611, 38905769986], 26: [203435721, 38886711299], 27: [244031498, 38822645761], 28: [244082720, 38715588610, 38777249794, 38822215681], 29: [38906601473, 38906601477, 98857601], 30: [38822666242], 31: [38777249795, 6122080], 32: [38886203393], 33: [258770817], 34: [203409858], 35: [258772642], 36: [244068388], 37: [38886641672, 38906597378], 38: [244072715, 258740865, 38860615685, 38885879809], 39: [38860615684], 40: [38822645761], 41: [244082696, 38777266181], 42: [244031498, 38822645761], 43: [244031489], 44: [38822645761], 45: [258770817], 46: [258742935], 47: [38715588610, 38777249794, 38777249797], 48: [195908225], 49: [258770820], 50: [54095200]}
    5 top nodes labels
    |    |   row_id |   label |
    |---:|---------:|--------:|
    |  0 |        1 |     nan |
    |  1 |        2 |     485 |
    |  2 |        3 |     nan |
    |  3 |        4 |     nan |
    |  4 |        5 |     nan |
    5 last nodes
    |       |   row_id |   label |
    |------:|---------:|--------:|
    | 18624 |    18625 |     nan |
    | 18625 |    18626 |     nan |
    | 18626 |    18627 |     nan |
    | 18627 |    18628 |     nan |
    | 18628 |    18629 |     nan |



```python
def encode_data_mlb(data_raw, light=False, n=60):
    """
    使用 MultiLabelBinarizer 对节点特征进行 multi-hot 编码。

    参数：
        data_raw: dict，形如 {node_id: [feature1, feature2, ...]}
        light: 是否仅处理前 n 个节点
        n: 如果 light=True，指定处理前 n 个节点数量

    返回：
        data_encoded: dict，每个节点的 multi-hot 向量（list）
        sparse_feat_matrix: torch.Tensor 稀疏特征矩阵（如果 light=True），否则为 None
        mlb: MultiLabelBinarizer 对象（包含 classes_ 属性）
    """
    # 选取需要编码的节点
    node_ids = list(data_raw.keys())
    if light:
        node_ids = node_ids[:n]

    # 构造特征列表（list of list）
    feature_list = [data_raw[node] for node in node_ids]

    # MultiLabelBinarizer 编码
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(feature_list)

    # 构造 data_encoded dict
    data_encoded = {str(node_ids[i]): encoded[i].tolist() for i in range(len(node_ids))}

    # 返回
    if light:
        sparse_feat_matrix = torch.tensor(encoded, dtype=torch.float)
        return data_encoded, sparse_feat_matrix, mlb
    else:
        sparse_feat_matrix = torch.tensor(encoded, dtype=torch.float)
        return data_encoded, sparse_feat_matrix, mlb
        # return data_encoded, None, mlb
```


```python
data_encoded_vis,sparse_feat_matrix_vis,mlb=encode_data_mlb(data_dict,light=False,n=100)


value_lengths = [len(v) for v in data_encoded_vis.values()]
print(value_lengths[:5])
print(len(data_encoded_vis))
print("5 top nodes features")
print(dict(list(data_encoded_vis.items())[:5]))

plt.figure(figsize=(25,25))
plt.imshow(sparse_feat_matrix_vis[:100,:],cmap='Greys')
```

    [532, 532, 532, 532, 532]
    18629
    5 top nodes features
    {'1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}





    <matplotlib.image.AxesImage at 0x7f6bdc987c50>




    
![png](README_files/README_4_2.png)
    



```python
feat_counts = [sum(v) for v in data_encoded_vis.values()]

# 所有出现的ECI ID组成全集
feats = []
for k, v in data_encoded_vis.items():
    feats.extend([i for i, val in enumerate(v) if val == 1])


# 标签类别分布
plt.hist(train_df['label'], bins=250)
plt.title("Train Class ID distribution (PHONE209)")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()

# 每个节点的特征数量分布
plt.hist(feat_counts, bins=20)
plt.title("Number of ECI features per node")
plt.xlabel("#Features")
plt.ylabel("Count")
plt.show()

# 所有 ECI ID 的分布（是否某些 ECI 特别频繁）
plt.hist(feats, bins=532)
plt.title("ECI ID distribution")
plt.xlabel("ECI ID value")
plt.ylabel("Frequency")
plt.show()
```


    
![png](README_files/README_5_0.png)
    



    
![png](README_files/README_5_1.png)
    



    
![png](README_files/README_5_2.png)
    



```python
def construct_graph(data_encoded, target_df, edges, light=False):
    # 确保 data_encoded 的 key 是 int
    data_encoded = {int(k): v for k, v in data_encoded.items()}

    # 所有节点 ID
    all_node_ids = sorted(data_encoded.keys())

    # 所有节点特征
    node_features_list = [data_encoded[k] for k in all_node_ids]
    node_features = torch.tensor(node_features_list, dtype=torch.float)
    print(node_features.shape)
    # 标签对齐（确保 row_id 存在于 data_encoded 中）
    valid_target_df = target_df[target_df['row_id'].isin(all_node_ids)].copy()
    valid_target_df = valid_target_df.set_index('row_id').loc[all_node_ids]
    node_labels = torch.tensor(valid_target_df['label'].values, dtype=torch.long)
    print(node_labels.shape)
    # 边构造（双向）
    edge_index = torch.tensor(edges.values.tolist(), dtype=torch.long).T
    edge_index_rev = edge_index[[1, 0], :]
    full_edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    # 构建完整图
    full_graph = Data(x=node_features,  edge_index=full_edge_index,y=node_labels)
    if not light:
        return full_graph

    # --- 轻量图（只保留有标签的节点） ---
    labeled_node_ids = sorted(target_df.dropna(subset=['label'])['row_id'].astype(int).tolist())
    labeled_node_ids = [nid for nid in labeled_node_ids if nid in data_encoded]

    sub_idx_map = {nid: i for i, nid in enumerate(labeled_node_ids)}
    sub_feat = torch.tensor([data_encoded[nid] for nid in labeled_node_ids], dtype=torch.float)
    sub_labels = torch.tensor(target_df.set_index('row_id').loc[labeled_node_ids]['label'].values, dtype=torch.long)

    # 提取子图边：两个端点都在 labeled_node_ids 中
    sub_edge_list = [
        [sub_idx_map[src], sub_idx_map[dst]]
        for src, dst in edges.values.tolist()
        if src in sub_idx_map and dst in sub_idx_map
    ]
    sub_edge_tensor = torch.tensor(sub_edge_list, dtype=torch.long).T
    sub_edge_rev = sub_edge_tensor[[1, 0], :]
    sub_edge_index = torch.cat([sub_edge_tensor, sub_edge_rev], dim=1)

    sub_graph = Data(x=sub_feat, edge_index=sub_edge_index,y=sub_labels)
    return sub_graph

def draw_graph(data0):
    # if data0.num_nodes > 100:
    #     print(f"This is a big graph with {data0.num_nodes} nodes. Skipping plot...")
    #     return

    # 转换成 NetworkX 格式（无向图，保留节点特征）
    data_nx = to_networkx(data0, to_undirected=True)

    # 取 PyG 中的标签作为颜色，确保按 NetworkX 节点顺序映射
    node_colors = data0.y[list(data_nx.nodes)]

    # 使用 spring layout 自动布局
    # pos = nx.spring_layout(data_nx, seed=42)  # 固定 seed 以便复现图形
    pos= nx.spring_layout(data_nx,scale =1,seed=42)
    plt.figure(figsize=(12, 8))

    # 绘制节点和边
    nx.draw(
        data_nx,
        pos,
        node_color=node_colors,
        cmap=plt.cm.Set1,
        node_size=60,
        edge_color='gray',
        linewidths=1,
        with_labels=True,
        font_size=5
    )

    plt.title("Graph Visualization")
    plt.show()
```


```python
class SocialGNN(torch.nn.Module):
    def __init__(self, num_of_feat, f,num_of_label):
        super(SocialGNN, self).__init__()
        self.conv1 = GCNConv(num_of_feat, f)
        self.conv2 = GCNConv(f, num_of_label)
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```


```python
def masked_loss(predictions,labels,mask):
    mask=mask.float()
    mask=mask/torch.mean(mask)
    loss=criterion(predictions,labels)
    loss=loss*mask
    loss=torch.mean(loss)
    return (loss)

def masked_accuracy(predictions,labels,mask):
    mask=mask.float()
    mask/=torch.mean(mask)
    accuracy=(torch.argmax(predictions,axis=1)==labels).long()
    accuracy=mask*accuracy
    accuracy=torch.mean(accuracy)
    return (accuracy)

def train_social_predict(net,data,epochs=10,lr=0.01):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 00001
    best_accuracy=0.0

    train_losses=[]
    train_accuracies=[]

    val_losses=[]
    val_accuracies=[]

    test_accuracies=[]

    for ep in range(epochs+1):
        optimizer.zero_grad()
        out=net(data)
        loss=masked_loss(predictions=out,labels=data.y,mask=data.train_mask)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        train_accuracy=masked_accuracy(predictions=out,labels=data.y,mask=data.train_mask)
        train_accuracies.append(train_accuracy)

        val_loss=masked_loss(predictions=out,
                             labels=data.y,
                             mask=data.val_mask)
        val_losses.append(val_loss.item())

        val_accuracy=masked_accuracy(predictions=out,
                                     labels=data.y,
                                     mask=data.val_mask)
        val_accuracies.append(val_accuracy)

        if np.round(val_accuracy,4)> np.round(best_accuracy ,4):
            print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}"
                      .format(ep+1,epochs, loss.item(), train_accuracy, val_accuracy))
            best_accuracy=val_accuracy
            torch.save(net.state_dict(), 'best_model_predict.pt')

    plt.plot(train_losses,label='Train Loss')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='Train Acc')
    plt.legend()
    plt.show()
```


```python
g = construct_graph(
    data_encoded=data_encoded_vis,
    target_df=target_df,
    edges=edges,
    light=False
)
num_nan = target_df['label'].isna().sum()
print(f"label 中 NaN 的数量为: {num_nan}")
labels = torch.tensor(target_df['label'].fillna(-1).astype(int).values)

#构造mask
train_mask = labels != -1      # 有标签的用来训练
predict_mask = labels == -1    # 无标签的我们要预测

idx = torch.where(labels != -1)[0]      # 有标签的部分用来训练
train_ratio = 0.8
num_train = int(train_ratio * len(idx))

perm = torch.randperm(len(idx))
train_idx = idx[perm[:num_train]]
val_idx = idx[perm[num_train:]]

train_mask = torch.zeros_like(labels, dtype=torch.bool)
val_mask = torch.zeros_like(labels, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True


g.y = labels                # 标签
g.y=g.y-1
g.train_mask = train_mask
g.val_mask=val_mask
g.predict_mask = predict_mask
print(g)

```

    torch.Size([18629, 532])
    torch.Size([18629])
    label 中 NaN 的数量为: 18083
    Data(x=[18629, 532], edge_index=[2, 2120], y=[18629], train_mask=[18629], val_mask=[18629], predict_mask=[18629])



```python
num_of_feat=g.num_node_features
num_of_label = int(target_df['label'].nunique())

print(min(g.y))
print(num_of_feat,num_of_label)
net=SocialGNN(num_of_feat=num_of_feat,f=200,num_of_label=num_of_label)
criterion=nn.CrossEntropyLoss(ignore_index=-2)
train_social_predict(net,g,epochs=50,lr=0.1)
```

    tensor(-2)
    532 490
    Epoch 2/50, Train_Loss: 5.5434, Train_Accuracy: 0.1330, Val_Accuracy: 0.1545
    Epoch 3/50, Train_Loss: 3.5726, Train_Accuracy: 0.4358, Val_Accuracy: 0.3545
    Epoch 4/50, Train_Loss: 1.5398, Train_Accuracy: 0.6766, Val_Accuracy: 0.6273
    Epoch 5/50, Train_Loss: 0.8055, Train_Accuracy: 0.7202, Val_Accuracy: 0.7545
    Epoch 7/50, Train_Loss: 0.7167, Train_Accuracy: 0.7569, Val_Accuracy: 0.8182
    Epoch 8/50, Train_Loss: 0.6432, Train_Accuracy: 0.7615, Val_Accuracy: 0.8545
    Epoch 11/50, Train_Loss: 0.5503, Train_Accuracy: 0.8096, Val_Accuracy: 0.8818
    Epoch 18/50, Train_Loss: 0.3704, Train_Accuracy: 0.8234, Val_Accuracy: 0.9000
    Epoch 42/50, Train_Loss: 0.3047, Train_Accuracy: 0.8326, Val_Accuracy: 0.9091



    
![png](README_files/README_10_1.png)
    



    
![png](README_files/README_10_2.png)
    



```python
# # 重新构建同样结构的模型
# net = SocialGNN(num_of_feat=g.num_node_features, f=200, num_of_label=num_of_label)

# # 加载训练好的权重
# net.load_state_dict(torch.load('best_model_predict.pt'))
net.eval()
with torch.no_grad():  # 预测时不需要计算梯度
    out = net(g)  # 模型输出 logits
    preds = out.argmax(dim=1)  # 每个节点预测的类别（取最大概率索引）

# 只取 test_mask 对应的预测结果
preds = preds[g.predict_mask]
preds_original = preds + 1
print("预测标签：", preds_original)

# 获取 predict_mask 对应的节点索引
pred_indices = torch.where(g.predict_mask)[0].numpy()
print(len(pred_indices))

# 创建预测 DataFrame
result_df = pd.DataFrame({
    'node_id': pred_indices,
    'pred_label': (preds + 1).numpy()  # 转回 1~490
})
print(result_df)
```

    预测标签： tensor([ 75, 131, 200,  ...,   8, 419,  76])
    18083
           node_id  pred_label
    0            0          75
    1            2         131
    2            3         200
    3            4         354
    4            5          75
    ...        ...         ...
    18078    18624         354
    18079    18625         333
    18080    18626           8
    18081    18627         419
    18082    18628          76
    
    [18083 rows x 2 columns]



```python
# 标签类别分布
plt.hist(result_df['pred_label'], bins=250)
plt.title("Predict Class ID distribution (PHONE209)")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()
```


    
![png](README_files/README_12_0.png)
    

