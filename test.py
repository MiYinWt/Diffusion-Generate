import torch

data = torch.load('./datas/crossdocked2020/crossdocked_pocket10_pose_split.pt')


import sys

print(data['train'])