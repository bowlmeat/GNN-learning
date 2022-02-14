#
# # 读取数据
# # def load_feat():
# #     path = "data.csv"
# #     feat_list = []
# #     with open(path, 'r') as f:
# #         for line in f.readlines():
# #             feat = line.strip().split(',')
# #             feat_list.append(feat[1:])  # 去除id
# #         # hg.ndata['feature'] = feat_list
# #     return feat_list
#
#
#
#
# #
# # def load_feat():
# #     path = "data.csv"
# #     feat_list = []
# #     with open(path, 'r') as f:
# #         for line in f.readlines()[1:]:
# #             feat = line.strip().split(',')
# #             feat = [[int(f)] for f in feat[1:]] # 升维度，变数据类型
# #             feat_list.append(feat)  # 去除id
# #         # hg.ndata['feature'] = feat_list
# #     return feat_list
# #
# #
# #
# # mydata = load_feat()
# # print(mydata)
#
#
# import torch
# import numpy as np
#
# # 处理维度
# def data_process(x):
#     x = x.squeeze()
#     x = x.numpy()
#     y = []
#     for i in range(x.size):
#         y.append([x[i]])
#
#     return torch.tensor(y)
#
#
# # 处理字典
# def vals_process(dict):
#     for val in dict.values():
#         return val
#
#
#
#
# a = {'road': torch.tensor([1, 2, 3, 4, 5])}
# print(vals_process(a))
#
#
# # print("土法调试")
# # print("边的类型" + etype)
# # print("源点信息" + srctype)
# # print(Wh)
#
# import torch
# import torch.utils.data as data
#
# def load_feat():
#     path = "data.csv"
#     feat_list = []
#     with open(path, 'r') as f:
#         for line in f.readlines()[1:]:
#             feat = line.strip().split(',')
#             feat = [[int(f)] for f in feat[1:]]  # 升维度，变数据类型
#             feat_list.append(feat)  # 去除id
#         # hg.ndata['feature'] = feat_list
#     # print(feat_list)
#     return feat_list
#
#
# def load_array(feat, batch_size, is_train=True):
#     """构造数据迭代器"""
#     feat = torch.tensor(feat)
#     dataset = data.TensorDataset(feat)   # 这里的*：作用于可迭代对象
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)
#
# batch_size = 1
# data_iter = load_array(load_feat(), batch_size)
# # print(next(iter(data_iter)))
#
#
# for node_feat in data_iter:
#     for feat in node_feat:
#         print('-----------')
#         print(feat)
import torch
import torch.utils.data as data

# 读取数据
def load_feat():
    path = "data.csv"
    feat_list = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            feat = line.strip().split(',')
            feat = [[int(f)] for f in feat[1:]]  # 升维度，变数据类型
            feat_list.append(feat)  # 去除id
        # hg.ndata['feature'] = feat_list
    return feat_list


def load_array(feat, batch_size, is_train=True):
    """构造数据迭代器"""
    feat = torch.tensor(feat, dtype=torch.float)    ## 设置数据类别
    dataset = data.TensorDataset(feat)   ## 这里的*：作用于可迭代对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 1
data_iter = load_array(load_feat(), batch_size)

hg = DataGraph().hg
model = Model(
    in_size=6,
    hidden_size=6,
    out_size=6,
    canonical_etypes=hg.canonical_etypes
)

optimizer = torch.optim.Adam(model.parameters())

# 训练
for feat in data_iter:  # 迭代
    for node_feat in feat:
        for epoch in range(10):
            # node_feat = torch.tensor(node_feat, dtype=torch.float).T
            # node_feat = np.array(node_feat).flatten()
            # print('--------------')
            # print(node_feat)

            feat, pos_score, neg_score = model(hg, node_feat)
            loss = compute_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_l = loss(model(hg, node_feat))
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

































