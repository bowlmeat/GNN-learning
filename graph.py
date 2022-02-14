# -*- coding: utf-8 -*-
import torch
import dgl

import numpy as np
import pandas as pd

torch.manual_seed(0)

# -----设置搜索路径------
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
# ---------------------------

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


# 处理维度
def data_process(x):
    x = x.squeeze()
    x = x.detach().numpy()    # 梯度，需要detach
    y = []
    for i in range(x.size):
        y.append([x[i]])

    return torch.tensor(y)

# 处理字典
def vals_process(dict):
    for val in dict.values():
        return val

# 设置基本参数
num_nodes = 6
num_edges = 15

# 建图: 图的结构
signal1_src = [0, 2, 4]
signal1_dst = [1, 3, 5]
signal2_src = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
signal2_dst = [2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 4, 5]

# 定义类
class DataGraph():

    def __init__(self):   # 建图基本结构

        num_nodes_dict = {'road': num_nodes}

        self.hg = dgl.heterograph({
            ('road', 'signal1', 'road'): (torch.tensor(signal1_src), torch.tensor(signal1_dst)),
            ('road', 'signal2', 'road'): (torch.tensor(signal2_src), torch.tensor(signal2_dst))
        }, num_nodes_dict=num_nodes_dict)
        # self.hg = dgl.DGLHeteroGraph(self.hg)

        # 双向图
        self.hg = dgl.add_reverse_edges(self.hg)  # 玛德没法用双向图  ## 现在可以用了

        # 随机两类图的边特征
        self.hg.edges['signal1'].data['m'] = torch.randn(self.hg.number_of_edges('signal1'), 1)
        self.hg.edges['signal2'].data['m'] = torch.randn(self.hg.number_of_edges('signal2'), 1)

        # self.hg.nodes['road'].data['h'] = torch.tensor()

    # 构建图的节点特征
    # hg.nodes['road'].data['h'] = data()[0]    # roads_feats
    # feat_list = load_feat()


# -------------往下都是尝试---------
# hg = DataGraph().hg
# print(hg.canonical_etypes)

# 自定义单层
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, hg, feat):
        # print('---------look---------')
        # print(feat.shape)
        # print(feat)
        # print(torch.squeeze(feat, 0))
        # feat = torch.squeeze(feat, 0).T
        funcs = {}
        for srctype, etype, dsttype in hg.canonical_etypes:
            Wh = self.weight[etype](feat)     # 我要：每一种边对应的所有结点的特征 Wh = self.weight[etype](feat_dict[srctype])
            Wh = data_process(Wh)
            # Wh = Wh.reshape(6, 1)     # 转换为列向量
            hg.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))   # 在graph.ndata['h']中存储结果

        hg.multi_update_all(funcs, 'sum')
        return hg.nodes['road'].data['h']


# 卷积网络
class HeteroRGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, canonical_etypes):
        super(HeteroRGCN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        etypes = [etype for _, etype, _ in canonical_etypes]
        self.conv1 = HeteroRGCNLayer(self.in_size, self.hidden_size, etypes)
        self.conv2 = HeteroRGCNLayer(self.hidden_size, self.out_size, etypes)

    def forward(self, hg, feat):
        feat = self.conv1(hg, feat)
        feat = F.relu(feat)
        feat = self.conv2(hg, feat.T)
        return feat


# 计算相近度
class HeteroScorePredictor(nn.Module):
    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['s'] = h
            for etype in hg.canonical_etypes:
                hg.apply_edges(dgl.function.u_dot_v('s', 's', 'score'))

            # return {hg.edges[e]: hg.edges[e].data['score'] for e in hg.edges}
            return hg.edata['score']


# 模型
class Model(nn.Module):
    """sum:net输出可能是负值，目前没有约束"""
    def __init__(self, in_size, hidden_size, out_size, canonical_etypes):
        super(Model, self).__init__()
        self.net = HeteroRGCN(in_size, hidden_size, out_size, canonical_etypes)
        self.pred = HeteroScorePredictor()

    def forward(self, hg, h):
        h = self.net(hg, h)

        self.pos_g = hg.edge_type_subgraph(['signal1'])    # 提取子图
        self.neg_g = hg.edge_type_subgraph(['signal2'])

        pos_score = self.pred(self.pos_g, h)
        neg_score = self.pred(self.neg_g, h)

        return h, pos_score, neg_score


# model = Model(6, 6, 6, hg.canonical_etypes)
# h, pos_score, neg_score = model(hg, torch.tensor([[24], [0], [24], [18], [12], [18]], dtype=torch.float).T)

# print("----------------")
# print(pos_score)
# print('-----------------------')
# print(neg_score)
# heteroRGCN = HeteroRGCN(6, 6, 6, hg.canonical_etypes)
# h, pos_score, neg_score = heteroRGCN(hg, torch.tensor([[24], [0], [24], [18], [12], [18]], dtype=torch.float).T)   ## 必须调整数据类型为float



def load_array(feat, batch_size, is_train=True):
    """构造数据迭代器"""
    feat = torch.tensor(feat, dtype=torch.float)    ## 设置数据类别
    dataset = data.TensorDataset(feat)   ## 这里的*：作用于可迭代对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# batch_size = 10
# data_iter = load_array(load_feat(), batch_size)
# print(next(iter(data_iter)))



def compute_loss(pos_score, neg_score):    # clamp?
    """计算损失"""
    pos = - torch.sum(torch.log(F.softmax(pos_score)))    # 损失函数
    neg = - torch.sum(torch.log(F.softmax(-neg_score)))

    return pos + neg


# 主函数
def main():
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
                node_feat = torch.squeeze(node_feat, 0).T
                # node_feat = np.array(node_feat).flatten()
                print('--------------')
                print(node_feat.shape)

                feat, pos_score, neg_score = model(hg, node_feat)
                loss = compute_loss(pos_score, neg_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                train_l = loss(model(hg, node_feat))
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



if __name__ == '__main__':
    main()







