# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn

dgl.load_backend('pytorch')

import numpy as np

# from dgl.nn import GraphConv
from torch.optim import Adam

# 注意！这里还需要设置随机种子固定值
torch.manual_seed(0)
# 超参数
EPOCH = 40
LR = 0.02


# 异构图卷积层
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, hg, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in hg.canonical_etypes:
            Wh = self.weight[etype](feat_dict)     # 我要：每一种边对应的所有结点的特征 Wh = self.weight[etype](feat_dict[srctype])
            hg.nodes[srctype].data['Wh_%s' % etype] = Wh   ## 是否每个结点特征列出来  -->则需要添加循环


            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))   # 在graph.ndata['h']中存储结果

        hg.multi_updata_all(funcs, 'sum')
        return {ntype: hg.nodes[ntype].data['h'] for ntype in hg.ntypes}



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

    def forward(self, hg, h):
        h = self.conv1(hg, h)
        h = F.relu(h)
        h = self.conv2(hg, h)
        return h


# 计算相近程度
class HeteroScorePredictor(nn.Module):     ## 这一部分肯定还是存在很大的问题的
    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['s'] = h
            for etype in hg.canonical_etypes:
                hg.apply_edges(dgl.function.u_dot_v('s', 's', 'score'))
            return hg.edata['score']



# 模型
class Model(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, canonical_etypes):
        self.net = HeteroRGCN(in_size, hidden_size, out_size, canonical_etypes)
        self.pred = HeteroScorePredictor()

    def forward(self, hg, neg_g, h):
        h = self.net(hg, h)
        pos_score = self.pred(hg, h)
        neg_score = self.pred(neg_g, h)

        return self.net, pos_score, neg_score



