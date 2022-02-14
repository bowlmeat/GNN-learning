# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import graph

# 设置随机种子
torch.manual_seed(0)
# 超参数
EPOCH = 40
LR = 0.02


def load_array(feat, batch_size, is_train=True):
    """构造数据迭代器"""
    feat = torch.tensor(feat)
    dataset = data.TensorDataset(*feat)   ## 这里的*：作用于可迭代对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array(graph.load_feat(), batch_size)
# print(next(iter(data_iter)))


# 主函数·
def main():
    data = DataGraph()
    hg = data.hg

    model = Model(
        in_size={'road': 6},
        hidden_size=128,
        out_size=6,
        canonical_etypes=hg.canonical_etypes
    )
    optimizer = torch.optim.Adam(model.parameters())

    # 训练
    for epoch in range(EPOCH):
        for node_feat in hg.ndata['h']:         ## 这里就需要利用迭代器进行循环训练
            net_model, pos_score, neg_score = model(pos_graph, neg_graph, node_feat)
            loss = compute_loss(pos_score, neg_score, hg.canonical_etypes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_loss(pos_score, neg_score, canonical_etypes):
    all_losses = []
    for type in canonical_etypes:
        n_edges = pos_score[type].shape[0]
        all_losses.append(
            (1 - neg_score[given_type].view(n_edges, -1) +
             pos_score[given_type].unsqueeze(1)).clamp(min=0).mean()
        )

    return torch.stack(all_losses, dim=0).mean()


if __name__ == '__main__':
    main()






