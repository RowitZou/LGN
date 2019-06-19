# -*- coding: utf-8 -*-
# @Author: Yicheng Zou
# @Last Modified by:   Yicheng Zou,     Contact: yczou18@fudan.edu.cn


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAtt(nn.Module):
    def __init__(self, nhid, keyhid, nhead=10, head_dim=10, dropout=0.1, if_g=False):
        super(MultiHeadAtt, self).__init__()

        if if_g:
            self.WQ = nn.Conv2d(nhid * 3, nhead * head_dim, 1)
        else:
            self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(nhid)

        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def forward(self, query_h, value, mask, query_g=None):

        if not (query_g is None):
            query = torch.cat([query_h, query_g], -1)
        else:
            query = query_h
        query = query.permute(0, 2, 1)[:, :, :, None]
        value = value.permute(0, 3, 1, 2)

        residual = query_h
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim

        B, QL, H = query_h.shape

        _, _, VL, VD = value.shape  # VD = 1 or VD = QL

        assert VD == 1 or VD == QL
        # q: (B, H, QL, 1)
        # v: (B, H, VL, VD)
        q, k, v = self.WQ(query), self.WK(value), self.WV(value)

        q = q.view(B, nhead, head_dim, 1, QL)
        k = k.view(B, nhead, head_dim, VL, VD)
        v = v.view(B, nhead, head_dim, VL, VD)

        alpha = (q * k).sum(2, keepdim=True) / np.sqrt(head_dim)
        alpha = alpha.masked_fill(mask[:, None, None, :, :], -np.inf)
        alpha = self.drop(F.softmax(alpha, 3))
        att = (alpha * v).sum(3).view(B, nhead * head_dim, QL, 1)

        output = F.leaky_relu(self.WO(att)).permute(0, 2, 3, 1).view(B, QL, H)
        output = self.norm(output + residual)

        return output


class GloAtt(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(GloAtt, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(nhid)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim
        B, L, H = y.shape

        x = x.permute(0, 2, 1)[:, :, :, None]
        y = y.permute(0, 2, 1)[:, :, :, None]

        residual = x
        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h

        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        output = F.leaky_relu(self.WO(att)) + residual
        output = self.norm(output.permute(0, 2, 3, 1)).view(B, 1, H)

        return output


class Nodes_Cell(nn.Module):
    def __init__(self, input_h, hid_h, use_global=True, dropout=0.2):
        super(Nodes_Cell, self).__init__()

        self.use_global = use_global

        self.Wix = nn.Linear(input_h, hid_h)
        self.Wi2 = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wcx = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)

    def forward(self, h, h2, x, glo=None):

        x = self.drop(x)

        if self.use_global:
            glo = self.drop(glo)
            cat_all = torch.cat([h, h2, x, glo], -1)
        else:
            cat_all = torch.cat([h, h2, x], -1)

        ix = torch.sigmoid(self.Wix(cat_all))
        i2 = torch.sigmoid(self.Wi2(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        cx = torch.tanh(self.Wcx(cat_all))

        alpha = F.softmax(torch.cat([ix.unsqueeze(1), i2.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * cx) + (alpha[:, 1] * h2) + (alpha[:, 2] * h)

        return output


class Edges_Cell(nn.Module):
    def __init__(self, input_h, hid_h, use_global=True, dropout=0.2):
        super(Edges_Cell, self).__init__()

        self.use_global = use_global

        self.Wi = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wc = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)

    def forward(self, h, x, glo=None):

        x = self.drop(x)

        if self.use_global:
            glo = self.drop(glo)
            cat_all = torch.cat([h, x, glo], -1)
        else:
            cat_all = torch.cat([h, x], -1)

        i = torch.sigmoid(self.Wi(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        c = torch.tanh(self.Wc(cat_all))

        alpha = F.softmax(torch.cat([i.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * c) + (alpha[:, 1] * h)

        return output


class Global_Cell(nn.Module):
    def __init__(self, input_h, hid_h, dropout=0.2):
        super(Global_Cell, self).__init__()

        self.Wi = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wc = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)

    def forward(self, h, x):

        x = self.drop(x)

        cat_all = torch.cat([h, x], -1)
        i = torch.sigmoid(self.Wi(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        c = torch.tanh(self.Wc(cat_all))

        alpha = F.softmax(torch.cat([i.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * c) + (alpha[:, 1] * h)

        return output
