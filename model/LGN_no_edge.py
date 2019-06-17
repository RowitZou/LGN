# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.crf import CRF


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
    def __init__(self, hid_h, dropout=0.2):
        super(Nodes_Cell, self).__init__()

        self.Wix = nn.Linear(hid_h*4, hid_h)
        #self.Wig = nn.Linear(hid_h*4, hid_h)
        self.Wi2 = nn.Linear(hid_h*4, hid_h)
        self.Wf = nn.Linear(hid_h*4, hid_h)
        self.Wcx = nn.Linear(hid_h*4, hid_h)
        #self.Wcg = nn.Linear(hid_h, hid_h)

        self.drop = nn.Dropout(dropout)

    def forward(self, h, h2, x, glo):

        x = self.drop(x)
        glo = self.drop(glo)

        cat_all = torch.cat([h, h2, x, glo], -1)
        #cat_x = torch.cat([h, h2, x], -1)
        #cat_g = torch.cat([glo], -1)

        ix = torch.sigmoid(self.Wix(cat_all))
        #ig = torch.sigmoid(self.Wig(cat_all))
        i2 = torch.sigmoid(self.Wi2(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        cx = torch.tanh(self.Wcx(cat_all))
        #cg = torch.tanh(self.Wcg(cat_g))

        alpha = F.softmax(torch.cat([ix.unsqueeze(1), i2.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * cx) + (alpha[:, 1] * h2) + (alpha[:, 2] * h)

        return output

class GLobal_Cell(nn.Module):
    def __init__(self, hid_h, dropout=0.2):
        super(GLobal_Cell, self).__init__()

        self.Wi = nn.Linear(hid_h*2, hid_h)
        self.Wf = nn.Linear(hid_h*2, hid_h)
        self.Wc = nn.Linear(hid_h*2, hid_h)

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


class Graph(nn.Module):
    def __init__(self, data):
        super(Graph, self).__init__()

        self.gpu = data.HP_gpu
        self.word_alphabet = data.word_alphabet
        self.word_emb_dim = data.word_emb_dim
        self.gaz_emb_dim = data.gaz_emb_dim
        self.hidden_dim = 50
        self.num_head = 10  # 5 10 20
        self.head_dim = 20  # 10 20
        self.tf_dropout_rate = 0.1
        self.iters = 4
        self.bmes_dim = 10
        self.length_dim = 10
        self.max_gaz_length = 5
        self.emb_dropout_rate = 0.5
        self.cell_dropout_rate = 0.2

        # word embedding
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        assert data.pretrain_word_embedding is not None
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))

        # lstm
        self.emb_rnn_f = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.emb_rnn_b = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        # length embedding
        self.length_embedding = nn.Embedding(self.max_gaz_length, self.length_dim)

        self.dropout = nn.Dropout(self.emb_dropout_rate)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.edge2node_f = nn.ModuleList(
            [MultiHeadAtt(self.hidden_dim, self.hidden_dim+self.length_dim,
                          nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.glo_att_f_node = nn.ModuleList(
            [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.glo_att_f_edge = nn.ModuleList(
            [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.node_rnn_f = Nodes_Cell(self.hidden_dim, dropout=self.cell_dropout_rate)
        self.glo_rnn_f = GLobal_Cell(self.hidden_dim, dropout=self.cell_dropout_rate)

        self.edge2node_b = nn.ModuleList(
            [MultiHeadAtt(self.hidden_dim, self.hidden_dim+self.length_dim,
                          nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.glo_att_b_node = nn.ModuleList(
            [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.glo_att_b_edge = nn.ModuleList(
            [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
             for _ in range(self.iters)])

        self.node_rnn_b = Nodes_Cell(self.hidden_dim, self.cell_dropout_rate)
        self.glo_rnn_b = GLobal_Cell(self.hidden_dim, self.cell_dropout_rate)

        self.layer_att_W = nn.Linear(self.hidden_dim * 2, 1)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.gpu:
            self.word_embedding = self.word_embedding.cuda()
            self.length_embedding = self.length_embedding.cuda()
            self.norm = self.norm.cuda()
            self.node_rnn_f = self.node_rnn_f.cuda()
            self.edge2node_f = self.edge2node_f.cuda()
            self.glo_rnn_f = self.glo_rnn_f.cuda()
            self.glo_att_f_node = self.glo_att_f_node.cuda()
            self.glo_att_f_edge = self.glo_att_f_edge.cuda()
            self.edge2node_b = self.edge2node_b.cuda()
            self.node_rnn_b = self.node_rnn_b.cuda()
            self.glo_rnn_b = self.glo_rnn_b.cuda()
            self.glo_att_b_node = self.glo_att_b_node.cuda()
            self.glo_att_b_edge = self.glo_att_b_edge.cuda()
            self.emb_rnn_f = self.emb_rnn_f.cuda()
            self.emb_rnn_b = self.emb_rnn_b.cuda()
            self.layer_att_W = self.layer_att_W.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    def obtain_gaz_relation(self, batch_size, seq_len, gaz_list):

        assert batch_size == 1

        for sen in range(batch_size):
            sen_nodes_mask = torch.zeros([1, seq_len]).byte()
            sen_gazs_length = torch.zeros([1, self.length_dim])
            sen_gazs_mask_f = torch.zeros([1, seq_len]).byte()
            sen_gazs_mask_b = torch.zeros([1, seq_len]).byte()
            if self.cuda:
                sen_nodes_mask = sen_nodes_mask.cuda()
                sen_gazs_length = sen_gazs_length.cuda()
                sen_gazs_mask_f = sen_gazs_mask_f.cuda()
                sen_gazs_mask_b = sen_gazs_mask_b.cuda()

            for w in range(seq_len):
                if w < len(gaz_list[sen]) and gaz_list[sen][w]:
                    for gaz, gaz_len in zip(gaz_list[sen][w][0], gaz_list[sen][w][1]):

                        if gaz_len <= self.max_gaz_length:
                            gaz_length_index = torch.tensor(gaz_len-1, device=sen_gazs_length.device)
                        else:
                            gaz_length_index = torch.tensor(self.max_gaz_length-1, device=sen_gazs_length.device)
                        gaz_length = self.length_embedding(gaz_length_index)
                        sen_gazs_length = torch.cat([sen_gazs_length, gaz_length[None, :]], 0)

                        # mask: 需要mask的地方置为1, batch_size * gaz_num * seq_len
                        nodes_mask = torch.ones([1, seq_len]).byte()
                        gazs_mask_f = torch.ones([1, seq_len]).byte()
                        gazs_mask_b = torch.ones([1, seq_len]).byte()
                        if self.cuda:
                            nodes_mask = nodes_mask.cuda()
                            gazs_mask_f = gazs_mask_f.cuda()
                            gazs_mask_b = gazs_mask_b.cuda()

                        gazs_mask_f[0, w + gaz_len - 1] = 0
                        sen_gazs_mask_f = torch.cat([sen_gazs_mask_f, gazs_mask_f], 0)

                        gazs_mask_b[0, w] = 0
                        sen_gazs_mask_b = torch.cat([sen_gazs_mask_b, gazs_mask_b], 0)

                        sen_nodes_mask = torch.cat([sen_nodes_mask, nodes_mask], 0)

        batch_nodes_mask = sen_nodes_mask.unsqueeze(0)
        batch_gazs_mask_f = sen_gazs_mask_f.unsqueeze(0)
        batch_gazs_mask_b = sen_gazs_mask_b.unsqueeze(0)
        batch_gazs_length = sen_gazs_length.unsqueeze(0)
        return batch_nodes_mask, batch_gazs_mask_f, batch_gazs_mask_b, batch_gazs_length

    def get_tags(self, gaz_list, word_inputs, mask):

        #mask = 1 - mask
        node_embeds = self.word_embedding(word_inputs)  # batch_size, max_seq_len, embedding
        B, L, H = node_embeds.size()
        gaz_match = []

        nodes_mask, gazs_mask_f, gazs_mask_b, gazs_length = self.obtain_gaz_relation(B, L, gaz_list)
        _, N, _ = gazs_mask_f.size()

        node_embeds = self.dropout(node_embeds)

        nodes_f, _ = self.emb_rnn_f(node_embeds)

        glo_f = node_embeds.mean(1, keepdim=True)
        nodes_f_cat = nodes_f[:, None, :, :]
        glo_f_cat = glo_f[:, None, :, :]

        for i in range(self.iters):

            nodes_begin_f = torch.sum(nodes_f[:, None, :, :] * (1 - gazs_mask_b)[:, :, :, None].float(), 2)
            nodes_begin_f = torch.cat([torch.zeros([B, 1, H], device=nodes_f.device), nodes_begin_f[:, 1:N, :]], 1)
            nodes_att_f = self.edge2node_f[i](nodes_f, torch.cat([nodes_begin_f, gazs_length], -1).unsqueeze(2), gazs_mask_f)

            glo_att_f = self.glo_att_f_node[i](glo_f, nodes_f)

            nodes_f_r = torch.cat([torch.zeros([B, 1, self.hidden_dim], device=nodes_f.device), nodes_f[:, 0:(L-1), :]], 1)
            nodes_f = self.node_rnn_f(nodes_f, nodes_f_r, nodes_att_f, glo_att_f.expand(B, L, H))
            nodes_f_cat = torch.cat([nodes_f_cat, nodes_f[:, None, :, :]], 1)
            nodes_f = self.norm(torch.sum(nodes_f_cat, 1))

            glo_f = self.glo_rnn_f(glo_f, glo_att_f)
            glo_f_cat = torch.cat([glo_f_cat, glo_f[:, None, :, :]], 1)
            glo_f = self.norm(torch.sum(glo_f_cat, 1))

        nodes_b, _ = self.emb_rnn_b(torch.flip(node_embeds, [1]))
        nodes_b = torch.flip(nodes_b, [1])

        glo_b = node_embeds.mean(1, keepdim=True)
        nodes_b_cat = nodes_b[:, None, :, :]
        glo_b_cat = glo_b[:, None, :, :]

        for i in range(self.iters):

            nodes_begin_b = torch.sum(nodes_b[:, None, :, :] * (1 - gazs_mask_f)[:, :, :, None].float(), 2)
            nodes_begin_b = torch.cat([torch.zeros([B, 1, H], device=nodes_b.device), nodes_begin_b[:, 1:N, :]], 1)
            nodes_att_b = self.edge2node_b[i](nodes_b,
                                              torch.cat([nodes_begin_b, gazs_length], -1).unsqueeze(2), gazs_mask_b)

            glo_att_b = self.glo_att_b_node[i](glo_b, nodes_b)

            nodes_b_r = torch.cat([nodes_b[:, 1:L, :], torch.zeros([B, 1, self.hidden_dim], device=nodes_b.device)], 1)
            nodes_b = self.node_rnn_b(nodes_b, nodes_b_r, nodes_att_b, glo_att_b.expand(B, L, H))
            nodes_b_cat = torch.cat([nodes_b_cat, nodes_b[:, None, :, :]], 1)
            nodes_b = self.norm(torch.sum(nodes_b_cat, 1))

            glo_b = self.glo_rnn_b(glo_b, glo_att_b)
            glo_b_cat = torch.cat([glo_b_cat, glo_b[:, None, :, :]], 1)
            glo_b = self.norm(torch.sum(glo_b_cat, 1))

        nodes_cat = torch.cat([nodes_f_cat, nodes_b_cat], -1)
        layer_att = torch.sigmoid(self.layer_att_W(nodes_cat))
        layer_alpha = F.softmax(layer_att, 1)
        nodes = torch.sum(layer_alpha * nodes_cat, 1)

        tags = self.hidden2tag(nodes)

        return tags, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, word_seq_lengths, mask, batch_label):

        tags, _ = self.get_tags(gaz_list, word_inputs, mask)
        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq  # (batch_size,)  ,(b,seqlen?)

    def forward(self, gaz_list, word_inputs, word_seq_lengths, mask):
        tags, gaz_match = self.get_tags(gaz_list, word_inputs, mask)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)
        return tag_seq, gaz_match
