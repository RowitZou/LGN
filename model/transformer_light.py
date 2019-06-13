# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.crf import CRF
#from model.layers import MultiHeadAttention, PositionwiseFeedForward

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Vertices_Cell(nn.Module):
    def __init__(self, hid_1, hid_2):
        super(Vertices_Cell, self).__init__()

        self.i_e = nn.Linear(hid_1, hid_2)
        self.i_h = nn.Linear(hid_2, hid_2)

        self.f_e = nn.Linear(hid_1, hid_2)
        self.f_h = nn.Linear(hid_2, hid_2)

        self.c_e = nn.Linear(hid_1, hid_2)
        self.c_h = nn.Linear(hid_2, hid_2)

        nn.init.xavier_normal_(self.i_e.weight)
        nn.init.xavier_normal_(self.i_h.weight)
        nn.init.xavier_normal_(self.f_e.weight)
        nn.init.xavier_normal_(self.f_h.weight)
        nn.init.xavier_normal_(self.c_e.weight)
        nn.init.xavier_normal_(self.c_h.weight)


    def forward(self, input_1, input_2, vertices):

        i = torch.sigmoid(self.i_e(input_1) + self.i_h(input_2))
        f = torch.sigmoid(self.f_e(input_1) + self.f_h(input_2))
        c_new = torch.tanh(self.c_e(input_1) + self.c_h(input_2))

        alpha = F.softmax(torch.cat([i.unsqueeze(1), f.unsqueeze(1)], dim=1), dim=1)
        new_vertices = (alpha[:, 0] * c_new) + (alpha[:, 1] * vertices)

        return new_vertices

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))  #(head*b,len,h)*(head*b,h,len) = (head*b,len,len)  attn<i,j>=len的每个对leni的权重，也就是每个字对第i个字的权重
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(F.relu(self.fc(output)))
        output = self.layer_norm(output + residual)
        return output, attn


class Graph(nn.Module):
    def __init__(self, data):
        super(Graph, self).__init__()
        self.data = data
        self.gpu = data.HP_gpu
        self.hidden_dim = 50
        self.num_layer = data.HP_num_layer
        self.gaz_alphabet = data.gaz_alphabet
        self.word_alphabet = data.word_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.bmes_emb_dim = 25
        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        self.bmes_embedding = nn.Embedding(4, self.bmes_emb_dim)

        assert data.pretrain_gaz_embedding is not None
        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])
        self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))

        assert data.pretrain_word_embedding is not None
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))

        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(data.posi_alphabet_size, self.word_emb_dim, padding_idx=0), freeze=True)

        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.ModuleList([nn.LSTM(self.word_emb_dim, self.hidden_dim, batch_first=True)
                                  for _ in range(self.num_layer)]).cuda()

        self.node_att = nn.ModuleList([MultiHeadAttention(4, self.hidden_dim, self.hidden_dim, self.hidden_dim)
                                       for _ in range(self.num_layer)])
        #self.node_pooling = nn.ModuleList([PositionwiseFeedForward(self.hidden_dim, self.hidden_dim * 2, self.hidden_dim)
        #                                   for _ in range(self.num_layer)])
        #self.gaz_pooling = nn.ModuleList([MultiHeadAttention(1, self.hidden_dim, self.hidden_dim * 2, self.hidden_dim, self.hidden_dim)
        #                                   for _ in range(self.num_layer)])

        self.glo_att = nn.ModuleList([MultiHeadAttention(4, self.hidden_dim, self.hidden_dim, self.hidden_dim)
                                      for _ in range(self.num_layer)])

        #self.node_cell = Vertices_Cell(self.hidden_dim, self.hidden_dim)

        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.position_embedding = self.position_embedding.cuda()
            self.bmes_embedding = self.bmes_embedding.cuda()
            self.node_att = self.node_att.cuda()
            #self.node_pooling = self.node_pooling.cuda()
            #self.gaz_pooling = self.gaz_pooling.cuda()
            self.glo_att = self.glo_att.cuda()
            #self.node_cell = self.node_cell.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    """
    def obtain_gaz_relation(self, batch_size, seq_len, gaz_list):

        assert batch_size == 1
        batch_gaz_embed = torch.tensor([])
        batch_nodes_mask = torch.tensor([], dtype=torch.uint8)
        batch_bmes_embed = torch.tensor([])
        batch_gazs_mask = torch.tensor([])
        if self.cuda:
            batch_gaz_embed = batch_gaz_embed.cuda()
            batch_nodes_mask = batch_nodes_mask.cuda()
            batch_bmes_embed = batch_bmes_embed.cuda()
            batch_gazs_mask = batch_gazs_mask.cuda()

        for sen in range(batch_size):
            sen_gaz_embed = torch.zeros([0, self.gaz_emb_dim])
            sen_nodes_mask = torch.zeros([0, seq_len], dtype=torch.uint8)
            sen_bmes_embed = torch.zeros([0, seq_len, self.bmes_emb_dim])
            sen_gazs_mask = torch.zeros([0, seq_len], dtype=torch.uint8)
            if self.cuda:
                sen_gaz_embed = sen_gaz_embed.cuda()
                sen_nodes_mask = sen_nodes_mask.cuda()
                sen_bmes_embed = sen_bmes_embed.cuda()
                sen_gazs_mask = sen_gazs_mask.cuda()

            for w in range(seq_len):
                if w < len(gaz_list[sen]) and gaz_list[sen][w]:
                    for gaz, gaz_len in zip(gaz_list[sen][w][0], gaz_list[sen][w][1]):

                        gaz_index = torch.tensor(gaz).cuda() if self.cuda else torch.tensor(gaz)
                        gaz_embedding = self.gaz_embedding(gaz_index)
                        sen_gaz_embed = torch.cat([sen_gaz_embed, gaz_embedding.unsqueeze(0)], dim=0)

                        # mask: 需要mask的地方置为1, batch_size * gaz_num * seq_len
                        nodes_mask = torch.ones([1, seq_len], dtype=torch.uint8)
                        bmes_embed = torch.zeros([1, seq_len, self.bmes_emb_dim])

                        gazs_mask = torch.ones([1, seq_len], dtype=torch.uint8)
                        gazs_mask[0, w + gaz_len - 1] = 0
                        sen_gazs_mask = torch.cat([sen_gazs_mask, gazs_mask.unsqueeze(0)], dim=0)

                        if self.cuda:
                            nodes_mask = nodes_mask.cuda()
                            bmes_embed = bmes_embed.cuda()

                        for index in range(gaz_len):
                            nodes_mask[0, w + index] = 0
                            if gaz_len == 1:
                                bmes_index = torch.tensor(3).cuda() if self.cuda else torch.tensor(3)   # S
                            elif index == 0:
                                bmes_index = torch.tensor(0).cuda() if self.cuda else torch.tensor(0)   # B
                            elif index == gaz_len - 1:
                                bmes_index = torch.tensor(2).cuda() if self.cuda else torch.tensor(2)   # E
                            else:
                                bmes_index = torch.tensor(1).cuda() if self.cuda else torch.tensor(1)   # M
                            bmes_embed[0, w + index, :] = self.bmes_embedding(bmes_index)

                        sen_nodes_mask = torch.cat([sen_nodes_mask, nodes_mask], dim=0)
                        sen_bmes_embed = torch.cat([sen_bmes_embed, bmes_embed], dim=0)

            sen_gazs_unk_mask = torch.ones([1, seq_len], dtype=torch.uint8)
            sen_gazs_unk_mask[0, (1-sen_gazs_mask).sum(dim=0) < 0.5] = 0.
            sen_gazs_mask = torch.cat([sen_gazs_unk_mask, sen_gazs_mask], dim=0)

            if sen_gaz_embed.size(0) != 0:
                batch_gaz_embed = sen_gaz_embed.unsqueeze(0)    # 只有在batch_size=1时可以这么做
                batch_nodes_mask = sen_nodes_mask.unsqueeze(0)
                batch_bmes_embed = sen_bmes_embed.unsqueeze(0)
                batch_gazs_mask = sen_gazs_mask.unsqueeze(0)

        return batch_gaz_embed, batch_nodes_mask, batch_bmes_embed, batch_gazs_mask
    """

    def obtain_gaz_relation(self, batch_size, seq_len, gaz_list, mask):

        assert batch_size == 1
        adjoin_index = torch.tensor(0).cuda() if self.cuda else torch.tensor(0)
        adjoin_emb = self.gaz_embedding(adjoin_index)

        gazs_send_embeds = torch.zeros(batch_size, seq_len+1, seq_len, self.gaz_emb_dim)
        gazs_rec_embeds = torch.zeros(batch_size, seq_len+1, seq_len, self.gaz_emb_dim)
        rel_send_gaz = torch.ones([batch_size, seq_len+1, seq_len], dtype=torch.uint8)
        rel_rec_gaz = torch.ones([batch_size, seq_len+1, seq_len], dtype=torch.uint8)

        if self.cuda:
            gazs_send_embeds = gazs_send_embeds.cuda()
            gazs_rec_embeds = gazs_rec_embeds.cuda()
            rel_send_gaz = rel_send_gaz.cuda()
            rel_rec_gaz = rel_rec_gaz.cuda()

        for sen in range(batch_size):
            for w in range(seq_len):
                if not mask[sen][w]:
                    break
                if w < len(gaz_list[sen]):
                    gazs_send_embeds[sen][w][w] = adjoin_emb
                    gazs_rec_embeds[sen][w + 1][w] = adjoin_emb
                    rel_send_gaz[sen][w][w] = 0
                    rel_rec_gaz[sen][w + 1][w] = 0

        for sen in range(batch_size):
            for w in range(seq_len):
                if not mask[sen][w]:
                    break
                if w < len(gaz_list[sen]) and gaz_list[sen][w]:
                    for gaz, gaz_len in zip(gaz_list[sen][w][0], gaz_list[sen][w][1]):
                        gaz_index = torch.tensor(gaz).cuda() if self.cuda else torch.tensor(gaz)
                        gaz_embedding = self.gaz_embedding(gaz_index)

                        gazs_send_embeds[sen][w + 1][w + gaz_len - 1] = gaz_embedding
                        gazs_rec_embeds[sen][w + gaz_len - 1][w] = gaz_embedding
                        rel_send_gaz[sen][w + 1][w + gaz_len - 1] = 0
                        rel_rec_gaz[sen][w + gaz_len - 1][w] = 0

        return gazs_send_embeds, gazs_rec_embeds, rel_send_gaz, rel_rec_gaz

    def get_tags(self, gaz_list, word_inputs, mask):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        word_embs = self.word_embedding(word_inputs)  # batch_size, max_seq_len, embedding
        gaz_match = []

        # position embedding
        posi_inputs = torch.zeros(batch_size, seq_len).long()
        for batch in range(batch_size):
            posi_temp = torch.LongTensor([i + 1 for i in range(seq_len) if mask[batch][i]])
            posi_inputs[batch, 0:posi_temp.size(0)] = posi_temp
        if self.gpu:
            posi_inputs = posi_inputs.cuda()
        position_embs = self.position_embedding(posi_inputs)

        # 节点的初始表示, batch_size * seq_len * emb_size
        raw_nodes = self.dropout(word_embs + position_embs)

        # gaz的初始表示，batch_size * seq_len * seq_len * emb_size
        #raw_send_embed, raw_rec_embed, rel_send_gaz, rel_rec_gaz = \
        #    self.obtain_gaz_relation(batch_size, seq_len, gaz_list, mask)

        #gazs_send = self.dropout(raw_send_embed)
        #gazs_rec = self.dropout(raw_rec_embed)
        nodes_send = raw_nodes
        #nodes_rec = raw_nodes
        glo = torch.mean(nodes_send, dim=1)

        for layer in range(self.num_layer):

            layer_index = layer # layer_index = 0 表示只有一套参数

            #  SEND
            nodes_att_list = []
            #gazs_pooling_list = []

            # 前后padding的 nodes
            padding_embed = torch.zeros([batch_size, 1, self.hidden_dim]).cuda() if self.cuda else \
                torch.zeros([batch_size, 1, self.hidden_dim])
            nodes_padding = torch.cat([padding_embed, nodes_send, padding_embed], dim=1)
            #gazs_padding = torch.cat([nodes_padding.unsqueeze(2).expand(batch_size, seq_len+1, seq_len, self.hidden_dim),
            #                          gazs_send], dim=-1)

            for i in range(seq_len):

                node_att_cell = torch.cat([nodes_padding[:, i, :].unsqueeze(1),
                                           nodes_padding[:, (i+1), :].unsqueeze(1),
                                           nodes_padding[:, (i+2), :].unsqueeze(1),
                                           raw_nodes[:, i, :].unsqueeze(1),
                                           glo.unsqueeze(1),
                                           ], dim=1)

                node_att, _att = self.node_att[layer_index](nodes_send[:, i:(i+1), :], node_att_cell, node_att_cell)
                nodes_att_list.append(node_att)

                #gaz_pooling_cell = gazs_padding[:, :, i, :]
                #gaz_pooling_mask = rel_send_gaz[:, :, i].unsqueeze(1)
                #gaz_pooling, _att = self.gaz_pooling[layer_index](nodes_send[:, i:(i+1), :], gaz_pooling_cell, gaz_pooling_cell, gaz_pooling_mask)
                #gazs_pooling_list.append(gaz_pooling)

            nodes_send = torch.cat(nodes_att_list, dim=1)
            #nodes_send = self.node_pooling[layer_index](nodes_att)
            #gazs_pooling = torch.cat(gazs_pooling_list, dim=1)

            #nodes_send = self.node_cell(nodes_att, gazs_pooling, nodes_send)

            #nodes_send = self.dropout(nodes_att)

            glo_cell = torch.cat([glo.unsqueeze(1),
                                  nodes_send], dim=1)

            glo, _att = self.glo_att[layer_index](glo.unsqueeze(1), glo_cell, glo_cell)
            glo = glo.squeeze(1)
            #nodes_send, _ = self.node_att[layer](nodes_send, nodes_send, nodes_send)
            #nodes_send = self.node_pooling[layer](nodes_send)
            #nodes_send, _ = self.rnn[layer](nodes_send)

        tags = self.hidden2tag(nodes_send)  # (b,m,t)

        return tags, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, word_seq_lengths, mask, batch_label):

        tags, _ = self.get_tags(gaz_list, word_inputs, mask)
        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq  # (batch_size,)  ,(b,seqlen?)

    def forward(self, gaz_list, word_inputs, word_seq_lengths, mask):
        tags, gaz_match = self.get_tags(gaz_list, word_inputs, mask)
        # tags_ = tags.transpose(0,1).contiguous()
        # mask_ = mask.transpose(0,1).contiguous()
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)
        # tag_seq = self.crf_.decode(tags_, mask=mask_)
        # tag_seq = self.crf_.decode(tags, mask=mask)
        return tag_seq, gaz_match
