'''encoder和decoder需要的子层'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout = 0.1):
        '''

        :param n_head:多头注意力机制中并行进行的注意力头的个数
        :param d_model:输入向量的维度
        :param d_k:键（key）的维度
        :param d_v:值（value）的维度
        :param dropout:Dropout概率
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #将输入映射到多个查询空间、键空间、值空间
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        #将多头注意力的输出维度变换回 d_model，以便与原始输入进行融合。
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        #计算注意力权重
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        '''

        :param q:
        :param k:
        :param v:
        :param mask:
        :return: 输出张量 q 和注意力权重张量 attn
        '''

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        #sz_b（batch size）、len_q（查询序列长度）、len_k（键序列长度）、len_v（值序列长度)。
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        #通过线性变换层后张量的形状为 sz_b  x lq x (n*dv)
        #调整后形状为 sz_b x len_q x n_head x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        #交换维度1和维度2，以便后续的注意力计算能够正确进行矩阵乘法
        #形状为 sz_b x n_head x len_q x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        #对张量进行转置操作，将维度1和维度2交换回原始顺序，并使用 contiguous 方法使张量在内存中连续存储。
        #使用 view 方法将多个头的结果进行拼接，然后在通过fc层，得到形状为 (batch_size, seq_len, d_model) 的张量
        q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
        q.self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    '''实现一个两层前馈神经网络'''

    def __init__(self, d_in, d_hid, dropout = 0.1):
        '''

        :param d_in:输入向量的维度
        :param d_hid:隐藏层的维度
        :param dropout:Dropout概率
        '''
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


