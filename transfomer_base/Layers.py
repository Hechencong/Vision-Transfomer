'''定义 编码器层和解码器层'''
import torch.nn as nn
import torch
from SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    '''编码器层的定义 由两层组成'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout = 0.1):
        '''

        :param d_model: 模型的维度或特征数
        :param d_inner: 前馈神经网络中隐藏层的维度
        :param n_head: 多头注意力机制的头数
        :param d_k: 键的维度
        :param d_v: 值的维度
        :param dropout: dropout概率
        '''
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask = None):
        '''

        :param enc_input: 输入张量
        :param self_attn_mask: 自注意力掩码
        :return: 编码器输出张量 和 自注意力权重张量
        '''

        enc_output , enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask = slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    '''解码器层的定义 由三层组成'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        #实现自注意力机制
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #实现编码-解码注意力机制
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output,
                slf_attn_mask = None, dec_enc_attn_mask = None):
        '''

        :param dec_input:
        :param enc_output:
        :param slf_attn_mask:
        :param dec_enc_attn_mask:
        :return: 解码器输出张量 、 自注意力权重张量 、 编码-解码注意力权重张量
        '''
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask = slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask = dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

