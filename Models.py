'''定义 transformer 模型'''
import torch
import torch.nn as nn
import numpy as np
from Layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    '''用于生成填充位置的掩码，即将输入序列张量中的填充位置标记为False，用于产生Encoder的Mask，它是一列Bool值，负责把标点mask掉。'''
    '''将seq与pad_idx进行比较，得到一个布尔类型的张量，其中True表示非填充的位置，False表示填充的位置'''
    '''并将形状(batch_size, sequence_length)变为(batch_size, 1, sequence_length)'''
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    '''用于生成后续信息的掩码，即将序列张量中当前位置之后的位置标记为False'''
    '''得到一个上三角为0、下三角和主对角线为1的张量'''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device) , diagonal = 1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    '''生成位置编码'''
    def __init__(self, d_hid, n_position):
        '''

        :param d_hid: 隐藏层维度
        :param n_position: 位置编码表的长度
        '''
        super(PositionalEncoding, self).__init__()
        #将位置编码表作为模型的固定缓冲区，它不会作为需要优化的模型参数。
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))


    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        '''
        用于生成一个形状为(n_position, d_hid)正弦位置编码表
        :param n_position: 位置编码表的长度
        :param d_hid: 隐藏层维度
        :return: 1 x n_position x d_hid 的Tensor类型的正弦位置编码
        '''

        def get_position_angle_vec(position):
            '''
            生成单个位置的角度向量
            '''
            return [position / np.power(10000, (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):

        #使用.clone().detach()创建一个副本，以确保位置编码表不会被梯度更新
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    '''编码器模型'''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        '''

        :param n_src_vocab: 输入词汇量
        :param d_word_vec: 词向量的维度
        :param n_layers: 编码器层数
        :param n_head: 注意力头数
        :param d_k: 键key的维度
        :param d_v: 值value的维度
        :param d_model: 模型的总体维度
        :param d_inner: 内部前馈神经网络的维度
        :param pad_idx: 填充标记的索引
        :param dropout: Dropout率，默认为0.1
        :param n_position: 位置编码的最大长度，默认为200
        :param scale_emb: 是否对词嵌入进行缩放，默认为False
        '''
        super().__init__()

        #用于将输入的词序列映射为词向量表示
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers) ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.scale_emb = scale_emb

    def forward(self, src_seq, src_mask, return_attns=False):

        #存储每个编码层的自注意力权重
        enc_slf_attn_list = []

        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask = src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    '''解码器模型'''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        '''

        :param n_trg_vocab: 目标语言的词汇量
        :param d_word_vec: 词向量的维度
        :param n_layers: 解码器层数
        :param n_head: 注意力头数
        :param d_k: 键key的维度
        :param d_v: 值value的维度
        :param d_model: 模型的总体维度
        :param d_inner: 内部前馈神经网络的维度
        :param pad_idx: 填充标记的索引
        :param n_position: 位置编码的最大长度，默认200
        :param dropout: Dropout的概率，默认0.1
        :param scale_emb: 是否对词嵌入进行缩放，默认为False
        '''
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers) ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        '''

        :param trg_seq: 目标序列
        :param trg_mask: 目标序列的掩码
        :param enc_output: 编码器输出
        :param src_mask: 源序列的掩码
        :param return_attns: 是否返回注意力权重
        :return:
        '''

        #存储自注意力和编码器-解码器注意力的权重
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        else:
            return dec_output,


class Transformer(nn.Module):
    '''Transformer模型'''
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        '''

        :param n_src_vocab: 源语言词汇表大小
        :param n_trg_vocab: 目标语言词汇表大小
        :param src_pad_idx: 填充源序列的特殊索引
        :param trg_pad_idx: 填充目标序列的特殊索引
        :param d_word_vec: 词向量的维度。
        :param d_model: 模型的隐藏层维度。
        :param d_inner: 前馈神经网络内部层的维度。
        :param n_layers: 编码器和解码器的层数。
        :param n_head: 多头自注意力机制的头数。
        :param d_k: 注意力机制中key的维度
        :param d_v: 注意力机制中value的维度
        :param dropout: Dropout的概率
        :param n_position: 位置编码的最大长度
        :param trg_emb_prj_weight_sharing: 是否共享目标语言词向量投影层权重
        :param emb_src_trg_weight_sharing: 是否共享源语言和目标语言词向量层的权重
        :param scale_emb_or_prj: 控制是否乘以缩放因子
        '''

        super().__init__()

        #根据论文《Attention Is All You Need》中的描述，在嵌入层和预softmax线性变换中共享相同的权重矩阵，
        # 对应的选项有三种：'emb'、'prj' 和 'none'。
        # 在满足 trg_emb_prj_weight_sharing = Ture 的情况下
        # 如果选择了 'scale_emb_or_prj' 为 'emb'，则使用嵌入层的输出进行缩放；
        # 如果选择了 'scale_emb_or_prj' 为 'prj'，则使用线性投影输出进行缩放。
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False

        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, n_layers=n_layers,
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            pad_idx=src_pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec, n_layers=n_layers,
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            pad_idx=trg_pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb)

        #定义了一个线性变换层trg_word_prj , 该层将Decoder输出的向量映射为目标语言词汇的分布
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        #生成源序列的掩码
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        #生成目标序列的掩码
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        #通过一个线性层self.trg_word_prj将解码器的输出映射到目标词汇空间，得到一个logit张量
        # [batch_size x src_vocab_size x trg_vocab_size]
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))

