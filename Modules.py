import torch
from torch import nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        '''
        :param temperature: 用于缩放注意力权重
        :param attn_dropout: 表示注意力层的dropout概率
        '''
        super.__init__()
        self.temperature = temperature
        self.attn_dropout = attn_dropout


    def forward(self, q, k, v, mask=None):
        '''
        计算注意力权重并进行加权求和
        :param q: query
        :param k: key
        :param v: value
        :param mask: Masked Multi-Head Self-attention中需要使用的mask矩阵
        :return: 输出结果和注意力权重
        '''
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask == None:
            attn = mask.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



