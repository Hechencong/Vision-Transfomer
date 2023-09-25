import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    '''定义一个两层前馈网络'''
    def __init__(self, d_in, d_hid, dropout=0.5):
        '''

        :param d_in: 输入张量的维度
        :param d_hid:  隐藏层的维度
        :param dropout: dropout的概率
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hid, d_in),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    '''实现自注意力机制'''
    def __init__(self, d_in, n_heads=8, d_head=64, dropout=0.5):
        '''

        :param d_in: 输入特征的维度
        :param n_headw: 注意力头的数量
        :param d_head: 注意力头的特征维度
        :param dropout: dropout概率
        '''
        super().__init__()
        inner_dim = n_heads * d_head
        project_out = not ( n_heads == 1 and d_head == d_in)

        self.n_head = n_heads
        self.scale = d_head ** -0.5

        self.norm = nn.LayerNorm(d_in)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.to_qkv = nn.Linear(d_in, inner_dim * 3, bias=False)

        # 输出变换
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_in),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        x = self.norm(x)

        # b:batch size  n:序列长度  h:注意力头数  d:特征维度
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # attn: b x h x n x n      v: b x h x n x d
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, d_in, n_layers, n_heads, d_head, mlp_dim, dropout=0.5):
        '''

        :param d_in: 输入向量的维度
        :param n_layers: Teansformer的层数
        :param n_heads: 注意力机制中的头数
        :param d_head: 注意力头的维度
        :param mlp_dim: 向MLP输出的向量的维度
        :param dropout: dropout的概率
        '''
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.Layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.Layers.append(nn.ModuleList([
                Attention(d_in=d_in, n_heads=n_heads, d_head=d_head, dropout=dropout),
                FeedForward(d_in=d_in, d_hid=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.Layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes,
                 d_in, n_layers, n_heads, mlp_dim, pool='cls', channels=3,
                 d_head=64, dropout=0.5, emb_dropout=0.5):
        '''

        :param image_size: 图像大小
        :param patch_size: patch的大小
        :param num_classes: 类别数量
        :param d_in: transformer的输入向量的维度
        :param n_layers: Transformer的层数
        :param n_heads: 注意力头数
        :param mlp_dim: MLP隐藏层维度
        :param pool: 池化类型
        :param channels: 图像的通道数
        :param d_head: 注意力头的维度
        :param dropout: dropout概率
        :param emb_dropout: 嵌入层的dropout概率
        '''
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 , 'Image dimension must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be cls or mean.'

        #将图片转换为patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_in),
            nn.LayerNorm(d_in),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_in))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_in))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(d_in, n_layers, n_heads, d_head, mlp_dim, dropout=dropout)

        self.pool = pool
        # 恒等映射层
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(d_in, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        #pos_embedding的形状为[1,N+1,d]，其中N是最大patch数量
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

