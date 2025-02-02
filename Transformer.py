import torch
import torch.nn as nn
import numpy as np

# ===========================
# 正弦位置编码（Sinusoidal Position Encoding）
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = torch.tensor(pe, dtype=torch.float32).unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ===========================
# 多头自注意力（Multi-Head Self-Attention）
# ===========================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.W_o(attn_output)

# ===========================
# Transformer 编码器（Encoder）
# ===========================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# ===========================
# Transformer 解码器（Decoder）
# ===========================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output = self.cross_attn(x)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

# ===========================
# Transformer 整体模型
# ===========================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, hidden_dim, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x, x)
        return x

# ===========================
# 示例运行 Transformer 模型
# ===========================
vocab_size = 30522  # 词汇表大小
max_len = 512  # 最大序列长度
d_model = 512  # 词向量维度
num_heads = 8  # 多头注意力
num_layers = 6  # Transformer 层数
hidden_dim = 2048  # FFN 隐藏层大小

# 创建 Transformer 模型
model = Transformer(vocab_size, d_model, num_heads, num_layers, hidden_dim, max_len)

# 生成随机 Token ID（batch_size=2, seq_len=10）
tokens = torch.randint(0, vocab_size, (2, 10))
output = model(tokens)
print("Transformer Output Shape:", output.shape)  # (batch_size, seq_len, d_model)
