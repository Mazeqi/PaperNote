import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(num_embeddings  = vocab_size, embedding_dim  = embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings = maxlen, embedding_dim = embed_dim)
    
    # 这里默认输入的是tensor
    def forward(self, x):
        maxlen = x.shape[-1]
        positions = torch.arange(0, maxlen)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

input_ = torch.LongTensor([1,2,3,4,5,6,7,8,9]).reshape(3,3,1)
print(input_.shape)
test = TokenAndPositionEmbedding(50000, 10, 3)
a = test(input_)
print(a)
print(a.shape)