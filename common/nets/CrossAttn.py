
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_size, heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(query_dim, embed_size, bias=False)
        self.keys = nn.Linear(key_value_dim, embed_size, bias=False)
        self.values = nn.Linear(key_value_dim, embed_size, bias=False)
        # self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask):
        N = queries.shape[0]
        query_len = queries.shape[1]
        key_len = keys.shape[1]

        # Linear projections
        queries_proj = self.queries(queries)
        keys_proj = self.keys(keys)
        values_proj = self.values(values)


        # Reshape into multiple heads
        queries_proj = queries_proj.reshape(
            N, query_len, self.heads, self.head_dim)
        keys_proj = keys_proj.reshape(
            1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)
        values_proj = values_proj.reshape(
            1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)

        # Permute to bring heads dimension in front
        # Shape: [N, heads, query_len, head_dim]
        queries_proj = queries_proj.permute(0, 2, 1, 3)
        # Shape: [N, heads, key_len, head_dim]
        keys_proj = keys_proj.permute(0, 2, 1, 3)
        # Shape: [N, heads, key_len, head_dim]
        values_proj = values_proj.permute(0, 2, 1, 3)

        # Step 1: Reshape queries and keys for batched matrix multiplication
        queries_proj = queries_proj.reshape(
            N * self.heads, query_len, self.head_dim)
        keys_proj = keys_proj.reshape(N * self.heads, key_len, self.head_dim)

        # Step 2: Matrix multiplication (batch matmul)
        # Shape: [N * self.heads, query_len, key_len]
        energy = torch.matmul(queries_proj, keys_proj.transpose(-1, -2))

        # Step 3: Reshape back to original shape
        energy = energy.reshape(N, self.heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        # Apply attention weights to values
        values_proj = values_proj.reshape(
            N * self.heads, key_len, self.head_dim)
        out = torch.matmul(attention.reshape(
            N * self.heads, query_len, key_len), values_proj)
        
        # print(attention.reshape(
        #     N * self.heads, query_len, key_len))

        # Reshape to (N, query_len, heads, head_dim) and combine heads
        out = out.reshape(N, self.heads, query_len, self.head_dim).permute(
            0, 2, 1, 3).reshape(N, query_len, self.embed_size)

        # out = self.fc_out(out)
        return out


class SelfAttention(nn.Module):
    """
    Implementation of plain self attention mechanism with einsum operations
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://theaisummer.com/transformer/
    """
    def __init__(self, dim):
        """
        Args:
            dim: for NLP it is the dimension of the embedding vector
            the last dimension size that will be provided in forward(x),
            where x is a 3D tensor
        """
        super().__init__()
        # for Step 1
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        # for Step 2
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided'

        # Step 1
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Step 2
        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 3
        return torch.einsum('b i j , b j d -> b i d', attention, v)
    

class eca_block(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x
    
# # Example usage
# query_dim = 320
# key_value_dim = 768
# embed_size = 320
# heads = 8
# queries = torch.rand((64, 64, query_dim))
# keys = torch.rand((1, 50, key_value_dim))
# values = torch.rand((1, 50, key_value_dim))
# mask = None

# cross_attention_layer = MultiHeadCrossAttention(
#     query_dim, key_value_dim, embed_size, heads)
# out = cross_attention_layer(queries, keys, values, mask)
# print(out.shape)  # Should print torch.Size([64, 64, 320])


# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, query_dim, key_value_dim, embed_size, heads):
#         super(MultiHeadCrossAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads

#         assert (
#             self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"

#         self.queries = nn.Linear(query_dim, embed_size, bias=False)
#         self.keys = nn.Linear(key_value_dim, embed_size, bias=False)
#         self.values = nn.Linear(key_value_dim, embed_size, bias=False)
#         self.fc_out = nn.Linear(embed_size, embed_size)

#     def preprocess(self, keys, queries, values):
#         # iNIT
#         N = queries.shape[0]
#         query_len = queries.shape[1]
#         print(self.keys.weight.shape)

#         key_len = keys.shape[1]
#         keys_proj = self.keys(keys)
#         print(keys_proj.shape)  # 1,50,320
#         keys_proj = keys_proj.reshape(
#             1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
#         # Shape: [N, heads, head_dim,key_len]
#         keys_proj = keys_proj.permute(0, 2, 3, 1)
#         print(keys_proj.shape)  # 1,8,40,50
#         Wq = self.queries.weight
#         # Wq = Wq.reshape(1, self.embed_size, self.heads, self.head_dim)
#         Wq = Wq.reshape(1, self.heads, self.head_dim, self.embed_size)
#         print(Wq.shape)  # 320,8,40
#         Wq = Wq.permute(0, 1, 3, 2)  # Shape: [N, heads, key_len, head_dim]
#         print(Wq.shape)  # 1,8,320,40
#         self.qk = torch.matmul(Wq, keys_proj)
#         print(self.qk.shape)
#         # self.qk = self.queries(keys_proj)
#         # self.qk = self.qk.reshape(1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
#         # print(self.qk.shape)

#         # computation
#         # Shape: [N * self.heads, query_len, key_len]
#         energy = torch.matmul(queries, self.qk)
#         print(energy.shape)

#         # Step 3: Reshape back to original shape
#         energy = energy.reshape(N, self.heads, query_len, key_len)

#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
#         # until here it works

#         # Fuse V and out
#         values_proj = self.values(values)
#         print(values_proj.shape)  # 1,50,320
#         values_proj = values_proj.reshape(
#             1, key_len, self.heads, self.head_dim)
#         print(values_proj.shape)  # 1,50,8,40
#         # Shape: [N, heads, key_len, head_dim]
#         values_proj = values_proj.permute(0, 2, 1, 3)
#         print(values_proj.shape)  # 1,8,50,40
#         W_out = self.fc_out.weight.T
#         # method1
#         W_out = W_out.reshape(1, self.heads, self.head_dim, self.embed_size)
#         print(W_out.shape)  # 1,8,40,320
#         # method2
#         # W_out = W_out.reshape(1, self.embed_size, self.heads, self.head_dim)
#         # print(W_out.shape) # 1,320,8,40
#         # W_out = W_out.permute(0, 2, 3, 1)  # Shape: [N, heads, head_dim,key_len]
#         # print(W_out.shape) # 1,8,40,320
#         # multiply values_proj and W_out
#         out = torch.matmul(values_proj, W_out)
#         print(out.shape)
#         print('check above')

#         # compute
#         out = torch.matmul(attention, out)
#         print(out.shape)
#         out = torch.sum(out, dim=1)
#         out += self.fc_out.bias
#         print(out.shape)
#         return out

#     def precompute(self):
#         # iNIT
#         N = queries.shape[0]
#         query_len = queries.shape[1]
#         print(self.keys.weight.shape)

#         # Fuse Q and K
#         key_len = keys.shape[1]
#         keys_proj = self.keys(keys)
#         keys_proj = keys_proj.reshape(
#             1, key_len, self.heads, self.head_dim).repeat(1, 1, 1, 1)
#         # Shape: [N, heads, head_dim,key_len]
#         keys_proj = keys_proj.permute(0, 2, 3, 1)
#         Wq = self.queries.weight
#         Wq = Wq.reshape(1, self.heads, self.head_dim, self.embed_size)
#         Wq = Wq.permute(0, 1, 3, 2)  # Shape: [N, heads, key_len, head_dim]
#         self.qk = torch.matmul(Wq, keys_proj)

#         # Fuse V and out
#         values_proj = self.values(values)
#         values_proj = values_proj.reshape(
#             1, key_len, self.heads, self.head_dim)
#         # Shape: [N, heads, key_len, head_dim]
#         values_proj = values_proj.permute(0, 2, 1, 3)
#         W_out = self.fc_out.weight.T
#         W_out = W_out.reshape(1, self.heads, self.head_dim, self.embed_size)
#         self.Vout = torch.matmul(values_proj, W_out)

#     def fused_fwd(self, queries):
#         # init
#         N = queries.shape[0]
#         query_len = queries.shape[1]
#         key_len = keys.shape[1]

#         # computation qk
#         # Shape: [N * self.heads, query_len, key_len]
#         energy = torch.matmul(queries, self.qk)
#         energy = energy.reshape(N, self.heads, query_len, key_len)

#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         # compute attention mask
#         attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

#         # compute out
#         out = torch.matmul(attention, self.Vout)
#         out = torch.sum(out, dim=1)
#         out += self.fc_out.bias

#         return out

#     def forward(self, queries, keys, values, mask):
#         N = queries.shape[0]
#         query_len = queries.shape[1]
#         key_len = keys.shape[1]

#         # Linear projections
#         queries_proj = self.queries(queries)
#         keys_proj = self.keys(keys)
#         values_proj = self.values(values)

#         # Reshape into multiple heads
#         queries_proj = queries_proj.reshape(
#             N, query_len, self.heads, self.head_dim)
#         keys_proj = keys_proj.reshape(
#             1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)
#         values_proj = values_proj.reshape(
#             1, key_len, self.heads, self.head_dim).repeat(N, 1, 1, 1)

#         # Permute to bring heads dimension in front
#         # Shape: [N, heads, query_len, head_dim]
#         queries_proj = queries_proj.permute(0, 2, 1, 3)
#         # Shape: [N, heads, key_len, head_dim]
#         keys_proj = keys_proj.permute(0, 2, 1, 3)
#         # Shape: [N, heads, key_len, head_dim]
#         values_proj = values_proj.permute(0, 2, 1, 3)

#         # Step 1: Reshape queries and keys for batched matrix multiplication
#         queries_proj = queries_proj.reshape(
#             N * self.heads, query_len, self.head_dim)
#         keys_proj = keys_proj.reshape(N * self.heads, key_len, self.head_dim)

#         # Step 2: Matrix multiplication (batch matmul)
#         # Shape: [N * self.heads, query_len, key_len]
#         energy = torch.matmul(queries_proj, keys_proj.transpose(-1, -2))

#         # Step 3: Reshape back to original shape
#         energy = energy.reshape(N, self.heads, query_len, key_len)
#         # return energy
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
#         # return attention
#         # Apply attention weights to values
#         values_proj = values_proj.reshape(
#             N * self.heads, key_len, self.head_dim)
#         out = torch.matmul(attention.reshape(
#             N * self.heads, query_len, key_len), values_proj)

#         # Reshape to (N, query_len, heads, head_dim) and combine heads
#         out = out.reshape(N, self.heads, query_len, self.head_dim).permute(
#             0, 2, 1, 3).reshape(N, query_len, self.embed_size)

#         out = self.fc_out(out)
#         return out


if __name__ == "__main__":

    # Example usage
    query_dim = 320
    key_value_dim = 768
    embed_size = 320
    heads = 8
    queries = torch.rand((1, 4096, query_dim))
    keys = torch.rand((1, 50, key_value_dim))
    values = keys
    # values = torch.rand((1, 50, key_value_dim))
    mask = None

    cross_attention_layer = MultiHeadCrossAttention(
        query_dim, key_value_dim, embed_size, heads)
    test = cross_attention_layer.preprocess(keys, queries, values)
    out = cross_attention_layer(queries, keys, values, mask)
    print(out.shape)  # Should print torch.Size([64, 64, 320])

    cross_attention_layer.precompute()
    out = cross_attention_layer.fused_fwd(queries)
    print(out.shape)  # Should print torch.Size([64, 64, 320])

    # prompt: calculate the mse between test and out

    import torch.nn as nn

    mse_loss = nn.MSELoss()
    mse = mse_loss(test, out)
    print(mse)

    test

    out

    print(cross_attention_layer.fc_out.bias.shape)

    # prompt: how to add bias value to out

    # Add bias to the output
    out = cross_attention_layer.fc_out(out) + cross_attention_layer.fc_out.bias
