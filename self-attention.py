import torch
print("hello")
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("key_2 shape: ", keys.shape)
print("values.shape: ", value_2.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("attention score for input 2 for query 2:", attn_score_22)

attn_score_2 = query_2 @ keys.T
print("attention score for input 2 for all queries:", attn_score_2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_score_2/d_k**0.5, dim=-1)
print("attention weight for input 2:", attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print("context vector for input 2:", context_vec_2)
