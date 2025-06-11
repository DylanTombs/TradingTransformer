import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize weights: (d_model, d_model)
        self.w_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.w_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # Bias terms
        self.b_q = np.zeros((d_model,))
        self.b_k = np.zeros((d_model,))
        self.b_v = np.zeros((d_model,))
        self.b_o = np.zeros((d_model,))

    def linear(self, x, weight, bias):
        return np.matmul(x, weight) + bias  # (batch, seq_len, d_model)

    def split_heads(self, x):
        # Split last dim into (num_heads, d_k) and transpose
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)

    def combine_heads(self, x):
        # (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, seq_len, d_model)
        """
        batch_size = q.shape[0]

        # Linear projections
        q_proj = self.linear(q, self.w_q, self.b_q)
        k_proj = self.linear(k, self.w_k, self.b_k)
        v_proj = self.linear(v, self.w_v, self.b_v)

        # Split into multiple heads
        q_heads = self.split_heads(q_proj)
        k_heads = self.split_heads(k_proj)
        v_heads = self.split_heads(v_proj)

        # Scaled Dot-Product Attention
        output_heads = []
        for i in range(self.num_heads):
            out, _ = scaled_dot_product_attention(q_heads[:, i], k_heads[:, i], v_heads[:, i], mask)
            output_heads.append(out)

        # Stack heads back together: list of (batch, seq_len, d_k) → (batch, heads, seq_len, d_k)
        output = np.stack(output_heads, axis=1)

        # Combine heads
        combined = self.combine_heads(output)

        # Final output projection
        final_output = self.linear(combined, self.w_o, self.b_o)

        return final_output
    


def softmax(x, axis=-1):
    # Stable softmax
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v shape: (batch_size, seq_len, d_k)
    mask shape (optional): (batch_size, seq_len, seq_len)
    """
    d_k = q.shape[-1]

    # Step 1: Compute raw attention scores
    scores = np.matmul(q, np.transpose(k, (0, 2, 1)))  # shape: (batch, seq_len, seq_len)

    # Step 2: Scale scores
    scores /= np.sqrt(d_k)

    # Step 3: Apply mask (optional)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax over last axis (attention weights)
    attn_weights = softmax(scores, axis=-1)

    # Step 5: Multiply by values
    output = np.matmul(attn_weights, v)  # shape: (batch, seq_len, d_v)

    return output, attn_weights
