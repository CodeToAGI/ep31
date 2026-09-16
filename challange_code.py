"""
EP31 Challenge Solution — Bahdanau Attention on the EP30 number-word translator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# 1. Bahdanau Attention Layer
# ────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs : (batch, src_len, hidden)
        decoder_hidden  : (batch, hidden)
        Returns:
            context     : (batch, hidden)
            alpha       : (batch, src_len)
        """
        # Project
        e_proj = self.W1(encoder_outputs)                       # (B, T, H)
        d_proj = self.W2(decoder_hidden).unsqueeze(1)           # (B, 1, H)

        # Scores → weights
        scores = self.v(torch.tanh(e_proj + d_proj)).squeeze(-1)  # (B, T)
        alpha  = F.softmax(scores, dim=1)                         # (B, T)

        # Context
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, alpha


# ────────────────────────────────────────────────
# 2. Attention Decoder (drop-in replacement for EP30 Decoder)
# ────────────────────────────────────────────────
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn       = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, token, hidden, encoder_outputs):
        """
        token            : (batch, 1)
        hidden           : (h, c) from LSTM
        encoder_outputs  : (batch, src_len, hidden)
        """
        emb = self.embed(token)                                 # (B, 1, E)
        # Use last layer’s hidden state for attention query
        h_dec = hidden[0][-1]                                   # (B, H)
        context, alpha = self.attention(encoder_outputs, h_dec)

        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1) # (B, 1, E+H)
        out, hidden = self.rnn(rnn_in, hidden)
        logits = self.fc(out.squeeze(1))                        # (B, vocab)
        return logits, hidden, alpha


# ────────────────────────────────────────────────
# 3. Quick heatmap helper (exactly what appears in the video)
# ────────────────────────────────────────────────
def plot_attention(weights, src_tokens, tgt_tokens, title="Attention Heatmap"):
    """
    weights : (tgt_len, src_len) numpy array
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.set_title(title)

    # Write numeric values
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", color="w" if weights[i, j] > 0.5 else "k")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────
# 4. Minimal usage example (plug into your EP30 training loop)
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy shapes just to show the call signature
    B, SRC_LEN, HIDDEN, VOCAB = 2, 5, 128, 20
    enc_out = torch.randn(B, SRC_LEN, HIDDEN)
    h0 = (torch.randn(1, B, HIDDEN), torch.randn(1, B, HIDDEN))
    token = torch.randint(0, VOCAB, (B, 1))

    decoder = AttentionDecoder(VOCAB, embed_size=64, hidden_size=HIDDEN)
    logits, h_new, alpha = decoder(token, h0, enc_out)

    print("logits :", logits.shape)   # (B, VOCAB)
    print("alpha  :", alpha.shape)    # (B, SRC_LEN)

    # Example heatmap (random weights for illustration)
    dummy_weights = np.random.dirichlet(np.ones(4), size=3)
    plot_attention(
        dummy_weights,
        src_tokens=["one", "two", "three", "<EOS>"],
        tgt_tokens=["1", "2", "3"],
        title="Example Attention Heatmap"
    )
