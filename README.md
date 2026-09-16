# EP31 — Attention Mechanism Explained

**Deep Learning Series · Episode 31 · Module 7**  
The key idea that broke the context-vector bottleneck and led directly to Transformers.

## What you’ll learn
- Why the fixed context vector in EP30 fails on long sentences
- Attention intuition (the decoder “asks a question” of every encoder state)
- Alignment scores: Additive (Bahdanau), Dot-Product (Luong), Scaled Dot-Product
- Softmax → attention weights → dynamic context vector `c_t`
- Full PyTorch implementation of Bahdanau attention on top of the EP30 seq2seq model
- Attention-weight heatmaps that show the model focusing on the right source words
- BLEU improvement vs the no-attention baseline

## Key formulas
Score (additive):      e_ti = vᵀ tanh(W1 · h_enc_i + W2 · h_dec_t)
Score (dot-product):   e_ti = h_dec_t · h_enc_i
Score (scaled):        e_ti = (h_dec_t · h_enc_i) / √d_k
Weights:               α_ti = softmax(e_ti)
Context:               c_t  = Σ α_ti · h_enc_i
text## Files
| File                    | Description                              |
|-------------------------|------------------------------------------|
| `ep31_attention.py`     | BahdanauAttention + AttentionDecoder + heatmap helper |
| `generate_dl_ep31.py`   | Full video generation pipeline           |
| `manim_dl_ep31.py`      | All Manim scenes used in the episode     |

## Challenge
1. Install: `pip install torch matplotlib`
2. Add the `BahdanauAttention` layer to your EP30 number-word translator
3. Train on the dataset `one → 1`, `two → 2`, … `ten → 10`
4. Generate attention heatmaps for at least 5 phrases
5. Answer: Which source word does the model attend to most for each output digit?
6. Post your heatmap + BLEU improvement vs the no-attention baseline

## Run the challenge solution
```bash
python ep31_attention.py
