# Pico GPT Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PICO GPT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                                INPUT LAYER                              │
│                                                                         │
│  Input Text: "Hello world"                                              │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │   TOKENIZER     │  Character/BPE tokenization                       │
│  │  SimpleTokenizer│  "Hello world" → [72, 101, 108, 108, 111, ...]    │
│  │  BPETokenizer   │                                                    │
│  └─────────────────┘                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  Token IDs: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING LAYERS                              │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐                          │
│  │ Token Embedding  │    │Position Embedding│                          │
│  │    (wte)         │    │     (wpe)        │                          │
│  │  vocab_size      │    │   block_size     │                          │
│  │     ↓            │    │      ↓           │                          │
│  │  n_embd dim      │    │   n_embd dim     │                          │
│  └──────────────────┘    └──────────────────┘                          │
│            │                       │                                   │
│            └───────────┬───────────┘                                   │
│                        ▼                                               │
│                 ┌─────────────┐                                        │
│                 │  Element +  │                                        │
│                 │   Dropout   │                                        │
│                 └─────────────┘                                        │
│                        │                                               │
│                        ▼                                               │
│              Embedded Sequence                                         │
│             [batch_size, seq_len, n_embd]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER BLOCKS (n_layer)                    │
│                                                                         │
│  ┌─ BLOCK 1 ──────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  Input: x                                                          │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────┐                                                  │ │
│  │  │ Layer Norm 1 │                                                  │ │
│  │  └──────────────┘                                                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────────────────────────────────────┐                  │ │
│  │  │         CAUSAL SELF-ATTENTION               │                  │ │
│  │  │                                             │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │ Q,K,V   │  Linear projection             │                  │ │
│  │  │  │ Linear  │  (n_embd → 3 * n_embd)         │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  │      │                                      │                  │ │
│  │  │      ▼                                      │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │Multi-Head│  Split into n_head            │                  │ │
│  │  │  │Attention │  Compute attention weights    │                  │ │
│  │  │  │         │  Apply causal mask             │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  │      │                                      │                  │ │
│  │  │      ▼                                      │                  │ │
│  │  │  ┌─────────┐                                │                  │ │
│  │  │  │Output   │  Concatenate heads             │                  │ │
│  │  │  │Linear   │  Project back (n_embd)         │                  │ │
│  │  │  │+Dropout │                                │                  │ │
│  │  │  └─────────┘                                │                  │ │
│  │  └──────────────────────────────────────────────┘                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────┐    ◄── Residual Connection                           │ │
│  │  │    +     │                                                      │ │
│  │  └──────────┘                                                      │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────┐                                                  │ │
│  │  │ Layer Norm 2 │                                                  │ │
│  │  └──────────────┘                                                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────────────────────────────────────────┐                  │ │
│  │  │                 MLP                          │                  │ │
│  │  │                                              │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │   Linear    │  n_embd → 4 * n_embd       │                  │ │
│  │  │  │  (c_fc)     │                            │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  │         │                                   │                  │ │
│  │  │         ▼                                   │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │    GELU     │  Activation function       │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  │         │                                   │                  │ │
│  │  │         ▼                                   │                  │ │
│  │  │  ┌─────────────┐                            │                  │ │
│  │  │  │   Linear    │  4 * n_embd → n_embd       │                  │ │
│  │  │  │ (c_proj)    │                            │                  │ │
│  │  │  │  +Dropout   │                            │                  │ │
│  │  │  └─────────────┘                            │                  │ │
│  │  └──────────────────────────────────────────────┘                  │ │
│  │     │                                                              │ │
│  │     ▼                                                              │ │
│  │  ┌──────────┐    ◄── Residual Connection                           │ │
│  │  │    +     │                                                      │ │
│  │  └──────────┘                                                      │ │
│  │     │                                                              │ │
│  │     ▼  Output to next block                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ BLOCK 2...n ────────────────────────────────────────────────────┐   │
│  │                    (Same structure)                              │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FINAL OUTPUT LAYER                           │
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │ Final Layer Norm │  Normalize final transformer output              │
│  │     (ln_f)       │                                                  │
│  └──────────────────┘                                                   │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                   │
│  │ Language Model   │  Linear: n_embd → vocab_size                     │
│  │ Head (lm_head)   │  Weight tied with input embeddings              │
│  └──────────────────┘                                                   │
│           │                                                            │
│           ▼                                                            │
│     Logits: [batch_size, seq_len, vocab_size]                         │
│                                                                         │
│  For Training:                    For Generation:                      │
│  ┌────────────────┐               ┌─────────────────┐                  │
│  │ Cross Entropy  │               │  Temperature    │                  │
│  │     Loss       │               │    Scaling      │                  │
│  │                │               │       │         │                  │
│  │ Compare with   │               │       ▼         │                  │
│  │ target tokens  │               │  ┌──────────┐   │                  │
│  └────────────────┘               │  │ Top-k    │   │                  │
│                                   │  │Sampling  │   │                  │
│                                   │  └──────────┘   │                  │
│                                   │       │         │                  │
│                                   │       ▼         │                  │
│                                   │  ┌──────────┐   │                  │
│                                   │  │Multinomial   │                  │
│                                   │  │ Sampling │   │                  │
│                                   │  └──────────┘   │                  │
│                                   └─────────────────┘                  │
│                                           │                            │
│                                           ▼                            │
│                                  Next Token Prediction                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            KEY COMPONENTS                               │
│                                                                         │
│ Configuration (GPTConfig):                                             │
│  • block_size: 1024 (max sequence length)                             │
│  • vocab_size: 50304 (vocabulary size)                                │
│  • n_layer: 12 (number of transformer blocks)                         │
│  • n_head: 12 (number of attention heads)                             │
│  • n_embd: 768 (embedding dimension)                                  │
│  • dropout: 0.0 (dropout rate)                                        │
│                                                                         │
│ Training Components:                                                    │
│  • Training Loop: batch processing, loss calculation                   │
│  • AdamW Optimizer: weight decay regularization                        │
│  • Learning Rate Scheduling: configurable learning rate               │
│  • Validation: periodic loss evaluation                                │
│  • Checkpointing: model state saving                                   │
│                                                                         │
│ Generation Features:                                                    │
│  • Autoregressive: generates one token at a time                       │
│  • Temperature Sampling: controls randomness                           │
│  • Top-k Sampling: limits vocabulary for coherent output               │
│  • Causal Masking: prevents looking ahead during attention             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW SUMMARY                            │
│                                                                         │
│  Text Input → Tokenization → Embeddings → Transformer Blocks →        │
│  → Layer Norm → Language Model Head → Logits → Sampling → Output      │
│                                                                         │
│  Training: Input/Target pairs, Cross-entropy loss, Backpropagation     │
│  Generation: Autoregressive sampling with temperature and top-k        │
└─────────────────────────────────────────────────────────────────────────┘
```