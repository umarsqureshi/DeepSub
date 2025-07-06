"""
Configuration parameters for training.
"""

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 7e-5

# Model parameters
MODEL_CONFIG = {
    'upscale': 1,
    'in_chans': 1,
    'img_size': (64, 64),
    'window_size': 8,
    'img_range': 1.,
    'depths': [6, 6, 6, 6, 6, 6],
    'embed_dim': 180,
    'num_heads': [6, 6, 6, 6, 6, 6],
    'mlp_ratio': 2,
    'upsampler': '',
    'resi_connection': '1conv'
} 


class TopTaggingMamba(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm0      = nn.LayerNorm(hidden_dim)

        self.mambas = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=16, expand=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # raw logit
        )

    def forward(self, x):
        x = self.input_proj(x)   # → (batch, seq_len, hidden_dim)
        x = self.norm0(x)

        for m, ln in zip(self.mambas, self.norms):
            delta = m(x)          # (batch, seq_len, hidden_dim)
            x = x + 0.5 * delta   # scale residual
            x = ln(x)
            x = self.dropout(x)

        x = x.mean(dim=1)         # global avg pool → (batch, hidden_dim)
        return self.classifier(x).squeeze(-1)  # → (batch,)