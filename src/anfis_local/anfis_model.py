# src/anfis/anfis_model.py
import torch.nn as nn

class ANFISDietModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)   # 4 diet classes
        )

    def forward(self, x):
        return self.net(x)
