#!/usr/bin/env python3
"""
Train satisfaction classifier (11 GUI features) with **early stopping**.

• Model: the existing `AnfisNet` (64‑32‑5) you already have.
• Feature vector exactly matches what the GUI sends.
• Class imbalance handled with `class_weight="balanced"`.
• Logs every 5 epochs, stops automatically when validation accuracy hasn’t
  improved for **8 consecutive checks**.
• Saves the **best** model and the scaler.

Run inside `venv_anfis`:
    $ source venv_anfis/bin/activate
    $ python src/anfis/train_satisfaction.py
"""

from pathlib import Path
import sys, joblib, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# ── paths & imports ─────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE))
from models.anfis_diet import AnfisNet  # noqa: E402

# ── hyper‑params ───────────────────────────────────
SEED      = 42
MAX_EPOCH = 120
BATCH     = 128
LR        = 1e-4
PATIENCE  = 8      # early‑stop patience (×5‑epoch checks)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── load data ──────────────────────────────────────
df = pd.read_csv(BASE / "data" / "user_feedback.csv")
df["satisfaction"] = pd.to_numeric(df["satisfaction"], errors="coerce")
df = df[df["satisfaction"].between(1, 5)]
df["satisfaction"] = df["satisfaction"].astype(int) - 1

# feature engineering
activity_map       = {"Low": 0, "Medium": 1, "High": 2}
df["gender_enc"]   = (df["gender"].str.upper() == "M").astype(int)
df["activity_enc"] = df["activity_level"].map(activity_map).fillna(1).astype(int)
df["bmi"]          = (df["weight"] / ((df["height"] / 100) ** 2)).clip(0)

FEATS = [
    "age", "gender_enc", "bmi", "activity_enc",
    "calories", "total_fat", "sugar", "sodium",
    "protein", "saturated_fat", "carbs",
]

X = df[FEATS].fillna(0).values.astype("float32")
y = df["satisfaction"].values.astype("int64")

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# scaler
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw).astype("float32")
X_val   = scaler.transform(X_val_raw).astype("float32")

MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(scaler, MODEL_DIR / "scaler_satisfaction.pkl")

# dataloaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                         batch_size=BATCH, shuffle=True)
val_tensor   = torch.tensor(X_val)
val_target   = torch.tensor(y_val)

# model + loss
net   = AnfisNet(input_dim=len(FEATS), output_dim=5)
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
opt    = torch.optim.Adam(net.parameters(), lr=LR)

# early‑stop loop
best_acc, no_improve = 0.0, 0
for epoch in range(1, MAX_EPOCH + 1):
    net.train()
    for xb, yb in train_loader:
        opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()

    if epoch == 1 or epoch % 5 == 0:
        net.eval()
        with torch.no_grad():
            preds = torch.argmax(net(val_tensor), 1)
            acc   = (preds == val_target).float().mean().item() * 100
        print(f"epoch {epoch:3d}  val acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(net.state_dict(), MODEL_DIR / "anfis_satisfaction.pth")
        else:
            no_improve += 5
        if no_improve >= PATIENCE:
            break

print(f"Best validation accuracy kept: {best_acc:.2f}% (model saved)")
