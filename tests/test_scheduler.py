import torch
from training.scheduler import WSDScheduler

# Create a dummy optimizer
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Initialize your scheduler
scheduler = WSDScheduler(
    optimizer=optimizer,
    warmup_steps=100,
    stable_steps=800,
    decay_steps=100,
    peak_lr=1e-3,
    min_lr=1e-5,
    decay_type="cosine",
)

# Simulate 1000 training steps
lrs = []
for step in range(1000):
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

print(f"Step 0 (Warmup Start):    {lrs[0]:.6f}")
print(f"Step 100 (Peak LR):        {lrs[100]:.6f}")
print(f"Step 500 (Mid-Stable):     {lrs[500]:.6f}")
print(f"Step 900 (Decay Start):    {lrs[900]:.6f}")
print(f"Step 999 (Decay End):      {lrs[999]:.6f}")