Yep. This is one of those places where PyTorch throws **seven knobs at you and expects you to magically know what they do**. 😂

Let's use your Fry pipeline as the context.

```text
BinaryDataset
      ↓
   DataLoader
      ↓
    Trainer
      ↓
    Fry model
```

## `batch_size`

This is simply:

> **How many training samples the model processes before one optimizer update.**

Suppose:

```python
batch_size = 8
```

Your `BinaryDataset` produces:

```text
sample 1
sample 2
sample 3
...
sample 8
```

The DataLoader groups them:

```text
X.shape = [8, 512]
Y.shape = [8, 512]
```

Then Fry processes those 8 sequences simultaneously.

After calculating the loss:

```text
forward
   ↓
loss
   ↓
backward
   ↓
optimizer.step()
```

you've completed **one training step**.

### Bigger batch

```text
batch_size = 32
```

means:

```text
[32, 512]
```

More samples per step → generally better hardware utilization, but **more memory usage**.

On your CPU-focused Fry model, we'll have to experiment with this.

---

# `num_workers`

This one is about **how many background processes load data**.

```python
num_workers = 0
```

means:

```text
Training process
      │
      ├── load sample
      ├── prepare sample
      ├── train
      ├── load sample
      └── train
```

Everything happens in the main process.

With:

```python
num_workers = 4
```

you roughly get:

```text
              ┌── Worker 0 ── loads data
              ├── Worker 1 ── loads data
Training ─────┼── Worker 2 ── loads data
              └── Worker 3 ── loads data
                    │
                    ▼
                 batches
                    │
                    ▼
                  Fry
```

While Fry is training on one batch, workers can prepare the next one.

### Why this matters

If your dataset is slow to read/process, the GPU/CPU can sit around waiting for data.

Workers help hide that latency.

But there's a catch for **your project**:

You're on Windows and Fry is deliberately CPU-friendly.

I'd initially use:

```python
num_workers = 0
```

and benchmark before increasing it.

More workers aren't automatically faster. They also introduce:

* process overhead
* extra memory
* multiprocessing complexity
* Windows-specific spawning overhead

---

# `pin_memory`

This one is primarily useful when transferring tensors from **CPU → GPU**.

Normally:

```text
RAM
 ↓
GPU
```

Pinned (page-locked) memory can make those transfers more efficient.

So:

```python
pin_memory=True
```

is commonly used when training on CUDA.

For Fry's intended **CPU training/inference**, it isn't particularly useful.

I'd therefore start with:

```python
pin_memory=False
```

If we later train a larger Misawantiga model on a GPU, we can turn it on.

---

# `shuffle`

You've already encountered this with `BinaryDataset`, but there's an important distinction.

For a normal map-style dataset:

```python
DataLoader(
    dataset,
    shuffle=True
)
```

means the DataLoader changes the order in which samples are requested.

But your `BinaryDataset` is an `IterableDataset`.

That's different.

Your dataset itself is responsible for determining the iteration order.

So our shuffling logic belongs primarily in **`BinaryDataset`**, which is why we implemented it there.

We don't want two independent shuffling mechanisms fighting each other.

---

# `drop_last`

Suppose you have:

```text
103 samples
batch_size = 16
```

You get:

```text
16
16
16
16
16
16
+ 7
```

With:

```python
drop_last=False
```

the final batch contains:

```text
7 samples
```

With:

```python
drop_last=True
```

those 7 samples are discarded.

For language-model training, this can sometimes be useful if you absolutely want every batch to have identical dimensions.

But we're dealing with an `IterableDataset`, so we'll decide whether we actually need it rather than blindly enabling it.

---

# `persistent_workers`

This matters when:

```python
num_workers > 0
```

Normally workers can be shut down and recreated between epochs.

With:

```python
persistent_workers=True
```

they stay alive between epochs.

That avoids repeatedly creating worker processes.

Conceptually:

```text
Without persistent workers:

Epoch 1 → create workers → train → destroy
Epoch 2 → create workers → train → destroy
Epoch 3 → create workers → train → destroy


With persistent workers:

create workers
     ↓
Epoch 1
     ↓
Epoch 2
     ↓
Epoch 3
     ↓
destroy workers
```

Again, probably irrelevant while we're starting with `num_workers=0`.

---

# `prefetch_factor`

This controls how many batches **each worker prepares ahead of time**.

For example:

```python
num_workers = 4
prefetch_factor = 2
```

means each worker can prepare roughly two batches ahead.

Conceptually:

```text
Worker 0 → [batch] [batch]
Worker 1 → [batch] [batch]
Worker 2 → [batch] [batch]
Worker 3 → [batch] [batch]
                         ↓
                       Trainer
```

So there can be multiple batches waiting before the model needs them.

Again, this only matters when workers are enabled.

---

# `timeout`

How long the DataLoader waits for a worker to return data before raising an error.

For example:

```python
timeout = 30
```

means roughly:

> "If a worker doesn't produce the requested data within 30 seconds, something is probably wrong."

Usually:

```python
timeout = 0
```

means no timeout.

We probably don't need to touch this for Fry.

---

# `collate_fn`

**This one is particularly important for understanding your `batching.py`.**

The DataLoader gets individual samples:

```python
(x1, y1)
(x2, y2)
(x3, y3)
```

and needs to turn them into a batch.

The function responsible for defining how that happens is the **collate function**.

Conceptually:

```text
(x₁, y₁)
(x₂, y₂)
(x₃, y₃)
       │
       ▼
   collate_fn
       │
       ▼
X = [x₁
     x₂
     x₃]

Y = [y₁
     y₂
     y₃]
```

For your fixed-length language-model sequences, this can be extremely simple because every sample already has the same length.

So `batching.py` may eventually contain something like a custom collator—but **we should only write one if PyTorch's default behavior isn't sufficient**.

---

# `batch_size` vs gradient accumulation

This one will become important when we build the Trainer.

Suppose your CPU can comfortably handle:

```text
batch_size = 4
```

but you want the optimization behavior of roughly:

```text
batch_size = 16
```

We can accumulate gradients:

```text
batch 1 → backward
batch 2 → backward
batch 3 → backward
batch 4 → backward
             ↓
       optimizer.step()
```

So:

```text
effective batch size
≈ batch_size × accumulation_steps
```

For example:

```text
4 × 4 = 16
```

This is **Trainer functionality**, not something `DataLoader` itself handles.

---

# For Fry, I'd start extremely boring

Something like:

```python
DataLoader(
    dataset,
    batch_size=8,
    num_workers=0,
    pin_memory=False,
)
```

Then benchmark.

The important thing is that **these parameters aren't independent magic switches**.

Think of them in categories:

```text
                DataLoader
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Batching       Loading      Memory
       │            │            │
 batch_size    num_workers   pin_memory
 drop_last     prefetch      ...
 collate_fn    persistent
```

And your overall pipeline becomes:

```text
BinaryDataset
   │
   │ individual (x, y)
   ▼
DataLoader
   │
   ├── batches them
   ├── optionally uses workers
   ├── optionally prefetches
   └── returns tensors
   │
   ▼
Trainer
   │
   ├── forward
   ├── loss
   ├── backward
   └── optimizer.step()
```

**For our immediate implementation, the three knobs I'd care about first are `batch_size`, `num_workers`, and `collate_fn`.** Everything else can remain at sensible defaults until we have a reason to change it.
