import json
import time
import torch

# Module imports
from misawantiga.config.defaults import Config
from misawantiga.dataset.encoder.binary_dataset import BinaryDataset
from misawantiga.models import build_model
from misawantiga.dataset.loader import create_loader

from training.optimizer import build_optimizer
from training.scheduler import WSDScheduler
from training.checkpoint import try_resume, save_checkpoint, ensure_hub_repo


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def get_dataset(self, is_train: bool = True) -> BinaryDataset:
        meta_path = self.config.train_meta if is_train else self.config.val_meta
        bin_path = self.config.train_bin if is_train else self.config.val_bin

        with open(meta_path, "r") as f:
            meta = json.load(f)

        dtype_name = meta.get("dtype")
        if not dtype_name:
            raise ValueError(f"Missing dtype in metadata: {meta_path}")

        dtype = self.config.dtype_map[dtype_name]

        return BinaryDataset(
            path=bin_path,
            block_size=self.config.block_size,
            dtype=dtype,
            shuffle=is_train,
            seed=self.config.seed if hasattr(self.config, "seed") else 36
        )

    def train(self):
        # Config and device setup
        config = self.config
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        print(f"\n==== Training run: {config.model_name} | Device: {device}")

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Datasets and loaders
        print("Initializing datasets...")
        train_dataset = self.get_dataset(is_train=True)
        val_dataset = self.get_dataset(is_train=False)

        train_loader = create_loader(train_dataset, config)
        val_loader = create_loader(val_dataset, config)

        # Build model
        model = build_model(config).to(device)
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer and WSD Scheduler
        optimizer = build_optimizer(model, config)

        warmup_steps = int(config.max_iters * config.warmup_ratio)
        decay_steps = int(config.max_iters * config.decay_ratio)
        stable_steps = config.max_iters - warmup_steps - decay_steps

        scheduler = WSDScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            min_lr=config.min_lr,
            decay_type="cosine"
        )

        ensure_hub_repo(config)

        # Checkpoint resuming
        start_iter = 0
        best_val_loss = float("inf")
        ckpt = try_resume(config)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt["iter_num"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            for _ in range(start_iter):
                scheduler.step()
            print(f"Resumed from step {start_iter}")

        # Training iterators
        train_iter = iter(train_loader)
        epoch = 0

        def next_batch():
            nonlocal train_iter, epoch
            try:
                return next(train_iter)
            except StopIteration:
                # New Epoch
                epoch += 1
                train_dataset.set_epoch(epoch)
                train_iter = iter(train_loader)
                return next(train_iter)

        # Main Training loop
        model.train()
        t0 = time.time()

        for iter_num in range(start_iter, config.max_iters):
            # Fetch real tokens x and targets y
            x, y = next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass (AMP bfloat16)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # Backward Pass & Optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Logging
            if iter_num % config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                current_lr = scheduler.get_last_lr()[0]
                tokens_per_sec = (config.batch_size * config.block_size) / max(dt, 1e-6)
                print(
                    f"Iter {iter_num:5d} | loss: {loss.item():.4f} | "
                    f"lr: {current_lr:.6f} | tok/s: {tokens_per_sec:,.0f} | "
                    f"step time: {dt*1000:.2f}ms"
                )

            # Evaluation and checkpointing
            if iter_num > 0 and iter_num % config.eval_interval == 0:
                model.eval()
                val_losses = []
                val_iter = iter(val_loader)

                with torch.no_grad():
                    # Average over `eval_iters` batches
                    for _ in range(getattr(config, "eval_iters", 20)):
                        try:
                            vx, vy = next(val_iter)
                        except StopIteration:
                            break
                        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)

                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            _, v_loss = model(vx, vy)
                        val_losses.append(v_loss.item())

                val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
                print(f"---- Eval at step {iter_num}: val_loss = {val_loss:.4f} ----")

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=self.config,
                    iter_num=iter_num,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    tag="best" if is_best else "latest"
                )
                model.train()

        print("🎉 Training finished successfully!")