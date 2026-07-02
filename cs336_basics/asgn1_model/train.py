import argparse
import os
import time
import numpy as np
import torch
import wandb
import cs336_basics.asgn1_model.dataloader as dataloader
import cs336_basics.asgn1_model.TransformerLM as TransformerLM
import cs336_basics.asgn1_model.AdamOpt as AdamOpt
import cs336_basics.asgn1_model.CE_loss as calculate_ce_loss
import cs336_basics.asgn1_model.LR as LR
import cs336_basics.asgn1_model.utils as utils
import cs336_basics.tokenizer as Tokenizer

import cs336_basics.asgn1_model.dataloader as dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the assignment language model")

    # Data
    parser.add_argument("--dataset-path", type=str, default="data/TinyStories-train.txt")
    parser.add_argument("--val-path", type=str, default="data/TinyStories-valid.txt")

    # Model
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    parser.add_argument("--seed", type=int, default=42)

    # Optimizer
    parser.add_argument("--alpha-max", type=float, default=5e-3)
    parser.add_argument("--alpha-min", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)

    # LR schedule
    parser.add_argument("--warmup-iters", type=int, default=200)

    # Gradient clipping
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Checkpointing & logging
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=20)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="cs336-a1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument("--test-sample", action="store_true", help="Run a test sample generation after training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Wandb init
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )


    tokenizer = Tokenizer.Tokenizer.from_files(vocab_filepath='cs336_basics/tinystories_train/vocab_dict.pkl', merges_filepath='cs336_basics/tinystories_train/merges_bytes.pkl', special_tokens=['<|endoftext|>'])

    if not os.path.exists('data/tinystories_train.npy'):

        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
        np.save('data/tinystories_train.npy', ids)

    if not os.path.exists('data/tinystories_val.npy'):

        with open(args.val_path, 'r', encoding='utf-8') as f:
            ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
        np.save('data/tinystories_val.npy', ids)

    train_tokens = 'data/tinystories_train.npy'
    val_tokens = 'data/tinystories_val.npy'


    # Load datasets
    data = np.load(train_tokens, mmap_mode='r')
    val_data = np.load(val_tokens, mmap_mode='r')

    # Instantiate model and optimizer
    lm = TransformerLM.TransformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta,
    )
    lm.to(device)

    optimizer = AdamOpt.AdamW(
        lm.parameters(),
        lr=args.alpha_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    lm.train()
    train_start = time.time()

    if args.test_sample:
        x, y = dataloader.sample_data(data, 1, args.context_length, device)

        for i in range(args.total_steps):
            it_start = time.time()

            # LR schedule
            lr = LR.lr_schedule(i + 1, args.alpha_max, args.alpha_min, 200, args.total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr

            # Forward + backward
            optimizer.zero_grad()
            logits = lm(x)
            loss = calculate_ce_loss.calculate_ce_loss(logits, y)
            loss.backward()
            LR.gradient_clip(lm.parameters(), args.max_grad_norm)
            optimizer.step()

            it_time = time.time() - it_start
            wall_time = time.time() - train_start

            # Logging
            if i % args.log_interval == 0:

                print(
                    f"step {i:>6}/{args.total_steps} | "
                    f"train_loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"it {it_time*1000:.0f}ms | "
                    f"wall {wall_time:.0f}s"
                )


    else:
        for i in range(args.total_steps):
            it_start = time.time()

            # LR schedule
            lr = LR.lr_schedule(i + 1, args.alpha_max, args.alpha_min, args.warmup_iters, args.total_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr

            # Forward + backward
            x, y = dataloader.sample_data(data, args.batch_size, args.context_length, device)
            optimizer.zero_grad()
            logits = lm(x)
            loss = calculate_ce_loss.calculate_ce_loss(logits, y)
            loss.backward()
            LR.gradient_clip(lm.parameters(), args.max_grad_norm)
            optimizer.step()

            it_time = time.time() - it_start
            wall_time = time.time() - train_start

            # Logging
            if i % args.log_interval == 0:
                grad_norm_sq = torch.zeros((), device=device)
                for p in lm.parameters():
                    if p.grad is not None:
                        grad_norm_sq += p.grad.norm() ** 2
                grad_norm = torch.sqrt(grad_norm_sq).item()

                dead_params = []
                for name, p in lm.named_parameters():
                    if p.grad is None:
                        dead_params.append(name)
                for name in dead_params:
                    print(f"dead_grad_param {name}")

                weight_norm_sq = torch.zeros((), device=device)
                for p in lm.parameters():
                    weight_norm_sq += p.data.norm() ** 2
                weight_norm = torch.sqrt(weight_norm_sq).item()

                loss_has_nan = loss.isnan().item()
                grad_has_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any().item()
                    for p in lm.parameters()
                )
                if loss_has_nan:
                    print("WARNING: loss is NaN")
                if grad_has_nan:
                    print("WARNING: gradient contains NaN")

                lm.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(args.val_batches):
                        x_val, y_val = dataloader.sample_data(val_data, args.batch_size, args.context_length, device)
                        val_loss += calculate_ce_loss.calculate_ce_loss(lm(x_val), y_val).item()
                val_loss /= args.val_batches
                lm.train()

                print(
                    f"step {i:>6}/{args.total_steps} | "
                    f"train_loss {loss.item():.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"grad_norm {grad_norm:.4f} | "
                    f"weight_norm {weight_norm:.4f} | "
                    f"lr {lr:.2e} | "
                    f"it {it_time*1000:.0f}ms | "
                    f"wall {wall_time:.0f}s"
                )

                if not args.no_wandb:
                    tokens_per_step = args.batch_size * args.context_length
                    tokens_per_sec = tokens_per_step / it_time if it_time > 0 else float("inf")

                    wandb_metrics = {
                        "train/loss": loss.item(),
                        "val/loss": val_loss,
                        "train/grad_norm": grad_norm,
                        "train/weight_norm": weight_norm,
                        "train/lr": lr,
                        "perf/tokens_per_sec": tokens_per_sec,
                        "perf/iter_time_ms": it_time * 1000,
                        "perf/wall_time_s": wall_time,
                    }

                    for name, p in lm.named_parameters():
                        if p.grad is not None:
                            safe_name = name.replace('.', '/')
                            wandb_metrics[f"grad_norm/{safe_name}"] = p.grad.norm().item()

                    wandb.log(wandb_metrics, step=i)


            # Checkpointing
            if i % args.checkpoint_interval == 0 and i > 0:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                checkpoint_file = os.path.join(args.checkpoint_path, f"checkpoint_{i}.pt")
                utils.save_checkpoint(lm, optimizer, i, checkpoint_file)
                print(f"checkpoint saved → {checkpoint_file}")

        # Final checkpoint
        os.makedirs(args.checkpoint_path, exist_ok=True)
        utils.save_checkpoint(lm, optimizer, args.total_steps,
                            os.path.join(args.checkpoint_path, "checkpoint_final.pt"))

        if not args.no_wandb:
            wandb.finish()
