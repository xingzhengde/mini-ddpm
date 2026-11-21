"""入口：训练条件 U-Net，或加载已有权重进行 DDIM 生成。"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from torchvision.utils import save_image

from model import UNet
from train import make_alpha_bars, train


def make_ddim_timesteps(training_timesteps: int, steps: int) -> list[int]:
    """从原始训练时间轴等距选点，返回升序整数索引。"""
    return torch.linspace(0, training_timesteps - 1, steps, dtype=torch.float64).round().long().tolist()


@torch.no_grad()
def sample_ddim(model, labels, initial_noise, alpha_bars, steps):
    model.eval()
    x_t = initial_noise.clone()
    # 与 mission2 一样，先按升序选点，然后逆序更新。
    times = make_ddim_timesteps(len(alpha_bars), steps)

    for i in reversed(range(len(times))):
        current_t = times[i]
        t = torch.full((len(x_t),), current_t, device=x_t.device, dtype=torch.long)
        predicted_noise = model(x_t, t, labels)
        alpha_bar_t = alpha_bars[current_t]
        x_estimated = (
            x_t - (1 - alpha_bar_t).sqrt() * predicted_noise
        ) / alpha_bar_t.sqrt()

        if i == 0:
            x_t = x_estimated  # 干净终点的累计系数为 1。
        else:
            alpha_bar_s = alpha_bars[times[i - 1]]
            x_t = (
                alpha_bar_s.sqrt() * x_estimated
                + (1 - alpha_bar_s).sqrt() * predicted_noise
            )
    return x_t


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "sample"), default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, nargs="+", default=[50])
    parser.add_argument("--digit", type=int, choices=range(10))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or any(k < 2 or k > 1000 for k in args.steps):
        parser.error("epochs 和 batch-size 需为正数，steps 需在 2 到 1000 之间")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # 沿用 mission2 的 FP32 设置。
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = {
        "base_channels": 32, "time_embedding_dim": 32,
        "timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02,
        "learning_rate": 2e-4, "batch_size": args.batch_size,
        "epochs": args.epochs, "seed": args.seed,
        "data_dir": str(root.parent / "mission2" / "data"),
    }

    checkpoint = args.checkpoint or args.output_dir / "last.pt"
    if args.mode == "train":
        model = UNet(config["base_channels"], config["time_embedding_dim"]).to(device)
        print(f"Device {device} | Parameters {sum(p.numel() for p in model.parameters()):,}")
        train(model, config, device, args.output_dir)
        checkpoint = args.output_dir / "last.pt"

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = UNet(config["base_channels"], config["time_embedding_dim"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    alpha_bars = make_alpha_bars(config, device)
    print(f"Loaded epoch {saved['epoch']} from {checkpoint}", flush=True)

    # 默认每行一个类别，共 10 行；指定 --digit 后全部生成该数字。
    labels = torch.arange(10, device=device).repeat_interleave(8)
    if args.digit is not None:
        labels.fill_(args.digit)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(80, 1, 28, 28, device=device, generator=generator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "digits" if args.digit is None else f"digit_{args.digit}"
    torch.save({"initial_noise": initial_noise.cpu(), "labels": labels.cpu()},
               args.output_dir / f"{prefix}_inputs.pt")

    with torch.no_grad():
        model(initial_noise, torch.zeros(80, device=device, dtype=torch.long), labels)
    timing_rows = []
    for steps in args.steps:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        samples = sample_ddim(model, labels, initial_noise, alpha_bars, steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        # 显示前 [-1,1] -> [0,1]；保留原始输出便于后续评估。
        stem = f"{prefix}_ddim_{steps}"
        torch.save(samples.cpu(), args.output_dir / f"{stem}.pt")
        save_image(((samples + 1) / 2).clamp(0, 1).cpu(),
                   args.output_dir / f"{stem}.png", nrow=8)
        timing_rows.append((steps, len(samples), elapsed))
        print(f"DDIM {steps} steps | {elapsed:.3f}s | {args.output_dir / (stem + '.png')}", flush=True)
    with (args.output_dir / f"{prefix}_timings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("steps", "num_samples", "sampling_seconds"))
        writer.writerows(timing_rows)


if __name__ == "__main__":
    main()
