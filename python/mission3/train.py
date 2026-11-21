"""读取 MNIST，训练噪声预测模型，保存权重和训练曲线。"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_alpha_bars(config, device):
    betas = torch.linspace(
        config["beta_start"], config["beta_end"], config["timesteps"]
    )
    return torch.cumprod(1 - betas, dim=0).to(device)


def train(model, config, device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])
    dataset = datasets.MNIST(
        config["data_dir"], train=True, download=False, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    alpha_bars = make_alpha_bars(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    model.train()

    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0.0
        total_images = 0
        for step, (x, labels) in enumerate(loader, start=1):
            x, labels = x.to(device), labels.to(device)
            t = torch.randint(0, len(alpha_bars), (len(x),), device=device)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x_t = alpha_bar_t.sqrt() * x + (1 - alpha_bar_t).sqrt() * noise

            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(x_t, t, labels)
            loss = (predicted_noise - noise).square().mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_images += len(x)
            if step == 1 or step % 100 == 0:
                print(
                    f"Epoch {epoch}/{config['epochs']} | "
                    f"Batch {step}/{len(loader)} | Loss {loss.item():.5f}",
                    flush=True,
                )

        average_loss = total_loss / total_images
        history.append((epoch, average_loss))
        print(f"Epoch {epoch} | Average Loss {average_loss:.5f}", flush=True)

        # 配置随权重保存，生成时使用训练时的网络和噪声时间表。
        torch.save({
            "model": model.state_dict(), "config": config,
            "epoch": epoch, "history": history,
        }, output_dir / "last.pt")
        with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("epoch", "mean_loss"))
            writer.writerows(history)

        fig, ax = plt.subplots()
        ax.plot(*zip(*history), marker="o")
        ax.set(xlabel="Epoch", ylabel="Mean noise MSE", title="Training loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
