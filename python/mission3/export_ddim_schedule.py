"""从 Mission 3 的检查点导出浏览器 DDIM 调度数据。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from main import make_ddim_timesteps
from train import make_alpha_bars


ROOT = Path(__file__).resolve().parent


def main() -> None:
    checkpoint = torch.load(ROOT / "outputs/last.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    alpha_bars = make_alpha_bars(config, torch.device("cpu"))
    # 参考索引按实际去噪顺序保存，让 JavaScript 可逐项核对自己的时间步。
    reference_timesteps = list(reversed(make_ddim_timesteps(len(alpha_bars), 20)))
    schedule = {
        "training_timesteps": len(alpha_bars),
        "beta_start": config["beta_start"],
        "beta_end": config["beta_end"],
        "eta": 0.0,
        "terminal_alpha_bar": 1.0,
        "alpha_bars": alpha_bars.tolist(),
        "reference_timesteps_20": reference_timesteps,
    }

    target = ROOT / "outputs/onnx/ddim_schedule.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temp_dir:
        staged = Path(temp_dir) / target.name
        staged.write_text(json.dumps(schedule, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(target)

    print(f"已导出 {len(alpha_bars)} 个累计系数到 {target}")
    print(f"20 步参考时间序列 {reference_timesteps}")


if __name__ == "__main__":
    main()
