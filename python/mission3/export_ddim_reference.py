"""从已对齐的初始噪声生成条件模型的 20 步 Python 参考结果。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from main import sample_ddim
from model import UNet
from train import make_alpha_bars


ROOT = Path(__file__).resolve().parent
STEPS = 20


def main() -> None:
    torch.set_num_threads(2)
    checkpoint = torch.load(ROOT / "outputs/last.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = UNet(config["base_channels"], config["time_embedding_dim"])
    model.load_state_dict(checkpoint["model"], strict=True)

    output_dir = ROOT / "outputs" / "onnx"
    fixture = json.loads((output_dir / "conditional.fixture.json").read_text(encoding="utf-8"))
    noise = torch.tensor(fixture["x_t"], dtype=torch.float32).reshape(1, 1, 28, 28)
    labels = torch.tensor([fixture["label"]], dtype=torch.int64)
    alpha_bars = make_alpha_bars(config, torch.device("cpu"))
    final_image = sample_ddim(model, labels, noise, alpha_bars, STEPS)
    reference = {
        "steps": STEPS,
        "label": fixture["label"],
        "final_x": final_image.flatten().tolist(),
    }

    target = output_dir / "conditional.ddim20.reference.json"
    with TemporaryDirectory(dir=output_dir) as temp_dir:
        staged = Path(temp_dir) / target.name
        staged.write_text(json.dumps(reference, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(target)
    print(f"已保存数字 {fixture['label']} 的 {STEPS} 步 Python 参考结果到 {target}")


if __name__ == "__main__":
    main()
