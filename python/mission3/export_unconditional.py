"""导出 Mission 2 的无条件 U-Net 并比较一次 CPU 推理。"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "onnx")
    args = parser.parse_args()

    # 使用 Mission 2 的加载函数，使它找到同目录下的无条件 U-Net 定义。
    mission2 = ROOT.parent / "mission2"
    sys.path.insert(0, str(mission2))
    from checkpoint import load_checkpoint

    checkpoint = mission2 / "checkpoints/train/20261021_200942_740711_cuda/last.pt"
    model, _ = load_checkpoint(checkpoint, torch.device("cpu"))

    # 两份脚本复用 Mission 3 评估中第一条标签为 7 的样本所用的噪声。
    evaluation = torch.load(
        ROOT / "outputs/evaluation/inputs.pt", map_location="cpu", weights_only=True,
    )
    index = torch.where(evaluation["labels"] == 7)[0][0].item()
    x_t = evaluation["initial_noise"][index:index + 1].contiguous()
    t = torch.tensor([999], dtype=torch.int64)

    with torch.inference_mode():
        expected = model(x_t, t).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "unconditional.onnx"
    # 先验证临时模型。检查通过后才替换正式文件。
    with TemporaryDirectory(dir=args.output_dir) as temp_dir:
        staged_model = Path(temp_dir) / destination.name
        with torch.no_grad():
            torch.onnx.export(
                model, (x_t, t), str(staged_model),
                input_names=["x_t", "t"], output_names=["epsilon"],
                opset_version=17, dynamo=False, external_data=False,
            )

        onnx.checker.check_model(str(staged_model), full_check=True)
        session = ort.InferenceSession(str(staged_model), providers=["CPUExecutionProvider"])
        actual, = session.run(["epsilon"], {"x_t": x_t.numpy(), "t": t.numpy()})

        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise ValueError("噪声预测包含 NaN 或 Inf")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        error = np.abs(actual - expected)
        for tensor in session.get_inputs() + session.get_outputs():
            print(f"{tensor.name}: {tensor.shape} {tensor.type}")
        print(f"平均绝对误差 {error.mean():.3e}，最大绝对误差 {error.max():.3e}")

        fixture_path = destination.with_suffix(".fixture.json")
        staged_fixture = Path(temp_dir) / fixture_path.name
        fixture = {
            "x_t": x_t.reshape(-1).tolist(),
            "t": t.item(),
            "epsilon_ref": expected.reshape(-1).tolist(),
        }
        staged_fixture.write_text(json.dumps(fixture, allow_nan=False), encoding="utf-8")
        staged_model.replace(destination)
        staged_fixture.replace(fixture_path)

    print(f"已导出 {destination}")
    print(f"已保存浏览器对齐数据 {fixture_path}")


if __name__ == "__main__":
    main()
