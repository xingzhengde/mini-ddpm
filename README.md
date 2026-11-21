# mini-ddpm-web

[打开 GitHub Pages 生成器](https://xingzhengde.github.io/mini-ddpm/)

浏览器使用 ONNX Runtime Web 和 WebGPU 执行 MNIST U-Net。JavaScript 按 `η=0` 实现 DDIM，并用 Canvas 显示 `28×28` 图像。生成器支持无条件模式、指定数字 `0–9`、`10 / 20 / 50 / 200` 步和固定种子。

## 在本机运行

在仓库根目录启动静态服务，用支持 WebGPU 的浏览器打开 [本机页面](http://127.0.0.1:8000/site/)。模型加载、采样进度、首次加载和本次采样耗时都在页面显示。

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

「运行固定样本对照」用固定的 float32 初始噪声生成数字 `7`，并列显示 20 步浏览器结果和原始 PyTorch 参考图。生成输入框的种子用于验证网页生成的可复现性。跨端实验使用测试文件中相同的初始张量。

## 模型接口

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| `unconditional.onnx` | `x_t` 为 `float32[1,1,28,28]`，`t` 为 `int64[1]` | `epsilon` 为 `float32[1,1,28,28]` |
| `conditional.onnx` | `x_t`、`t` 同上，`labels` 为 `int64[1]` | `epsilon` 为 `float32[1,1,28,28]` |

## 从原权重重新导出

导出脚本使用 ONNX opset 17，并在写出模型前运行 ONNX 检查和 PyTorch 与 ONNX Runtime CPU 的单步数值比对。可使用 Python 3.13、PyTorch 2.11、torchvision 0.26、ONNX 1.21、ONNX Runtime 1.26 与 Matplotlib。

从前两次实验取出以下原始文件，按相对路径放回仓库。

| 来源 | 仓库中的路径 |
| --- | --- |
| Mission 2 第 5 轮检查点 | `python/mission2/checkpoints/train/20261021_200942_740711_cuda/last.pt` |
| Mission 3 第 5 轮检查点 | `python/mission3/outputs/last.pt` |
| Mission 3 固定噪声及标签 | `python/mission3/outputs/evaluation/inputs.pt` |

在仓库根目录运行以下命令。若要更新发布站点，把生成的文件中与 `site/outputs/onnx/` 同名的五项复制到该目录。

```powershell
python -B python/mission3/export_unconditional.py
python -B python/mission3/export_conditional.py
python -B python/mission3/export_ddim_schedule.py
python -B python/mission3/export_ddim_reference.py
```

## GitHub Pages 部署

在 GitHub 的仓库设置中，把 **Pages → Build and deployment → Source** 设为 **GitHub Actions**。将改动推送到 `main`，或在 Actions 手动运行 **Deploy GitHub Pages**。工作流上传 `site/` 并将其发布到仓库的 Pages 项目地址。部署完成后，在公开页面分别运行两种生成模式和固定样本对照。
