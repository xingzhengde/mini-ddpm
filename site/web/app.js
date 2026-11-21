import { compareTensors, makeGaussianNoise, sampleDdim, selectTimesteps } from "./ddim.js";

const assets = new URL("../outputs/onnx/", import.meta.url);
const runtimeDirectory = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
const imageShape = [1, 1, 28, 28];
const imageSize = 28 * 28;
const form = document.getElementById("generator");
const controls = document.getElementById("controls");
const mode = document.getElementById("mode");
const digit = document.getElementById("digit");
const stepsInput = document.getElementById("steps");
const seedInput = document.getElementById("seed");
const compareButton = document.getElementById("compare");
const progress = document.getElementById("progress");
const status = document.getElementById("status");
const browserCanvas = document.getElementById("browser-image");
const pythonFigure = document.getElementById("python-result");
const errors = document.getElementById("errors");
const timings = document.getElementById("timings");
const sessions = new Map();
let schedule;
let runtime;

async function readJson(name) {
  const response = await fetch(new URL(name, assets));
  if (!response.ok) throw new Error(`无法读取 ${name}（HTTP ${response.status}）`);
  return response.json();
}

async function getRuntime() {
  if (runtime) return runtime;
  if (!window.isSecureContext) throw new Error("请通过 localhost 或 HTTPS 打开页面以使用 WebGPU");
  if (!navigator.gpu) throw new Error("当前浏览器未提供 WebGPU，请检查浏览器的 WebGPU 支持情况");
  if (!window.ort) throw new Error("ONNX Runtime Web 加载失败，请检查与 CDN 的连接");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("浏览器未找到可用的 WebGPU 适配器，请检查硬件加速和显卡驱动");
  const ort = window.ort;
  ort.env.wasm.wasmPaths = runtimeDirectory;
  ort.env.wasm.numThreads = 1;
  ort.env.webgpu.adapter = adapter;
  runtime = ort;
  return runtime;
}

async function getSession(model) {
  const cached = sessions.get(model);
  if (cached) return { ...cached, reused: true };

  const start = performance.now();
  const ort = await getRuntime();
  let session;
  try {
    session = await ort.InferenceSession.create(new URL(`${model}.onnx`, assets).href, {
      executionProviders: ["webgpu"],
    });
  } catch (error) {
    throw new Error(`WebGPU 模型初始化失败。${error.message}`, { cause: error });
  }
  const entry = { ort, session, loadMs: performance.now() - start };
  sessions.set(model, entry);
  return { ...entry, reused: false };
}

function drawImage(canvas, values) {
  const context = canvas.getContext("2d");
  const image = context.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < values.length; i++) {
    const gray = Math.round(Math.max(0, Math.min(1, (values[i] + 1) / 2)) * 255);
    image.data.set([gray, gray, gray, 255], i * 4);
  }
  context.putImageData(image, 0, 0);
}

function showComparison(image, reference) {
  const result = compareTensors(image, Float32Array.from(reference));
  drawImage(document.getElementById("python-image"), reference);
  document.getElementById("mean-error").textContent = result.mean.toExponential(3);
  document.getElementById("max-error").textContent = result.max.toExponential(3);
  pythonFigure.hidden = false;
  errors.hidden = false;
}

async function generate(getInputs) {
  controls.disabled = true;
  compareButton.disabled = true;
  progress.value = 0;
  pythonFigure.hidden = true;
  timings.hidden = true;
  errors.hidden = true;
  browserCanvas.getContext("2d").clearRect(0, 0, browserCanvas.width, browserCanvas.height);

  try {
    status.textContent = "正在准备生成数据";
    const inputs = await getInputs();
    schedule ??= await readJson("ddim_schedule.json");
    if (schedule.eta !== 0) throw new Error("此生成器要求 η=0 的 DDIM 调度");
    const timesteps = selectTimesteps(schedule.training_timesteps, inputs.steps);
    if (inputs.reference && timesteps.join(",") !== schedule.reference_timesteps_20.join(",")) {
      throw new Error("JavaScript 时间步与 Python 参考序列不一致");
    }

    status.textContent = "正在加载模型";
    const { ort, session, loadMs, reused } = await getSession(inputs.model);
    const labels = inputs.label === null
      ? null
      : new ort.Tensor("int64", BigInt64Array.of(BigInt(inputs.label)), [1]);
    const predictNoise = async (image, timestep) => {
      const feeds = {
        x_t: new ort.Tensor("float32", image, imageShape),
        t: new ort.Tensor("int64", BigInt64Array.of(BigInt(timestep)), [1]),
      };
      if (labels !== null) feeds.labels = labels;
      const { epsilon } = await session.run(feeds);
      return epsilon.data;
    };

    progress.max = inputs.steps;
    status.textContent = `采样中 0 / ${inputs.steps} 步`;
    const start = performance.now();
    const image = await sampleDdim(inputs.noise, schedule, timesteps, predictNoise, (step) => {
      progress.value = step;
      status.textContent = `采样中 ${step} / ${inputs.steps} 步`;
    });
    const sampleMs = performance.now() - start;
    drawImage(browserCanvas, image);
    if (inputs.reference) showComparison(image, inputs.reference);

    document.getElementById("load-time").textContent = `${(loadMs / 1000).toFixed(2)} 秒${reused ? "（已复用）" : ""}`;
    document.getElementById("sample-time").textContent = `${(sampleMs / 1000).toFixed(2)} 秒`;
    timings.hidden = false;
    status.textContent = `已完成 ${inputs.steps} / ${inputs.steps} 步 · ${inputs.caption} · WebGPU`;
  } catch (error) {
    status.textContent = `生成失败。${error.message}`;
  } finally {
    controls.disabled = false;
    compareButton.disabled = false;
  }
}

mode.addEventListener("change", () => {
  digit.disabled = mode.value !== "conditional";
});

document.getElementById("random-seed").addEventListener("click", () => {
  seedInput.value = crypto.getRandomValues(new Uint32Array(1))[0];
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const model = mode.value;
  const label = model === "conditional" ? Number(digit.value) : null;
  const seed = seedInput.valueAsNumber;
  const steps = Number(stepsInput.value);
  const caption = label === null ? `无条件 · 种子 ${seed}` : `数字 ${label} · 种子 ${seed}`;
  void generate(() => ({ model, label, steps, caption, noise: makeGaussianNoise(seed, imageSize) }));
});

compareButton.addEventListener("click", () => {
  void generate(async () => {
    const [fixture, reference] = await Promise.all([
      readJson("conditional.fixture.json"),
      readJson("conditional.ddim20.reference.json"),
    ]);
    if (reference.steps !== 20 || fixture.label !== reference.label) {
      throw new Error("固定样本与 Python 参考结果的配置不一致");
    }
    return {
      model: "conditional", label: fixture.label, steps: reference.steps,
      caption: `数字 ${fixture.label} · 固定样本对照`,
      noise: Float32Array.from(fixture.x_t), reference: reference.final_x,
    };
  });
});

controls.disabled = false;
compareButton.disabled = false;
status.textContent = "准备就绪";
