const f32 = Math.fround;

export function makeGaussianNoise(seed, length) {
  // Mulberry32 以一个 32 位种子初始化。将均匀随机数按 Box–Muller 变换成高斯噪声。
  let state = seed >>> 0;
  function uniform() {
    state = (state + 0x6D2B79F5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), state | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 0x100000000;
  }

  const noise = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - uniform()));
    const angle = 2 * Math.PI * uniform();
    noise[i] = radius * Math.cos(angle);
    if (i + 1 < length) noise[i + 1] = radius * Math.sin(angle);
  }
  return noise;
}

export function selectTimesteps(trainingTimesteps, steps) {
  const last = trainingTimesteps - 1;
  const intervals = steps - 1;
  return Array.from({ length: steps }, (_, index) => {
    const numerator = (intervals - index) * last;
    const lower = Math.floor(numerator / intervals);
    const twiceRemainder = 2 * (numerator % intervals);
    // torch.round 对正好一半的值取最近的偶数。
    if (twiceRemainder === intervals) return lower + (lower % 2);
    return lower + Number(twiceRemainder > intervals);
  });
}

function updateState(state, epsilon, alphaBar, previousAlphaBar) {
  const signal = f32(Math.sqrt(alphaBar));
  const noise = f32(Math.sqrt(f32(1 - alphaBar)));
  const previousSignal = f32(Math.sqrt(previousAlphaBar));
  const previousNoise = f32(Math.sqrt(f32(1 - previousAlphaBar)));
  const updated = new Float32Array(state.length);

  for (let i = 0; i < state.length; i++) {
    // 对应 Python 中各个 float32 张量运算的舍入边界。
    const estimatedClean = f32(f32(state[i] - f32(noise * epsilon[i])) / signal);
    updated[i] = previousAlphaBar === 1
      ? estimatedClean
      : f32(f32(previousSignal * estimatedClean) + f32(previousNoise * epsilon[i]));
  }
  return updated;
}

export async function sampleDdim(initialNoise, schedule, timesteps, predictNoise, onStep) {
  const alphaBars = Float32Array.from(schedule.alpha_bars);
  let state = initialNoise;

  for (const [index, timestep] of timesteps.entries()) {
    const epsilon = await predictNoise(state, timestep);
    if (epsilon.length !== state.length) throw new Error(`时间步 ${timestep} 的噪声形状有误`);
    const nextTimestep = timesteps[index + 1];
    const previousAlphaBar = nextTimestep === undefined
      ? schedule.terminal_alpha_bar
      : alphaBars[nextTimestep];
    state = updateState(state, epsilon, alphaBars[timestep], previousAlphaBar);
    if (!state.every(Number.isFinite)) throw new Error(`时间步 ${timestep} 产生了非有限值`);
    onStep(index + 1, timestep);
  }
  return state;
}

export function compareTensors(actual, reference) {
  if (actual.length !== reference.length) throw new Error("最终张量大小不同");
  let total = 0;
  let max = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - reference[i]);
    total += error;
    max = Math.max(max, error);
  }
  return { mean: total / actual.length, max };
}
