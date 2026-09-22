// verify_pll_core.js — 与 sim_pll_coherent.m 同方程的数值核对
// 缩放: N=8192 → fs=81.92MHz (帧长 100us、FFT 分辨率 10kHz 不变, 算法方程完全相同)
const lambda = 1550e-9, T = 100e-6, N = 8192, fs = N / T;
const df_bin = fs / N;                    // 10 kHz
const IF0 = 500e3, Kp_hz = 10, Kp_pll = 0.2, Kf_pll = 300, c0 = 3e8, phi0 = 0.7;
const wpi = x => { x %= 2 * Math.PI; if (x > Math.PI) x -= 2 * Math.PI; if (x < -Math.PI) x += 2 * Math.PI; return x; };

function fft(re, im) {                    // 就地 radix-2 (调用方传副本)
  const n = re.length;
  for (let i = 0, j = 0; i < n; i++) {
    if (j > i) { let t; t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    let m = n >> 1; while (m >= 1 && j >= m) { j -= m; m >>= 1; } j += m;
  }
  for (let s = 1, m = 2; m <= n; s++, m <<= 1) {
    const m2 = m >> 1, wRe = Math.cos(Math.PI / m2), wIm = -Math.sin(Math.PI / m2);
    for (let k = 0; k < n; k += m) {
      let wr = 1, wi = 0;
      for (let j = 0; j < m2; j++) {
        const a = k + j, b = a + m2;
        const tRe = re[b] * wr - im[b] * wi, tIm = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tRe; im[b] = im[a] - tIm;
        re[a] += tRe; im[a] += tIm;
        const nwr = wr * wRe - wi * wIm; wi = wr * wIm + wi * wRe; wr = nwr;
      }
    }
  }
}

let gs = 0;
function gauss() {
  if (gs) { const v = gs; gs = 0; return v; }
  let u = 0, v = 0;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; } while (u * u + v * v >= 1 || u === 0);
  const r = Math.sqrt(-2 * Math.log(u * u + v * v) / (u * u + v * v));
  gs = v * r; return u * r;
}

const tfr = new Float64Array(N); for (let i = 0; i < N; i++) tfr[i] = i / fs;

function demod(zRe, zIm, f) {              // IQ 解调求和 → 相位(帧中点)
  let sr = 0, si = 0;
  for (let i = 0; i < N; i++) {
    const a = -2 * Math.PI * f * tfr[i], c = Math.cos(a), s = Math.sin(a);
    sr += zRe[i] * c - zIm[i] * s; si += zRe[i] * s + zIm[i] * c;
  }
  return Math.atan2(si, sr);
}

function runCase({ fdMode, fd0 = 0, A = 0, fv = 0, Nf, mode, SNRdB }) {
  const f_track = new Float64Array(Nf + 1), f_est = new Float64Array(Nf);
  const f_true = new Float64Array(Nf), psi_w = new Float64Array(Nf);
  const zRe = new Float64Array(N), zIm = new Float64Array(N);
  const fRe = new Float64Array(N), fIm = new Float64Array(N);
  let psi_corr_prev = 0, phi_nco = 0, f_demod = 0, f_est_prev = 0;
  for (let n = 0; n < Nf; n++) {
    const tn = n * T, t_mid = tn + T / 2;
    for (let i = 0; i < N; i++) {
      const t = tn + tfr[i];
      let ph;
      if (fdMode === 'const') ph = 2 * Math.PI * (IF0 + fd0) * t + phi0;
      else { const pm = 4 * Math.PI * A / lambda * Math.sin(2 * Math.PI * fv * t); ph = 2 * Math.PI * IF0 * t + pm + phi0; }
      zRe[i] = Math.cos(ph); zIm[i] = Math.sin(ph);
    }
    if (isFinite(SNRdB)) {
      const s = Math.sqrt(Math.pow(10, -SNRdB / 10) / 2);
      for (let i = 0; i < N; i++) { zRe[i] += s * gauss(); zIm[i] += s * gauss(); }
    }
    for (let i = 0; i < N; i++) { fRe[i] = zRe[i]; fIm[i] = zIm[i]; }
    fft(fRe, fIm);
    let k = 0, mx = -1;
    for (let i = 0; i < N; i++) { const m = fRe[i] * fRe[i] + fIm[i] * fIm[i]; if (m > mx) { mx = m; k = i; } }
    let f_coarse = k * df_bin; if (f_coarse > fs / 2) f_coarse -= fs;
    if (n === 0) f_demod = f_coarse;
    else f_demod = f_est_prev;                                   // 解调参考=上一帧估计(连续跟踪)
    const psi_iq1 = demod(zRe, zIm, f_demod);
    const psi_corr = psi_iq1 + Math.PI * f_demod * T;           // 参考无关相位(demod 时间原点=帧起点, 帧中点在本地时间 T/2)
    let fe;
    if (n === 0) fe = f_coarse;
    else {
      const dpsi_c = wpi(psi_corr - psi_corr_prev - 2 * Math.PI * f_est_prev * T);
      const fe_c = f_est_prev + dpsi_c / (2 * Math.PI * T);
      if (Math.abs(f_coarse - f_est_prev) <= 1e4) fe = fe_c;    // 连续跟踪: 半bin±π模糊用上帧估计解除
      else {                                                     // 频率阶跃(>1 bin) → 粗频重捕获
        const dpsi_a = wpi(psi_corr - psi_corr_prev - 2 * Math.PI * f_coarse * T);
        fe = f_coarse + dpsi_a / (2 * Math.PI * T);
      }
    }
    psi_corr_prev = psi_corr;
    f_est_prev = fe;
    const f_res = fe - f_track[n];
    let df;
    if (mode === 'PLLwrap') {
      const psi_iq2 = demod(zRe, zIm, fe);
      psi_w[n] = wpi(psi_iq2 + Math.PI * fe * T - 2 * Math.PI * f_track[n] * t_mid);
      df = Kp_hz * (180 / Math.PI) * psi_w[n];
    } else if (mode === 'FLL') {
      df = Kp_hz * 360 * f_res * T;
    } else {
      // 完善版: II 型锁相环 — NCO 频率处 IQ 解调鉴相 + NCO 相位累加
      const psi_pd = demod(zRe, zIm, f_track[n]);
      const e_pll = wpi(psi_pd - (phi_nco + 2 * Math.PI * f_track[n] * T / 2));
      psi_w[n] = e_pll;
      phi_nco += 2 * Math.PI * f_track[n] * T + Kp_pll * e_pll;
      df = Kp_hz * 360 * f_res * T;
      if (Math.abs(f_res) < 1e4) df += Kf_pll * e_pll;
    }
    f_track[n + 1] = f_track[n] + df;
    f_est[n] = fe; f_est_prev = fe;
    const fd = fdMode === 'const' ? fd0 : 4 * Math.PI * fv * A / lambda * Math.cos(2 * Math.PI * fv * t_mid);
    f_true[n] = IF0 + fd;
  }
  return { f_track, f_est, f_true, psi_w };
}

function stats(f_track, f_true, from) {
  const e = []; for (let n = from; n < f_true.length; n++) e.push(f_track[n + 1] - f_true[n]);
  const mean = e.reduce((a, b) => a + b, 0) / e.length;
  const sd = Math.sqrt(e.reduce((a, b) => a + (b - mean) ** 2, 0) / e.length);
  const pp = Math.max(...e) - Math.min(...e);
  return { mean, sd, pp };
}

console.log('== 测试2 环路模式对比 (f_d=100kHz, SNR=40dB, 300帧) ==');
for (const [mode, name] of [['PLLwrap', '原稿字面 PLLwrap'], ['FLL', '解缠修正 FLL'], ['FLLPLL', '完善版 II型PLL']]) {
  const S = runCase({ fdMode: 'const', fd0: 1e5, Nf: 300, mode, SNRdB: 40 });
  const st = stats(S.f_track, S.f_true, 100);
  console.log(`  ${name}: 稳态误差 均值 ${st.mean.toFixed(1)} Hz, 峰峰 ${st.pp.toFixed(1)} Hz, σ ${st.sd.toFixed(1)} Hz`);
}

console.log('== 测试1 全范围阶跃 (FLLPLL, SNR=40dB, 120帧) ==');
for (const fd0 of [0, 10, 100, 1e3, 1e4, 1e5, 1e6, -1e6]) {
  const S = runCase({ fdMode: 'const', fd0, Nf: 120, mode: 'FLLPLL', SNRdB: 40 });
  const e = []; for (let n = 90; n < 120; n++) e.push(S.f_est[n] - S.f_true[n]);
  const bias = e.reduce((a, b) => a + b, 0) / e.length;
  const rmse = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / e.length);
  console.log(`  f_d=${String(fd0).padStart(8)} Hz: f_est 偏差 ${bias.toFixed(1)} Hz, RMS ${rmse.toFixed(1)} Hz`);
}

console.log('== 准静态阈值验证: f_est 帧率估计器 (FLLPLL, SNR=40dB, 500帧) ==');
for (const [fv, A] of [[100, 10e-6], [1e3, 1e-6], [1e3, 10e-6]]) {
  const S = runCase({ fdMode: 'sine', A, fv, Nf: 500, mode: 'FLLPLL', SNRdB: 40 });
  const fd_pk = 4 * Math.PI * fv * A / lambda;
  const slope = 2 * Math.PI * fv * fd_pk;
  const sincv = Math.sin(Math.PI * fv * T) / (Math.PI * fv * T);
  let sse = 0;
  for (let n = 0; n < 500; n++) {
    const truth = fd_pk * sincv * Math.cos(2 * Math.PI * fv * n * T);
    sse += (S.f_est[n] - IF0 - truth) ** 2;
  }
  console.log(`  fv=${fv}Hz A=${A * 1e6}um: fd峰值=${(fd_pk / 1e3).toFixed(1)}kHz, FM斜率=${(slope / 1e6).toFixed(0)}MHz/s (${(slope * T / 1e3).toFixed(1)}kHz/帧), RMS=${Math.sqrt(sse / 500).toFixed(1)} Hz`);
}

console.log('== 直接相位解调测振 (混频至 IF0 → 1MHz 降采样 → 解缠 → x=λφ/4π, SNR=40dB) ==');
{
  const BLK = Math.round(N / 100), NBLK = Math.floor(N / BLK);   // 帧内 100 个块 ≈ 1us
  const zRe = new Float64Array(N), zIm = new Float64Array(N);
  for (const [fv, A] of [[1e3, 10e-6], [1e4, 1.23e-6]]) {
    const fd_pk = 4 * Math.PI * fv * A / lambda;
    const Nf = 100;
    const t0 = 0.5 * BLK / fs;           // 测量起点=首块中点: 相位解调只能测相对位移(AC), 与 AC 参考比较
    let prev = NaN, unw = 0, sse = 0, cnt = 0;
    for (let n = 0; n < Nf; n++) {
      const tn = n * T;
      for (let i = 0; i < N; i++) {
        const t = tn + tfr[i];
        const pm = 4 * Math.PI * A / lambda * Math.sin(2 * Math.PI * fv * t);
        zRe[i] = Math.cos(pm + phi0); zIm[i] = Math.sin(pm + phi0);
      }
      const s = Math.sqrt(Math.pow(10, -4) / 2);
      for (let i = 0; i < N; i++) { zRe[i] += s * gauss(); zIm[i] += s * gauss(); }
      for (let b = 0; b < NBLK; b++) {
        let sr = 0, si = 0;
        for (let i = b * BLK; i < (b + 1) * BLK; i++) { sr += zRe[i]; si += zIm[i]; }
        const ps = Math.atan2(si, sr);
        if (isNaN(prev)) prev = ps;
        unw += wpi(ps - prev); prev = ps;
        const t = tn + (b + 0.5) * BLK / fs;
        sse += (lambda * unw / (4 * Math.PI) - A * (Math.sin(2 * Math.PI * fv * t) - Math.sin(2 * Math.PI * fv * t0))) ** 2; cnt++;
      }
    }
    console.log(`  fv=${fv}Hz (fd峰值=${(fd_pk / 1e3).toFixed(1)}kHz): x(t) 重构 RMS = ${(Math.sqrt(sse / cnt) * 1e9).toFixed(2)} nm`);
  }
}

console.log('== 测试6 多普勒过零 (线性扫频 -600k→+600kHz, 400帧, 30MHz/s) ==');
{
  const fd0 = -600e3, fd1 = 600e3, Nf = 400;
  const mu = (fd1 - fd0) / (Nf * T);
  const zRe = new Float64Array(N), zIm = new Float64Array(N);
  const fRe = new Float64Array(N), fIm = new Float64Array(N);
  const f_est = new Float64Array(Nf);
  let psi_corr_prev = 0, f_demod = 0, f_est_prev = 0;
  for (let n = 0; n < Nf; n++) {
    const tn = n * T, t_mid = tn + T / 2;
    for (let i = 0; i < N; i++) {
      const t = tn + tfr[i];
      const ph = 2 * Math.PI * (IF0 * t + fd0 * t + 0.5 * mu * t * t) + phi0;
      zRe[i] = Math.cos(ph); zIm[i] = Math.sin(ph);
    }
    const s = Math.sqrt(Math.pow(10, -4) / 2);
    for (let i = 0; i < N; i++) { zRe[i] += s * gauss(); zIm[i] += s * gauss(); }
    for (let i = 0; i < N; i++) { fRe[i] = zRe[i]; fIm[i] = zIm[i]; }
    fft(fRe, fIm);
    let k = 0, mx = -1;
    for (let i = 0; i < N; i++) { const m = fRe[i] * fRe[i] + fIm[i] * fIm[i]; if (m > mx) { mx = m; k = i; } }
    let f_coarse = k * df_bin; if (f_coarse > fs / 2) f_coarse -= fs;
    if (n === 0) f_demod = f_coarse;
    else f_demod = f_est_prev;
    const psi_iq1 = demod(zRe, zIm, f_demod);
    const psi_corr = psi_iq1 + Math.PI * f_demod * T;
    let fe;
    if (n === 0) fe = f_coarse;
    else {
      const dpsi_c = wpi(psi_corr - psi_corr_prev - 2 * Math.PI * f_est_prev * T);
      const fe_c = f_est_prev + dpsi_c / (2 * Math.PI * T);
      if (Math.abs(f_coarse - f_est_prev) <= 1e4) fe = fe_c;
      else {
        const dpsi_a = wpi(psi_corr - psi_corr_prev - 2 * Math.PI * f_coarse * T);
        fe = f_coarse + dpsi_a / (2 * Math.PI * T);
      }
    }
    psi_corr_prev = psi_corr;
    f_est_prev = fe;
    f_est[n] = fe;
  }
  let sse = 0;
  for (let n = 0; n < Nf; n++) {
    const truth = IF0 + fd0 + mu * n * T;   // 线性扫频: 帧平均频率 = 帧起始瞬时频率
    sse += (f_est[n] - truth) ** 2;
  }
  console.log(`  过零(扫频)跟踪 RMS = ${Math.sqrt(sse / Nf).toFixed(1)} Hz`);
}

console.log('== 测试5 噪声性能 (f_d=100kHz, 200帧, σ取后100帧) ==');
for (const snr of [0, 10, 20, 30, 40]) {
  const S = runCase({ fdMode: 'const', fd0: 1e5, Nf: 200, mode: 'FLLPLL', SNRdB: snr });
  const e = []; for (let n = 100; n < 200; n++) e.push(S.f_est[n] - S.f_true[n]);
  const sd = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / e.length);
  console.log(`  SNR=${String(snr).padStart(2)} dB: σ_fest = ${sd.toFixed(1)} Hz`);
}

console.log('== 测试7 测距 (R=75m, DA0 扫频 1MHz, 300帧, SNR=40dB) ==');
{
  const R = 75, dF = 1e6, Nf = 300;
  const rem = new Float64Array(Nf), fDA0 = new Float64Array(Nf);
  const zRe = new Float64Array(N), zIm = new Float64Array(N);
  for (let n = 0; n < Nf; n++) {
    fDA0[n] = n * dF / (Nf - 1);           // 相对 f_base 的扫频偏移
    const tn = n * T, IFn = fDA0[n];
    for (let i = 0; i < N; i++) {
      const t = tn + tfr[i];
      const ph = 2 * Math.PI * IFn * t + 4 * Math.PI * R * (IFn + 40e6) / c0 + phi0;
      zRe[i] = Math.cos(ph); zIm[i] = Math.sin(ph);
    }
    const s = Math.sqrt(Math.pow(10, -4) / 2);
    for (let i = 0; i < N; i++) { zRe[i] += s * gauss(); zIm[i] += s * gauss(); }
    // 在已知扫频 IF 处解调(残差为零): 解调相位 = 帧起始相位
    const psi_iq2 = demod(zRe, zIm, IFn);
    rem[n] = wpi(psi_iq2 - 2 * Math.PI * IFn * tn);
  }
  const phiArr = new Float64Array(Nf); phiArr[0] = rem[0];
  for (let n = 1; n < Nf; n++) phiArr[n] = phiArr[n - 1] + wpi(rem[n] - rem[n - 1]);
  let xm = 0, ym = 0;
  for (let n = 0; n < Nf; n++) { xm += fDA0[n]; ym += phiArr[n]; }
  xm /= Nf; ym /= Nf;
  let num = 0, den = 0;
  for (let n = 0; n < Nf; n++) { num += (fDA0[n] - xm) * (phiArr[n] - ym); den += (fDA0[n] - xm) ** 2; }
  const Rhat = (num / den) * c0 / (4 * Math.PI);
  console.log(`  R_hat = ${Rhat.toFixed(4)} m, 误差 = ${(Rhat - R).toFixed(3)} m`);
}
console.log('done');
