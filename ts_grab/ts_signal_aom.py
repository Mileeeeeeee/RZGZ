"""
方案 B: 40 MHz AOM 信号处理程序
================================

系统框图:
    1550nm 光源
        │
    4G/5G 调制
        │
    50/50 分光 ──┬─ Reference  ── 40M AOM ──────────────┐
                 └─ Measurment ── Delay t ── Target ── Return ─┘
                                                │
                  50/50 合束 → PD → ADC → FFT → IQ 解调 → I/Q
                                                │
                          Amplitude / Phase → Distance / Vibration

物理原理:
  - 参考臂经 40 MHz AOM 移频, 测量臂经 Delay t 打到 Target 后返回,
    两臂在 50/50 合束器处干涉, PD 输出 40 MHz 拍频信号;
  - ADC 采集拍频电压后, FFT 观察 40 MHz 载波及两侧振动边带;
  - IQ 解调 (正交数字下变频): 与 40 MHz 本振混频 + 低通滤波 + 抽取,
    得到复包络 z(t) = I(t) + j·Q(t);
  - 幅度 A(t) = |z(t)|: 携带 4G/5G 强度调制的包络相位 cos(2π·f_mod·τ),
    反解 τ 可得距离估计 L = c·τ/2 (模糊区间 c/(2·f_mod));
  - 相位 φ(t) = ∠z(t): 对应两臂光程差 (双程), ΔL = λ·φ/(4π),
    减去慢漂移趋势分离振动位移 (nm 级), 求导得振动速度, Welch 谱估计振动频率。

处理流程:
  1. 数据读取   : bin/CSV 码值 → 电压 (复用 ts_signal_fft 读取模块)
  2. FFT        : 以 AOM 频率为中心的全时段频谱 + 时频瀑布图,
                  提取载波峰值与两侧振动边带 (交叉验证振动频率)
  3. IQ 解调    : 数字下变频 (NCO × 低通 × 抽取) → I(t), Q(t)
  4. 幅度/相位  : A(t) = √(I²+Q²), φ(t) = unwrap(arctan2(Q,I)),
                  并补偿 AOM 实际频率与标称频率的残余频偏
  5. 距离/振动  : 相位通道 → 振动位移/速度/振动频率 (Welch 谱);
                  幅度通道 → 调制包络相位距离估计 (注明模糊区间)
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

import ts_signal_fft as fft_mod             # 复用数据读取/FFT 频谱分析模块与版本化参数

# 中文标注字体支持 (Windows 常见中文字体，缺失则回退默认字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 版本参数 =================
# 参数值统一定义在 ts_signal_fft.py 的 def 模块中, 不同版本用各自的参数 (get_params(version))
cfg = fft_mod.get_params('aom')

# ================= 方案 B 系统参数 =================
LASER_WAVELENGTH_M = cfg.LASER_WAVELENGTH_M   # 光源波长 (m)
SPEED_OF_LIGHT = cfg.SPEED_OF_LIGHT           # 光速 (m/s)
MOD_FREQ_HZ = cfg.MOD_FREQ_HZ                 # 激光强度调制频率 (Hz), 4G/5G 可切换
MOD_FREQ_HZ_ALT = cfg.MOD_FREQ_HZ_ALT         # 备用调制频率 (4 GHz), 与 5 GHz 差频解距离模糊
MOD_DEPTH = cfg.MOD_DEPTH                     # 强度调制深度 m (0~1)

# ================= AOM / IQ 解调参数 =================
AOM_FREQ_HZ = cfg.AOM_FREQ_HZ                 # AOM 移频量 = IQ 解调本振频率 (Hz)
IQ_LPF_CUTOFF_HZ = cfg.IQ_LPF_CUTOFF_HZ       # 低通滤波器截止频率 (Hz), 需覆盖最大振动频率并抑制 2×AOM 混频分量
IQ_LPF_ORDER = cfg.IQ_LPF_ORDER               # 低通滤波器阶数 (二阶节级联)
IQ_DECIM_FACTOR = cfg.IQ_DECIM_FACTOR         # 抽取因子: 解调后采样率 = fs / IQ_DECIM_FACTOR
IQ_CHUNK_SAMPLES = cfg.IQ_CHUNK_SAMPLES       # 解调分块采样点数 (内存受限时减小)
IQ_SETTLE_SAMPLES = cfg.IQ_SETTLE_SAMPLES     # 丢弃低通滤波器起始瞬态的解调输出样本数
RESIDUAL_CARRIER_SEARCH_HZ = cfg.RESIDUAL_CARRIER_SEARCH_HZ   # 残余载波频偏搜索范围 (Hz), 覆盖 AOM 实际频率偏差

# ================= FFT 阶段参数 =================
TIME_BLOCK_DURATION = cfg.TIME_BLOCK_DURATION   # FFT 时间块长度 (s), 10 ms → 100 Hz 频率分辨率 (100MHz 采样率下)
FREQ_SPAN = cfg.FREQ_SPAN                      # 频谱分析范围: AOM 频率 ± FREQ_SPAN (Hz), 覆盖载波漂移 + 最大振动频率
OVERLAP_RATIO = cfg.OVERLAP_RATIO              # 相邻时间块重叠比例
WINDOW_TYPE = cfg.WINDOW_TYPE                  # FFT 窗函数
DETREND_METHOD = cfg.DETREND_METHOD            # 时间块去趋势方法

# ================= 振动/距离参数 =================
VIB_HIGHPASS_HZ = cfg.VIB_HIGHPASS_HZ          # 振动位移高通截止 (Hz), 去除相位慢漂移(激光频率漂移等)
VIB_LPF_HZ = cfg.VIB_LPF_HZ                    # 振动位移低通截止 (Hz)
VIB_BANDPASS_ORDER = cfg.VIB_BANDPASS_ORDER    # 振动趋势低通/噪声低通滤波器阶数
WELCH_NPERSEG = cfg.WELCH_NPERSEG              # Welch 振动谱每段点数


def _fmt_freq(hz):
    """按频谱仪显示习惯格式化频率"""
    return fft_mod._fmt_freq(hz)


# ================= Stage 3: IQ 解调 (数字下变频) =================

def iq_demodulate(data, fs, lo_freq, lpf_cutoff=IQ_LPF_CUTOFF_HZ,
                  lpf_order=IQ_LPF_ORDER, decim=IQ_DECIM_FACTOR,
                  chunk=IQ_CHUNK_SAMPLES, settle=IQ_SETTLE_SAMPLES):
    """
    IQ 解调 (正交数字下变频):

        I(t) = LPF{ x(t) · cos(2π·f_LO·t) }
        Q(t) = LPF{ x(t) · (-sin(2π·f_LO·t)) }

    x(t) = A(t)·cos(2π·f_LO·t + φ(t)) 时, 复包络 z = I + jQ = (A/2)·e^{jφ},
    幅度与相位即混频后低通输出。滤波器状态 (zi) 跨块传递, 保证分块处理
    与整体处理结果一致; 抽取按整数倍进行, 输出采样率 = fs / decim。

    参数:
        data:       输入电压数据 (1D, float32)
        fs:         采样率 (Hz)
        lo_freq:    本振频率 (Hz), 即 AOM 移频量
        lpf_cutoff: 低通截止频率 (Hz)
        lpf_order:  低通滤波器阶数
        decim:      抽取因子
        chunk:      分块采样点数
        settle:     丢弃的低通滤波器起始瞬态样本数 (解调后采样率计)

    返回:
        dict: z(复包络 complex128), i(同相分量), q(正交分量),
              t(时间轴 s), fs(解调后采样率 Hz)
    """
    n = len(data)
    n_out = n // decim
    if n_out < 1:
        raise ValueError(f"数据长度 ({n}) 小于抽取因子 ({decim}), 无法解调")

    sos = signal.butter(lpf_order, lpf_cutoff, btype='low', fs=fs, output='sos')
    zi_i = signal.sosfilt_zi(sos)
    zi_q = signal.sosfilt_zi(sos)

    i_out = np.empty(n_out, dtype=np.float32)
    q_out = np.empty(n_out, dtype=np.float32)
    omega = 2.0 * np.pi * lo_freq / fs
    pos = 0

    print(f"IQ 解调: 本振 {_fmt_freq(lo_freq)}, 低通截止 {lpf_cutoff} Hz "
          f"({lpf_order} 阶), 抽取 {decim}x → 输出采样率 {fs/decim/1000:.1f} kHz")
    with tqdm(total=n_out, desc="IQ 解调进度", unit="点") as pbar:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            end -= (end - start) % decim          # 截取为 decim 整数倍
            if end <= start:
                break
            seg = data[start:end]
            idx = np.arange(start, end)

            mix_i = seg * np.cos(omega * idx)     # 同相混频
            mix_q = -seg * np.sin(omega * idx)    # 正交混频

            filt_i, zi_i = signal.sosfilt(sos, mix_i, zi=zi_i)
            filt_q, zi_q = signal.sosfilt(sos, mix_q, zi=zi_q)

            m = filt_i.size // decim
            i_out[pos:pos + m] = filt_i[:m * decim:decim]
            q_out[pos:pos + m] = filt_q[:m * decim:decim]
            pos += m
            pbar.update(m)

    # 丢弃低通滤波器起始瞬态 (滤波器建立时间内的输出不可靠)
    if settle > 0:
        settle = min(settle, max(0, pos - 1))
        i_out = i_out[settle:pos]
        q_out = q_out[settle:pos]
        pos -= settle

    fs_out = fs / decim
    t_axis = (np.arange(pos) + settle) / fs_out
    return {
        'z': (i_out + 1j * q_out).astype(np.complex128),
        'i': i_out,
        'q': q_out,
        't': t_axis,
        'fs': fs_out,
    }


# ================= Stage 4: 幅度/相位计算 =================

def correct_residual_carrier(demod, search_hz=RESIDUAL_CARRIER_SEARCH_HZ):
    """
    估计并补偿 AOM 实际频率与标称频率 (IQ 本振) 的残余频偏

    AOM 实际移频量可能与标称值存在偏差 Δf, 表现为复包络 z(t) 的缓慢旋转
    z(t) = A(t)·e^{j(2π·Δf·t + φ(t))}。

    估计方法: 半程重叠相位斜率中位数法 (Theil-Sen 变体)。取相隔半个记录
    长度的样本对的相位斜率, 取其中位数, 除以 2π 即频偏:
        slope_i = (φ[i + n/2] − φ[i]) / (T/2)
    振动正弦相位对斜率的扰动关于真实斜率对称分布, 中位数不受其影响
    (f_vib·T/2 为整数时振动贡献精确抵消), 对噪声与相位解缠绕的偶发
    2π 跳变 (中位数对离群值稳健) 也不敏感; 且强振动下边带能量高于
    载波 (J1 > J0) 不影响估计 (频谱峰值法会锁定到边带)。
    估计后对 z(t) 反向旋转消除 Δf, 共两轮迭代提高精度。

    返回:
        z_corr: 补偿后的复包络
        df:     残余频偏估计值 (Hz)
    """
    z = demod['z']
    t = demod['t']
    n = z.size
    half = n // 2

    df = 0.0
    zz = z
    for _ in range(2):   # 两轮迭代: 首轮粗估, 反向旋转后复估残余
        phase = np.unwrap(np.angle(zz))
        slopes = (phase[half:] - phase[:half]) / (t[half:] - t[:half])
        delta = float(np.median(slopes)) / (2.0 * np.pi)
        df += delta
        zz = zz * np.exp(-2j * np.pi * delta * t)

    if abs(df) > search_hz:
        print(f"[警告] 残余频偏估计 {df:.2f} Hz 超出搜索范围 ±{search_hz} Hz, 载波锁定可能异常")

    if df != 0.0:
        z_corr = z * np.exp(-2j * np.pi * df * demod['t'])
    else:
        z_corr = z
    print(f"残余载波频偏估计: {df:+.2f} Hz (AOM 实际频率 {_fmt_freq(AOM_FREQ_HZ + df)})")
    return z_corr, df


def amplitude_to_distance(amp, mod_freq=MOD_FREQ_HZ, mod_depth=MOD_DEPTH):
    """
    幅度通道 → 调制包络相位 → 距离估计

    PD 拍频幅度与两臂调制包络乘积的直流项成正比:
        A(τ) ∝ |1 + (m²/2)·cos(2π·f_mod·τ)|
    按最大值归一化后反解:
        cos(θ) = (A_norm·(1 + m²/2) − 1) · (2/m²),  θ = 2π·f_mod·τ ∈ [0, π]
        L = c·τ / 2
    距离模糊区间: ΔL = c / (2·f_mod) (5 GHz → 30 mm; 4 GHz → 37.5 mm),
    双频 (4G/5G 差频 1 GHz) 可把无模糊区间扩展为 c / (2·|f2−f1|) = 150 mm。

    注: 该距离为归一化相对估计, 绝对定标需已知最大幅度对应的参考距离。
    """
    amp_ref = float(np.max(amp))
    if amp_ref <= 0:
        return np.zeros_like(amp)
    a_norm = amp / amp_ref
    k = mod_depth ** 2 / 2.0
    cosv = np.clip((a_norm * (1.0 + k) - 1.0) / k, -1.0, 1.0)
    tau = np.arccos(cosv) / (2.0 * np.pi * mod_freq)
    return SPEED_OF_LIGHT * tau / 2.0


def compute_amplitude_phase(z, t_axis, fs):
    """
    幅度/相位计算:
      - 幅度 A(t) = |z(t)|
      - 相位 φ(t) = unwrap(arctan2(Q, I))
      - 光程差 ΔL(t) = λ·φ(t)/(4π)     (双程: 相位变化 2π 对应 λ/2)
      - 振动位移: ΔL 减去慢漂移趋势 (等效高通, 去除激光频率漂移等), 再低通抑制噪声
      - 振动速度: 振动位移对时间求导
      - 距离估计: 幅度通道 (调制包络相位)

    返回:
        dict: amp, phase, disp, vib_disp, vel, dist_amp
    """
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))

    # 双程光程差: Δφ = 2π·2ΔL/λ  →  ΔL = λ·φ/(4π)
    disp = phase * LASER_WAVELENGTH_M / (4.0 * np.pi)

    # 振动位移 = 位移 − 慢漂移趋势 (等效高通), 再低通抑制噪声。
    # 直接对含大 DC/漂移的信号做高通滤波, 零相位滤波 (sosfiltfilt) 的边缘
    # 反射会在整个记录激起高通瞬态振铃; 改为低通提取趋势再相减: 低通对
    # DC/慢漂移平滑通过, 无高通振铃问题。滤波前先按滤波器记忆长度
    # (3/f_c) 对称延拓, 否则低通在记录边缘只对振动的部分周期做平均,
    # 趋势在边缘失真 (表现为边缘处振动幅值被吞掉)。
    sos_trend = signal.butter(VIB_BANDPASS_ORDER, VIB_HIGHPASS_HZ,
                              btype='lowpass', fs=fs, output='sos')
    pad_len = min(int(3.0 * fs / VIB_HIGHPASS_HZ), disp.size)
    trend = signal.sosfiltfilt(sos_trend,
                               np.pad(disp, pad_len, mode='symmetric'))[pad_len:-pad_len]
    vib_disp = disp - trend

    sos_lp = signal.butter(VIB_BANDPASS_ORDER, VIB_LPF_HZ,
                           btype='lowpass', fs=fs, output='sos')
    pad_len = min(int(3.0 * fs / VIB_LPF_HZ), vib_disp.size)
    vib_disp = signal.sosfiltfilt(sos_lp,
                                  np.pad(vib_disp, pad_len, mode='symmetric'))[pad_len:-pad_len]

    # 振动速度: 位移对时间求导
    vel = np.gradient(vib_disp, 1.0 / fs)

    # 幅度通道距离估计 (调制包络相位)
    dist_amp = amplitude_to_distance(amp)

    return {
        'amp': amp,
        'phase': phase,
        'disp': disp,
        'vib_disp': vib_disp,
        'vel': vel,
        'dist_amp': dist_amp,
    }


def vibration_spectrum(vib_disp, fs):
    """
    Welch 功率谱估计振动频谱, 返回频率轴/PSD/峰值振动频率

    零填充 (nfft = 8×nperseg) 细化频率网格, 再对峰值做抛物线插值,
    使振动频率定位精度优于 1 Hz (直接取最大 bin 时精度受限于
    fs/nperseg 的谱分辨率)。
    """
    n = vib_disp.size
    nperseg = min(WELCH_NPERSEG, n)
    if nperseg < 8:
        raise ValueError(f"振动位移序列过短 ({n}), 无法估计振动频谱")
    f_axis, psd = signal.welch(vib_disp, fs=fs, nperseg=nperseg,
                               nfft=8 * nperseg, detrend='constant')

    k = int(np.argmax(psd))
    if 0 < k < psd.size - 1:
        a1, a0, a2 = psd[k - 1], psd[k], psd[k + 1]
        denom = a1 + a2 - 2 * a0
        if abs(denom) > 1e-30:
            off = -0.5 * (a2 - a1) / denom
            off = max(-0.5, min(0.5, off))
            vib_freq = float(f_axis[k] + off * (f_axis[1] - f_axis[0]))
        else:
            vib_freq = float(f_axis[k])
    else:
        vib_freq = float(f_axis[k])
    return f_axis, psd, vib_freq


# ================= 绘图 =================

def plot_iq(demod, save_path=None):
    """绘制 IQ 解调输出的同相/正交分量"""
    t, i, q = demod['t'], demod['i'], demod['q']
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, i, linewidth=cfg.LINE_WIDTH, color='tab:blue')
    axes[0].set_ylabel('I (V)')
    axes[0].set_title(f'IQ 解调: 同相分量 I (本振 {_fmt_freq(AOM_FREQ_HZ)}, '
                      f'LPF {IQ_LPF_CUTOFF_HZ/1000:.0f} kHz, {demod["fs"]/1000:.0f} kHz 输出)')
    axes[0].grid(True, alpha=cfg.GRID_ALPHA)

    axes[1].plot(t, q, linewidth=cfg.LINE_WIDTH, color='tab:orange')
    axes[1].set_ylabel('Q (V)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('IQ 解调: 正交分量 Q')
    axes[1].grid(True, alpha=cfg.GRID_ALPHA)

    plt.tight_layout()
    if save_path:
        fig_path = f'{save_path}iq{cfg.SAVE_SUFFIX}'
        plt.savefig(fig_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'IQ 解调图已保存到: {fig_path}')
    return fig


def plot_amplitude_phase(res, demod, save_path=None):
    """绘制幅度/相位: A(t), φ(t) (unwrap), 光程差 ΔL(t)"""
    t = demod['t']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, res['amp'], linewidth=cfg.LINE_WIDTH, color='tab:red')
    ax.set_ylabel('Amplitude A (V)')
    ax.set_title(f'幅度 A = √(I²+Q²) (AOM 拍频包络, 携带 4G/5G 调制相位)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    ax = axes[1]
    ax.plot(t, res['phase'], linewidth=cfg.LINE_WIDTH, color='tab:blue')
    ax.set_ylabel('Phase φ (rad)')
    ax.set_title('相位 φ = unwrap(arctan2(Q, I)) (对应双程光程差)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    ax = axes[2]
    ax.plot(t, res['disp'] * 1e6, linewidth=cfg.LINE_WIDTH, color='tab:green')
    ax.set_ylabel('ΔL (μm)')
    ax.set_xlabel('Time (s)')
    ax.set_title('光程差 ΔL = λ·φ/(4π) (λ = 1550 nm, 双程)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    plt.tight_layout()
    if save_path:
        fig_path = f'{save_path}amplitude_phase{cfg.SAVE_SUFFIX}'
        plt.savefig(fig_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'幅度/相位图已保存到: {fig_path}')
    return fig


def plot_distance_vibration(res, demod, f_vib, psd_vib, vib_freq, save_path=None):
    """绘制距离/振动: 振动位移, 振动速度, 振动频谱, 距离估计"""
    t = demod['t']
    amb = SPEED_OF_LIGHT / (2.0 * MOD_FREQ_HZ)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    ax = axes[0]
    ax.plot(t, res['vib_disp'] * 1e9, linewidth=cfg.LINE_WIDTH, color='tab:green')
    ax.set_ylabel('Displacement (nm)')
    ax.set_title(f'振动位移 (相位通道, 趋势相减 >{VIB_HIGHPASS_HZ:.0f} Hz, 低通 <{VIB_LPF_HZ/1000:.0f} kHz)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    ax = axes[1]
    ax.plot(t, res['vel'] * 1e3, linewidth=cfg.LINE_WIDTH, color='tab:orange')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('振动速度 (位移对时间求导)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    ax = axes[2]
    psd_db = 10 * np.log10(psd_vib + 1e-30)
    ax.plot(f_vib, psd_db, linewidth=cfg.LINE_WIDTH, color='tab:blue')
    ax.axvline(x=vib_freq, color='red', linestyle='--', alpha=0.6)
    ax.annotate(f'{vib_freq:.1f} Hz', xy=(vib_freq, psd_db.max()),
                xytext=(6, 0), textcoords='offset points',
                fontsize=cfg.SPECTRUM_ANNOTATION_FONT_SIZE + 2, color='red')
    ax.set_ylabel('PSD (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('振动频谱 (Welch)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    ax = axes[3]
    ax.plot(t, res['dist_amp'] * 1e3, linewidth=cfg.LINE_WIDTH, color='tab:purple')
    ax.set_ylabel('Distance (mm)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'距离估计 (幅度通道, {MOD_FREQ_HZ/1e9:g} GHz 调制, 模糊区间 {amb*1e3:.1f} mm)')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    plt.tight_layout()
    if save_path:
        fig_path = f'{save_path}distance_vibration{cfg.SAVE_SUFFIX}'
        plt.savefig(fig_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'距离/振动图已保存到: {fig_path}')
    return fig


# ================= 统计信息 =================

def collect_statistics(bin_file, result, channels, fs, spectrogram_data,
                       demod, res,
                       df_res, f_vib, psd_vib, vib_freq):
    """汇总各阶段统计信息，返回文本行列表（控制台打印与 txt 保存共用）"""
    lines = []
    sep = "=" * 60
    lines += [sep, "方案 B: 40 MHz AOM 信号分析统计", sep]
    lines.append(f"文件: {bin_file}")
    lines.append(f"采样点数: {result['num_samples']:,}")
    lines.append(f"采样率: {fs:,.0f} Hz ({fs/1e6:g} MHz)")
    lines.append(f"通道数: {len(channels)} (分析通道: {channels[0]})")
    lines.append(f"数据时长: {result['num_samples']/fs:.3f} s")

    # ---- FFT 阶段 ----
    lines += ["", sep, "Stage 2: FFT (频谱分析)", sep]
    lines.append(f"AOM 频率(频谱中心): {_fmt_freq(AOM_FREQ_HZ)}, 频移范围 ±{FREQ_SPAN/1000:.1f} kHz")
    lines.append(f"时间块: {TIME_BLOCK_DURATION*1000:.0f} ms × {spectrogram_data['n_blocks']:,} 块, "
                 f"重叠 {OVERLAP_RATIO*100:.0f}%, 分析频点数 {spectrogram_data['n_freqs']:,}")

    # ---- IQ 解调阶段 ----
    lines += ["", sep, "Stage 3: IQ 解调", sep]
    lines.append(f"本振频率: {_fmt_freq(AOM_FREQ_HZ)}")
    lines.append(f"低通截止: {IQ_LPF_CUTOFF_HZ/1000:.1f} kHz ({IQ_LPF_ORDER} 阶), "
                 f"抽取 {IQ_DECIM_FACTOR}x, 解调后采样率 {demod['fs']/1000:.1f} kHz, "
                 f"点数 {demod['z'].size:,}")
    lines.append(f"残余载波频偏: {df_res:+.2f} Hz → AOM 实际频率 {_fmt_freq(AOM_FREQ_HZ + df_res)}")

    # ---- 幅度/相位 ----
    amp, phase = res['amp'], res['phase']
    lines += ["", sep, "Stage 4: 幅度/相位", sep]
    lines.append(f"幅度 A: 均值={amp.mean():.4f} V, 标准差={amp.std():.4f} V, "
                 f"范围=[{amp.min():.4f}, {amp.max():.4f}] V")
    lines.append(f"相位 φ: 均值={phase.mean():.2f} rad, 标准差={phase.std():.2f} rad, "
                 f"范围=[{phase.min():.2f}, {phase.max():.2f}] rad")
    slope = float(np.polyfit(demod['t'], phase, 1)[0])          # rad/s
    drift_rate = slope * LASER_WAVELENGTH_M / (4.0 * np.pi)      # m/s
    lines.append(f"相位斜率: {slope:.3f} rad/s → 平均距离变化速率 {drift_rate*1e9:.2f} nm/s")

    # ---- 距离/振动 ----
    lines += ["", sep, "Stage 5: 距离/振动", sep]
    lines.append(f"振动频率 (Welch 谱峰值): {vib_freq/1000:.3f} kHz ({vib_freq:.1f} Hz)")
    lines.append(f"振动位移: 峰峰值={np.ptp(res['vib_disp'])*1e9:.1f} nm, "
                 f"标准差={res['vib_disp'].std()*1e9:.1f} nm, "
                 f"范围=[{res['vib_disp'].min()*1e9:.1f}, {res['vib_disp'].max()*1e9:.1f}] nm")
    lines.append(f"振动速度: 峰峰值={np.ptp(res['vel'])*1e3:.4f} mm/s, "
                 f"标准差={res['vel'].std()*1e3:.4f} mm/s")
    amb = SPEED_OF_LIGHT / (2.0 * MOD_FREQ_HZ)
    amb_dual = SPEED_OF_LIGHT / (2.0 * abs(MOD_FREQ_HZ - MOD_FREQ_HZ_ALT))
    d = res['dist_amp']
    lines.append(f"调制频率: {MOD_FREQ_HZ/1e9:g} GHz, 距离模糊区间: {amb*1e3:.1f} mm")
    lines.append(f"距离估计 (幅度通道): 均值={d.mean()*1e3:.2f} mm, "
                 f"范围=[{d.min()*1e3:.2f}, {d.max()*1e3:.2f}] mm")
    lines.append(f"提示: 幅度通道距离为归一化相对估计; 双频 (4G/5G, 差频 "
                 f"{abs(MOD_FREQ_HZ-MOD_FREQ_HZ_ALT)/1e9:g} GHz) 可将无模糊区间扩展为 {amb_dual*1e3:.1f} mm")
    lines.append(sep)
    return lines


def save_statistics_txt(save_path, lines):
    """将统计信息写入 txt 文件并打印到控制台"""
    text = "\n".join(lines) + "\n"
    txt_path = save_path + cfg.STATS_SUFFIX
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(text)
    print(text)
    print(f"统计信息已保存到: {txt_path}")
    return txt_path


# ================= 主流程 =================

def run_analysis(bin_file):
    """方案 B 完整流程: 数据读取 → FFT → IQ 解调 → 幅度/相位 → 距离/振动"""
    print(f"处理文件: {os.path.basename(bin_file)}")
    print("-" * 60)
    save_path = os.path.join(os.path.dirname(bin_file),
                             os.path.splitext(os.path.basename(bin_file))[0])
    timestr = time.strftime("%Y%m%d%H%M%S")
    save_path_suffix = f'{save_path}_{timestr}_'

    # ---- Stage 1: 数据读取 (PD → ADC) ----
    print("=" * 60)
    print("Stage 1: 数据读取 (PD → ADC 采集电压)")
    print("=" * 60)
    result = fft_mod.load_data(bin_file)
    voltage = result['voltage_data']
    fs = result['sampling_rate']
    channels = result.get('channels') or [f'CH{i+1}' for i in range(voltage.shape[1])]
    channel_count = result.get('channel_count', 1)

    # 解交错并选择分析通道（默认第一个通道）
    if voltage.ndim == 1 and channel_count > 1:
        data = voltage.reshape(-1, channel_count)[:, 0]
    else:
        data = voltage if voltage.ndim == 1 else voltage[:, 0]
    print(f"[成功] 读取完成: 采样点数={result['num_samples']:,}, 采样率={fs/1e6:g} MHz, "
          f"通道数={len(channels)}, 分析通道={channels[0]}, 时长={len(data)/fs:.3f} s")

    # ---- Stage 2: FFT (以 AOM 频率为中心的频谱 + 瀑布图) ----
    print("\n" + "=" * 60)
    print("Stage 2: FFT (频谱分析, 中心频率 = AOM 40 MHz)")
    print("=" * 60)
    spectrogram_data = fft_mod.calculate_spectrogram(
        data, fs=fs,
        time_block_duration=TIME_BLOCK_DURATION,
        center_freq=AOM_FREQ_HZ,
        span=FREQ_SPAN,
        window=WINDOW_TYPE,
        overlap=OVERLAP_RATIO,
        detrend=DETREND_METHOD)

    fft_mod.plot_combined(data, fs, spectrogram_data, AOM_FREQ_HZ, FREQ_SPAN,
                          title=os.path.basename(bin_file), save_path=save_path_suffix)

    # ---- Stage 3: IQ 解调 (数字下变频) ----
    print("\n" + "=" * 60)
    print("Stage 3: IQ 解调 (40 MHz 数字下变频 → I/Q)")
    print("=" * 60)
    demod = iq_demodulate(data, fs, AOM_FREQ_HZ)
    plot_iq(demod, save_path=save_path_suffix)

    # ---- Stage 4: 幅度/相位 (含残余载波频偏补偿) ----
    print("\n" + "=" * 60)
    print("Stage 4: 幅度/相位计算")
    print("=" * 60)
    z_corr, df_res = correct_residual_carrier(demod, RESIDUAL_CARRIER_SEARCH_HZ)
    res = compute_amplitude_phase(z_corr, demod['t'], demod['fs'])
    plot_amplitude_phase(res, demod, save_path=save_path_suffix)

    # ---- Stage 5: 距离/振动 ----
    print("\n" + "=" * 60)
    print("Stage 5: 距离/振动")
    print("=" * 60)
    f_vib, psd_vib, vib_freq = vibration_spectrum(res['vib_disp'], demod['fs'])
    print(f"振动频率 (Welch 谱峰值): {vib_freq/1000:.3f} kHz ({vib_freq:.1f} Hz)")
    print(f"振动位移: 峰峰值={np.ptp(res['vib_disp'])*1e9:.1f} nm, "
          f"标准差={res['vib_disp'].std()*1e9:.1f} nm")
    print(f"振动速度: 峰峰值={np.ptp(res['vel'])*1e3:.4f} mm/s, "
          f"标准差={res['vel'].std()*1e3:.4f} mm/s")
    plot_distance_vibration(res, demod, f_vib, psd_vib, vib_freq,
                            save_path=save_path_suffix)

    # ---- 统计信息汇总 ----
    save_statistics_txt(
        save_path_suffix,
        collect_statistics(bin_file, result, channels, fs, spectrogram_data,
                           demod, res,
                           df_res, f_vib, psd_vib, vib_freq))

    print(f"\n处理完成！所有结果已保存到: {save_path}")


# ================= 自检 =================

def self_test():
    """
    合成 40 MHz 拍频信号验证 IQ 解调链路:
    载波 = AOM 频率 + 150 Hz 频偏, 相位受 100 Hz 单频振动调制 (幅度 2 rad),
    验证: 残余频偏估计、振动频率、振动位移峰峰值 (λ/(4π)·Δφ_pp)
    """
    print("=" * 60)
    print("自检: 合成 40 MHz 拍频信号 → IQ 解调链路验证")
    print("=" * 60)

    fs = 100e6
    dur = 0.1
    n = int(fs * dur)
    t = np.arange(n) / fs
    df_true = 150.0                    # 模拟 AOM 实际频率偏差
    f_vib_true = 100.0                 # 模拟振动频率
    phi_vib = 2.0                      # 模拟振动引起的相位幅度 (rad)

    carrier = np.cos(2 * np.pi * (AOM_FREQ_HZ + df_true) * t
                     + phi_vib * np.cos(2 * np.pi * f_vib_true * t))
    x = (0.8 * carrier + 0.05 * np.random.randn(n)).astype(np.float32)

    demod = iq_demodulate(x, fs, AOM_FREQ_HZ)
    z_corr, df_est = correct_residual_carrier(demod, RESIDUAL_CARRIER_SEARCH_HZ)
    res = compute_amplitude_phase(z_corr, demod['t'], demod['fs'])
    f_vib, psd_vib, f_est = vibration_spectrum(res['vib_disp'], demod['fs'])

    disp_pp = np.ptp(res['vib_disp'])
    disp_pp_expected = 2.0 * LASER_WAVELENGTH_M * phi_vib / (4.0 * np.pi)   # λ/(4π)·Δφ_pp
    print(f"残余频偏: 估计={df_est:.2f} Hz, 真值={df_true:.1f} Hz")
    print(f"振动频率: 估计={f_est:.2f} Hz, 真值={f_vib_true:.1f} Hz")
    print(f"振动位移峰峰值: 估计={disp_pp*1e9:.1f} nm, 理论={disp_pp_expected*1e9:.1f} nm")

    ok = (abs(df_est - df_true) < 0.5 and abs(f_est - f_vib_true) < 1.0
          and abs(disp_pp / disp_pp_expected - 1.0) < 0.05)
    print("自检通过" if ok else "自检失败")
    return ok


if __name__ == "__main__":
    try:
        run_analysis(cfg.DEFAULT_BIN_FILE)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
