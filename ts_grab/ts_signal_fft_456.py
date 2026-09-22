"""
读取自动保存的二进制数据文件(bin)并转换为电压值
移植自 ART-SCOPE SDK 示例 ReadAutoSavedFile-Binary.py

修正后的转换逻辑 (符合 PCIe8922M 协议):
- 通过 ArtScope_GetInfoFromAutoSaveFile 获取文件头大小和转换参数:
    range_span   : 满量程范围 (mV), 如 ±5V -> 10000mV
    range_offset : 零点偏移 (mV), 如 ±5V -> 5000mV
    wMaxLSB      : 最大码值掩码 (12bit 为 0x0FFF)
    channelCount : 通道数 (多通道按 index % channelCount 交错存储)
- 每个采样点 2 字节 (uint16)
- 电压(mV) = (range_span / 4096) * (code & wMaxLSB) - range_offset
- 码值高位(bit15~12)为标志位，转换时用 wMaxLSB 掩码屏蔽
"""

import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import gc
from tqdm import tqdm
from scipy import signal
from scipy.fft import rfft, rfftfreq
from numpy.lib.stride_tricks import sliding_window_view

# 中文标注字体支持 (Windows 常见中文字体，缺失则回退默认字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import ts_signal_fft_config as cfg   # 所有可调参数集中在该配置文件

# ========== 转换参数获取 ==========
def _get_info_from_header_txt(bin_file_path):
    """方式2: ACTS1000 界面保存的 bin 无文件头，从同名 _header.txt 解析等价参数。"""
    stem = os.path.splitext(bin_file_path)[0]
    stem = re.sub(r"_\d{17}(_\d+)?$", "", stem)   # 去掉文件名中的采集时间戳和序号
    header_file = stem + "_header.txt"
    if not os.path.exists(header_file):
        raise FileNotFoundError(f"DLL 无法识别该文件，且找不到头部参数文件: {header_file}")

    params = {}
    with open(header_file, encoding="utf-8") as fp:
        for line in fp:
            key, _, value = line.strip().partition(":")
            params[key] = value

    channel_count = int(params["chanEnableCount"])
    resolution = int(params["resolution"])
    w_max_lsb = (1 << resolution) - 1

    # 从 header 中获取量程信息
    range_span = np.empty(channel_count, dtype=np.float64)
    range_offset = np.empty(channel_count, dtype=np.float64)
    range_mv = np.empty(channel_count, dtype=np.float64)

    for ch in range(channel_count):
        range_max = float(params[f"{ch}rangeMaxValue"])   # mV
        range_min = float(params[f"{ch}rangeMinValue"])   # mV

        # 根据协议: Volt = (range_span / 4096) * code - range_offset
        # 其中: range_span = range_max - range_min, range_offset = range_max
        range_span[ch] = range_max - range_min
        range_offset[ch] = range_max
        range_mv[ch] = range_max

    return {
        "header_bytes": 0,          # 界面保存的 bin 无文件头
        "channel_count": channel_count,
        "range_span": range_span,
        "range_offset": range_offset,
        "w_max_lsb": w_max_lsb,
        "sample_rate": float(params["SampleRate"]),
        "range_mv": range_mv,
        "channel_range": np.array([f"±{int(rv / 1000)}V" for rv in range_mv])
    }


def read_bin_to_voltage(bin_file_path):
    """
    读取 bin 文件并转换为电压值

    根据协议修正转换公式:
    Volt(mV) = (range_span / 4096) * (code & wMaxLSB) - range_offset
    其中: range_span = 2 * range_max, range_offset = range_max

    参数:
        bin_file_path: bin文件路径

    返回:
        dict: 包含电压数据、采样点数、采样率等信息
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"文件不存在: {bin_file_path}")

    # ========== 1. 获取转换参数 ==========
    info = _get_info_from_header_txt(bin_file_path)
    print("从 _header.txt 解析参数")

    header_bytes = info["header_bytes"]
    channel_count = info["channel_count"]
    range_span = info["range_span"]      # mV
    range_offset = info["range_offset"]  # mV
    w_max_lsb = info["w_max_lsb"]

    print("=" * 60)
    print("bin 数据读取 (uint16, 2字节/点)")
    print("=" * 60)
    print(f"文件头大小: {header_bytes} 字节")
    print(f"通道数: {channel_count} (交错存储)")
    print(f"最大码值掩码: 0x{w_max_lsb:X} (12-bit)")
    print(f"各通道量程: {info.get('channel_range', ['未知'])}")
    print(f"各通道范围跨度 (mV): {range_span}")
    print(f"各通道偏移量 (mV): {range_offset}")
    print("=" * 60)

    # ========== 2. 内存映射读取原始码值 ==========
    file_size = os.path.getsize(bin_file_path)
    total_samples = round((file_size - header_bytes) / 2)   # 每采样点2字节
    print(f"文件实际大小: {file_size / (1024 ** 2):.1f} MB, 采样点数: {total_samples:,}")

    raw_data = np.memmap(bin_file_path, dtype=np.uint16, mode='r', offset=header_bytes, shape=(total_samples,))

    # ========== 3. 码值转电压 (修正为协议公式) ==========
    n_use = total_samples - total_samples % channel_count   # 丢弃不足一组的尾部数据
    if n_use == 0:
        raise ValueError("没有有效数据!")

    # 预分配内存 (mV)，转换完成后原地转为 V
    voltage_data = np.empty(n_use, dtype=np.float32)

    # float32 计算系数，避免产生 float64 临时数组（省内存且更快）
    scale_mv = (range_span / cfg.ADC_RESOLUTION).astype(np.float32)   # (n_ch,)
    offset_mv = range_offset.astype(np.float32)

    chunk = cfg.CHUNK_SAMPLES - cfg.CHUNK_SAMPLES % channel_count
    code_min, code_max = np.inf, -np.inf

    for start in range(0, n_use, chunk):
        end = min(start + chunk, n_use)
        block = raw_data[start:end].reshape(-1, channel_count)   # 每行一组交错通道

        # 屏蔽高位标志位 (12-bit 有效数据)
        codes = block & w_max_lsb

        code_min = min(code_min, int(codes.min()))
        code_max = max(code_max, int(codes.max()))

        # ===== 修正的转换公式 =====
        # 协议公式: Volt(mV) = (range_span / cfg.ADC_RESOLUTION) * code - range_offset
        # range_span: 满量程范围 (mV), range_offset: 零点偏移 (mV)
        # ±5V: Volt(mV) = (10000.0/cfg.ADC_RESOLUTION) * code - 5000.00
        # ±1V: Volt(mV) = (2000.0/cfg.ADC_RESOLUTION) * code - 1000.00
        voltage_data[start:end] = (
            codes.astype(np.float32) * scale_mv[None, :] - offset_mv[None, :]
        ).ravel()

        print(f"  转换进度: {end}/{n_use}")

    # mV -> V（原地转换，避免额外内存拷贝）
    voltage_data *= 1e-3

    # ========== 4. 统计信息 ==========
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"有效数据点数: {n_use:,}")
    print(f"ADC码值范围: [{code_min}, {code_max}] (掩码后)")
    print(f"电压范围: [{voltage_data.min():.4f}, {voltage_data.max():.4f}] V")
    mean_v, std_v = _mean_std(voltage_data)
    print(f"电压平均值: {mean_v:.4f} V")
    print(f"电压标准差: {std_v:.4f} V")
    print("=" * 60)

    return {
        'voltage_data': voltage_data,
        'num_samples': n_use,
        'sampling_rate': info["sample_rate"],
        'channels': info.get('channel_range', [f'CH{i + 1}' for i in range(channel_count)]),
        'shape': (n_use, channel_count),
        'range_span': range_span,      # 保留转换参数
        'range_offset': range_offset,
        'w_max_lsb': w_max_lsb,
        'channel_count': channel_count
    }


def read_csv_to_voltage(file_path, chunk_size=cfg.CSV_CHUNK_SIZE, sampling_rate=None):
    """
    分块读取大型CSV文件并转换为电压数据

    参数:
        file_path: CSV文件路径
        chunk_size: 每块的行数，根据内存调整
        sampling_rate: 采样率（Hz），如果提供则返回

    返回:
        dict: 包含电压数据、采样点数、采样率、通道名称等信息
    """
    print(f"开始处理文件: {file_path}")

    # 获取总行数（用于进度条）
    print("正在统计文件行数...")
    with open(file_path, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # 减去标题行
    print(f"总数据行数: {total_rows:,}")

    # 先读取第一行获取列名
    first_chunk = pd.read_csv(file_path, nrows=0)
    channels = first_chunk.columns.tolist()
    num_channels = len(channels)
    print(f"通道数量: {num_channels}")
    print(f"通道名称: {channels}")

    # 预估内存并创建空数组
    voltage_data = np.empty((total_rows, num_channels), dtype=np.float32)

    # 分块读取并处理
    print("开始处理数据...")
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    row_counter = 0
    with tqdm(total=total_rows, desc="处理进度", unit="行") as pbar:
        for chunk in chunk_reader:
            rawdata = chunk.values.astype(np.float32)
            # CSV 数据已经是码值，转换为电压
            voltage = (rawdata - cfg.CODE_OFFSET) * cfg.VOLTAGE_SCALE / cfg.ADC_RESOLUTION

            current_chunk_size = len(voltage)
            voltage_data[row_counter:row_counter + current_chunk_size, :] = voltage

            row_counter += current_chunk_size
            pbar.update(current_chunk_size)

            del chunk, rawdata, voltage
            if row_counter % (chunk_size * cfg.CSV_GC_INTERVAL) == 0:
                gc.collect()

    print(f"数据读取完成！共 {row_counter:,} 行")

    result = {
        'voltage_data': voltage_data,
        'num_samples': row_counter,
        'channels': channels,
        'shape': voltage_data.shape
    }

    if sampling_rate is not None:
        result['sampling_rate'] = sampling_rate
        result['duration'] = row_counter / sampling_rate
        print(f"采样率: {sampling_rate} Hz")
        print(f"数据时长: {row_counter / sampling_rate:.2f} 秒")

    return result


# ========== 时频分析功能 ==========
def _fmt_freq(hz):
    """按频谱仪显示习惯格式化频率: >= 1 MHz 用 MHz，否则用 kHz"""
    return f"{hz/1e6:.3f} MHz" if abs(hz) >= 1e6 else f"{hz/1e3:.3f} kHz"


def calculate_spectrogram(data, fs, time_block_duration, center_freq, span,
                         window=cfg.WINDOW_TYPE, overlap=cfg.OVERLAP_RATIO,
                         detrend=cfg.DETREND_METHOD):
    """
    计算数据的频谱图（时频分析）

    参数:
        data: 输入数据 (1D numpy数组)
        fs: 采样率 (Hz)
        time_block_duration: 每个时间块的长度 (秒)
        center_freq: 目标拍频 (Hz)，本振光与信号光的频差，即频谱分析中心频率
                     （参考频谱仪显示）
        span: 频移范围 (Hz)，信号光随目标振动产生的瞬时频移范围，
              分析范围为 [center_freq - span, center_freq + span]
        window: 窗函数类型
        overlap: 重叠比例 (0-1)
        detrend: 去趋势方法

    返回:
        dict: 包含频谱图数据、时间轴、频率轴等信息
    """
    samples_per_block = int(time_block_duration * fs)
    step = int(samples_per_block * (1 - overlap))
    n_blocks = (len(data) - samples_per_block) // step + 1
    if n_blocks < 1:
        raise ValueError(
            f"数据长度 ({len(data)}) 不足以构成一个时间块 ({samples_per_block} 采样点)")

    freq_axis = rfftfreq(samples_per_block, 1/fs)
    # 零填充: 将时间块补零到 pad_len，再做 FFT，等效插值提高频率定位精度
    # 不改变实际频谱分辨率，但可更精确地定位峰值频率
    pad_len = samples_per_block * cfg.FFT_ZERO_PAD_FACTOR
    freq_axis_padded = rfftfreq(pad_len, 1/fs)
    # 以目标拍频为中心，±span 范围进行频谱分析（振动瞬时频移区域）
    freq_mask = (freq_axis_padded >= center_freq - span) & (freq_axis_padded <= center_freq + span)
    freq_selected = freq_axis_padded[freq_mask]
    n_freqs = freq_selected.size
    if n_freqs == 0:
        raise ValueError(
            f"频率范围 [{center_freq - span:.0f}, {center_freq + span:.0f}] Hz 内无频率点，"
            f"请检查 TARGET_FREQ/FREQ_SPAN 与采样率 (fs={fs:.0f} Hz, Nyquist={fs/2:.0f} Hz)")

    freq_resolution_native = fs / samples_per_block
    freq_resolution_effective = fs / pad_len
    print(f"\n频谱分析参数:")
    print(f"  - 时间块长度: {time_block_duration*1000:.2f} ms ({samples_per_block:,} 采样点)")
    print(f"  - 重叠率: {overlap*100:.0f}%")
    print(f"  - 时间块数: {n_blocks:,}")
    print(f"  - FFT 原生频率分辨率: {freq_resolution_native:.1f} Hz")
    print(f"  - 零填充: {cfg.FFT_ZERO_PAD_FACTOR}x → 有效分辨率: {freq_resolution_effective:.1f} Hz")
    print(f"  - 目标拍频(中心频率): {_fmt_freq(center_freq)}")
    print(f"  - 频移范围: [{_fmt_freq(center_freq - span)}, {_fmt_freq(center_freq + span)}] (±{span/1000:.1f} kHz)")
    print(f"  - 分析频点数: {n_freqs}")

    # 窗函数（归一化到单位 RMS，float32 以加速 FFT）
    windows = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'rectangular': np.ones
    }
    win = windows.get(window, np.hanning)(samples_per_block)
    win = (win / np.sqrt(np.mean(win ** 2))).astype(np.float32)

    # 滑动窗口视图（零拷贝），分块向量化 FFT：
    # 相比逐块 Python 循环，消除了每块的函数调用/切片开销，且 rfft 只用一半计算量
    block_views = sliding_window_view(data, samples_per_block)[::step]
    chunk_blocks = max(1, int(cfg.SPECTROGRAM_CHUNK_MB * 1e6 // (16 * pad_len)))
    spectrogram = np.empty((n_blocks, n_freqs), dtype=np.float32)

    print("\n计算频谱图...")
    with tqdm(total=n_blocks, desc="处理进度") as pbar:
        for start in range(0, n_blocks, chunk_blocks):
            end = min(start + chunk_blocks, n_blocks)
            block = np.array(block_views[start:end])   # 拷贝为连续内存 (float32)

            if detrend == 'constant':
                block -= block.mean(axis=1, keepdims=True)
            elif detrend == 'linear':
                block = signal.detrend(block, axis=1).astype(np.float32)

            block *= win
            psd = np.abs(rfft(block, n=pad_len, axis=1))   # (chunk, pad_len//2+1) 零填充提高频率定位精度
            psd *= psd                                 # |X|^2
            psd *= 2.0 / (fs * samples_per_block)      # PSD (one-sided, factor 2 for rfft)
            spectrogram[start:end] = psd[:, freq_mask]
            pbar.update(end - start)

    time_axis = np.arange(n_blocks) * step / fs + samples_per_block / (2 * fs)

    return {
        'spectrogram': spectrogram,
        'time_axis': time_axis,
        'freq_axis': freq_selected,
        'freq_mask': freq_mask,
        'n_blocks': n_blocks,
        'n_freqs': n_freqs
    }


def _setup_freq_axis_scale(ax, freq_axis, center_freq):
    """
    按频谱仪显示习惯设置频率轴: 对数刻度（含目标拍频刻度），
    频率范围含 0 或负频率时自动改用 symlog 对称对数刻度。
    """
    if cfg.SPECTROGRAM_FREQ_SCALE == 'log':
        if freq_axis[0] <= 0:
            ax.set_yscale('symlog', linthresh=cfg.SPECTROGRAM_FREQ_LINTHRESH)
            print(f"[提示] 频率范围 [{_fmt_freq(freq_axis[0])}, {_fmt_freq(freq_axis[-1])}] "
                  f"包含0或负频率，频率轴改用 symlog 对称对数刻度显示")
        else:
            ax.set_yscale('log')
            # 对数间隔刻度 + 目标拍频刻度（合并相距小于 1 kHz 的刻度避免标签重叠）
            ticks = np.append(np.geomspace(freq_axis[0], freq_axis[-1],
                                           cfg.SPECTROGRAM_FREQ_NTICKS), center_freq)
            ax.set_yticks(np.unique(np.round(ticks / 1000)) * 1000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_freq(v)))


def plot_combined(voltage, fs, spectrogram_data, center_freq, span,
                  title=None, save_path=None, n_plot=cfg.N_PLOT,
                  vmin=None, vmax=None, cmap=cfg.DEFAULT_CMAP,
                  spectrum_peaks=None, sideband_peaks=None,
                  spectrum_energy=None):
    """
    绘制合并图:
      第一行: 时间-电压波形 | 全时段频谱 (标注高于阈值的峰值点及能量值)
      第二行: 时频瀑布图

    center_freq: 目标拍频 (Hz)，本振光与信号光的频差，即频谱分析中心频率
                 （参考频谱仪显示）
    spectrum_peaks: extract_spectrum_peaks() 返回的字典，若提供则在全时段频谱图上
                    标注高于阈值的峰值点（频率 + 能量值）并绘制阈值参考线
    sideband_peaks: extract_sideband_peaks() 返回的字典，若提供则在全时段频谱图上
                    标注目标拍频两侧的频移带振动峰值（仅标注对称性校验确认的峰值）
    spectrum_energy: 全时段能量谱 (1D, 与 freq_axis 等长)，若提供则直接使用
                    （通常为载波对齐后的能量谱），否则按时间积分原始频谱图
    """
    spectrogram = spectrogram_data['spectrogram']
    time_axis = spectrogram_data['time_axis']
    freq_axis = spectrogram_data['freq_axis']

    # float32 计算 dB，避免产生整张频谱图大小的 float64 临时数组
    spectrogram_db = 10 * np.log10(spectrogram + np.float32(cfg.SPECTROGRAM_DB_FLOOR),
                                   dtype=np.float32)

    if vmin is None:
        vmin = np.percentile(spectrogram_db, cfg.VMIN_PERCENTILE)
    if vmax is None:
        vmax = np.percentile(spectrogram_db, cfg.VMAX_PERCENTILE)

    fig = plt.figure(figsize=cfg.FIGURE_SIZE)
    gs = fig.add_gridspec(2, 2, height_ratios=cfg.GRID_HEIGHT_RATIOS)
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_tf = fig.add_subplot(gs[1, :])

    if title:
        fig.suptitle(title, fontsize=cfg.SUPTITLE_FONT_SIZE)

    # 时间-电压波形（全时段，按 n_plot 抽稀，避免为全量数据创建时间轴大数组）
    t_axis = np.arange(0, len(voltage), n_plot, dtype=np.float64) / fs
    ax_wave.plot(t_axis, voltage[::n_plot], linewidth=cfg.LINE_WIDTH)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Voltage (V)")
    ax_wave.set_title(f"SampleRate {fs/1e6:.0f} MHz, every {n_plot} samples ({len(voltage):,} pts)")
    ax_wave.grid(True, alpha=cfg.GRID_ALPHA)

    # 全时段频谱（能量随时间积分；使用载波对齐后的能量谱时载波漂移被补偿）
    if spectrum_energy is None:
        spectrum_energy = np.sum(spectrogram, axis=0)
    energy_spectrum_db = 10 * np.log10(spectrum_energy + cfg.SPECTROGRAM_DB_FLOOR)
    ax_spec.plot(freq_axis, energy_spectrum_db,
                 linewidth=cfg.LINE_WIDTH, color=cfg.SPECTRUM_LINE_COLOR)
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Integrated Energy (dB)')
    ax_spec.set_title(f'Full-Period Spectrum (carrier-aligned, Target: {_fmt_freq(center_freq)}, span ±{span/1000:.0f} kHz)')
    ax_spec.grid(True, alpha=cfg.GRID_ALPHA)
    # 目标拍频参考线（频谱仪中心频率标记）
    ax_spec.axvline(x=center_freq, **cfg.TARGET_FREQ_LINE_STYLE)
    ax_spec.text(center_freq, 0.98, f' TARGET {_fmt_freq(center_freq)}',
                 transform=ax_spec.get_xaxis_transform(), rotation=90,
                 va='top', ha='left',
                 fontsize=cfg.SPECTRUM_ANNOTATION_FONT_SIZE + 1,
                 color=cfg.TARGET_FREQ_LINE_STYLE.get('color', 'red'))
    ax_spec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_freq(v)))

    # 标注高于阈值的峰值点及能量值
    if spectrum_peaks:
        ax_spec.axhline(y=spectrum_peaks['threshold_db'], **cfg.SPECTRUM_THRESHOLD_LINE_STYLE)
        peak_freqs = spectrum_peaks['peak_freqs']
        peak_db = spectrum_peaks['peak_energy_db']
        ax_spec.scatter(peak_freqs, peak_db,
                        color=cfg.SPECTRUM_PEAK_COLOR, marker=cfg.SPECTRUM_PEAK_MARKER,
                        s=cfg.SPECTRUM_PEAK_MARKER_SIZE, zorder=5)
        for f, p in zip(peak_freqs, peak_db):
            ax_spec.annotate(f'{f/1000:.1f} kHz\n{p:.1f} dB',
                             xy=(f, p),
                             xytext=(0, cfg.SPECTRUM_ANNOTATION_OFFSET),
                             textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=cfg.SPECTRUM_ANNOTATION_FONT_SIZE,
                             color=cfg.SPECTRUM_PEAK_COLOR)

    # 标注载波两侧的侧边带振动峰值（仅对称性校验确认者；振动频率 = 峰值与载波的偏移量）
    if sideband_peaks:
        for name, band in sideband_peaks.items():
            if not band or not band.get('confirmed') or band['peak_freqs'].size == 0:
                continue
            f = band['peak_freqs'][0]
            p = band['peak_db'][0]
            vib = band['vib_freqs'][0]
            ax_spec.scatter([f], [p], color=cfg.SIDEBAND_PEAK_COLOR,
                            marker=cfg.SPECTRUM_PEAK_MARKER,
                            s=cfg.SPECTRUM_PEAK_MARKER_SIZE, zorder=6)
            ax_spec.annotate(f"{'下' if name == 'lower' else '上'}侧边带 {f/1000:.1f} kHz\n振动 {vib/1000:.1f} kHz, {p:.1f} dB",
                             xy=(f, p),
                             xytext=(0, -cfg.SPECTRUM_ANNOTATION_OFFSET),
                             textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=cfg.SPECTRUM_ANNOTATION_FONT_SIZE,
                             color=cfg.SIDEBAND_PEAK_COLOR)

    # 时频瀑布图
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    im = ax_tf.imshow(spectrogram_db.T, aspect='auto', origin='lower',
                      extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    _setup_freq_axis_scale(ax_tf, freq_axis, center_freq)
    ax_tf.set_xlabel('Time (s)')
    ax_tf.set_ylabel('Frequency (Hz)')
    ax_tf.set_title(f'Spectrogram (Target {_fmt_freq(center_freq)}, shift ±{span/1000:.0f} kHz)')
    ax_tf.grid(True, alpha=cfg.SPECTROGRAM_GRID_ALPHA, linestyle=cfg.SPECTROGRAM_GRID_STYLE)
    ax_tf.axhline(y=center_freq, **cfg.SPECTROGRAM_TARGET_LINE_STYLE)

    cbar = plt.colorbar(im, ax=ax_tf, label=cfg.COLORBAR_LABEL)

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}{cfg.SAVE_SUFFIX}', dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'Combined figure saved to: {save_path}{cfg.SAVE_SUFFIX}')

    # plt.show()
    return fig


def extract_peak_frequency(spectrogram_data):
    """
    提取每个时间块的最大频率点，使用二次插值精确定位子频段峰值

    二次插值公式: 对峰值 bin k 及其相邻 bin k-1, k+1 的幅度 a_{-1}, a_0, a_{+1}，
    拟合抛物线 y = a_0 + 0.5*(a_{+1} - a_{-1})*d + 0.5*(a_{+1} + a_{-1} - 2*a_0)*d^2，
    偏移量 d = -0.5*(a_{+1} - a_{-1})/(a_{+1} + a_{-1} - 2*a_0)
    精确峰值频率 = freq[k] + d * delta_f

    返回:
        peak_freqs: 精确峰值频率 (Hz)
        peak_amps: 峰值 PSD 值
    """
    spectrogram = spectrogram_data['spectrogram']
    freq_axis = spectrogram_data['freq_axis']

    n_blocks, n_freqs = spectrogram.shape
    peak_indices = np.argmax(spectrogram, axis=1)  # 每块峰值 bin 索引
    peak_amps = np.max(spectrogram, axis=1)

    # 二次插值精确定位子频段峰值 (边界处回退到 bin 中心)
    delta_f = freq_axis[1] - freq_axis[0] if n_freqs > 1 else 1.0
    peak_freqs = np.empty(n_blocks, dtype=np.float64)

    for i in range(n_blocks):
        k = peak_indices[i]
        if k == 0 or k == n_freqs - 1:
            # 边界: 无法插值，回退到 bin 中心
            peak_freqs[i] = freq_axis[k]
            continue
        a_left = spectrogram[i, k - 1]
        a_center = spectrogram[i, k]
        a_right = spectrogram[i, k + 1]
        # 分母: 抛物线曲率项
        denom = a_left + a_right - 2 * a_center
        if abs(denom) < 1e-30:
            # 平顶: 无法插值
            peak_freqs[i] = freq_axis[k]
        else:
            offset = -0.5 * (a_right - a_left) / denom
            # 限制偏移在 [-0.5, 0.5] 范围内
            offset = max(-0.5, min(0.5, offset))
            peak_freqs[i] = freq_axis[k] + offset * delta_f

    return peak_freqs, peak_amps


def carrier_aligned_energy(spectrogram_data, center_freq,
                           search_hz=cfg.CARRIER_SEARCH_HZ):
    """
    逐时间块对齐瞬时载波后积分，得到全时段能量谱（载波漂移补偿）

    载波频率随激光/本振漂移缓慢变化（本数据约 ±1 kHz），若直接沿时间积分，
    载波能量被漂移展宽到 ±1 kHz，侧边带提取不得不使用很宽的排除带，
    导致低频振动信号被掩盖、噪声起伏被误判为振动峰值。

    方法: 每个时间块在 center_freq ± search_hz 内锁定瞬时载波峰值 bin，
    将该块频谱平移使载波对齐到 center_freq 对应 bin 后再积分:
      - 载波泄漏集中于 center_freq 附近，可用窄保护带 (CARRIER_GUARD_HZ) 排除
      - 侧边带(±振动频率)在各时间块中相对瞬时载波位置固定，积分后尖锐、不被漂移展宽

    参数:
        spectrogram_data: calculate_spectrogram() 返回的字典
        center_freq: 目标拍频 (Hz)，对齐基准
        search_hz: 瞬时载波搜索范围 (Hz)，需覆盖载波漂移范围

    返回:
        energy: 载波对齐后的全时段能量谱 (1D, 与 freq_axis 等长)
        freq_axis: 频率轴 (Hz)
    """
    spectrogram = spectrogram_data['spectrogram']
    freq_axis = spectrogram_data['freq_axis']

    # 对齐基准 bin: 最接近 center_freq 的 bin
    center_bin = int(np.argmin(np.abs(freq_axis - center_freq)))
    search_idx = np.nonzero(np.abs(freq_axis - center_freq) <= search_hz)[0]
    if search_idx.size == 0:
        raise ValueError(f"载波搜索范围 ±{search_hz} Hz 内无频率点")

    aligned = np.empty_like(spectrogram)
    for i in range(spectrogram.shape[0]):
        row = spectrogram[i]
        carrier_bin = search_idx[int(np.argmax(row[search_idx]))]
        shift = center_bin - carrier_bin
        if shift == 0:
            aligned[i] = row
            continue
        row = np.roll(row, shift)
        # 平移后频谱两端会混入对端频率内容，置零避免污染侧边带区域
        if shift > 0:
            row[:shift] = 0.0
        else:
            row[shift:] = 0.0
        aligned[i] = row

    energy = aligned.sum(axis=0, dtype=np.float64)
    return energy, freq_axis


def extract_spectrum_peaks(spectrogram_data, threshold_db=cfg.SPECTRUM_THRESHOLD_DB,
                           energy=None):
    """
    在全时段能量谱上提取高于阈值的峰值点（频率 + 能量值）

    能量谱 = 各时间块 PSD 沿时间求和（全时段能量积分）；
    若提供 energy（载波对齐后的能量谱），则使用之——载波漂移被补偿后峰值更尖锐，
    阈值提取更准确。

    阈值   = 能量谱峰值 - threshold_db (dB)

    返回:
        dict: peak_freqs(峰值频率), peak_energy_db(峰值能量值, dB),
              threshold_db(绝对阈值, dB), max_db(能量谱峰值, dB),
              n_above(高于阈值的频率点总数)
    """
    if energy is None:
        energy = np.sum(spectrogram_data['spectrogram'], axis=0)
    energy_db = 10 * np.log10(energy + cfg.SPECTROGRAM_DB_FLOOR)
    max_db = float(energy_db.max())
    threshold_abs = max_db - threshold_db

    peaks, _ = signal.find_peaks(energy_db, height=threshold_abs)

    return {
        'peak_freqs': spectrogram_data['freq_axis'][peaks],
        'peak_energy_db': energy_db[peaks],
        'threshold_db': threshold_abs,
        'max_db': max_db,
        'n_above': int(np.sum(energy_db >= threshold_abs)),
    }


def extract_sideband_peaks(spectrogram_data, center_freq, energy=None,
                           guard_hz=cfg.CARRIER_GUARD_HZ,
                           min_snr_db=cfg.SIDEBAND_MIN_SNR_DB,
                           symmetry_hz=cfg.SIDEBAND_SYMMETRY_HZ):
    """
    在目标拍频两侧的频移带中提取振动信号峰值（载波对齐 + 噪声底门限 + 对称性校验）

    目标拍频 = center_freq（本振光与信号光的频差），振动信号以瞬时频移形式分布在
    拍频两侧，峰值频率与拍频的偏移量即为振动频率。

    默认基于载波对齐后的全时段能量谱 (carrier_aligned_energy):
      - 逐时间块对齐瞬时载波后积分，消除载波漂移(约±1 kHz)造成的能量展宽，
        因此只需窄保护带 (guard_hz) 排除载波泄漏，低频振动信号不被掩盖
      - 噪声底门限: 峰值须高于本带噪声底(中位数) min_snr_db，过滤噪声起伏
        (此前仅要求高于本带最大值以下 10 dB，噪声带内几乎全部频点都满足)
      - 对称性校验: 真实单频振动的 FM 边带在上/下频带对称出现且振动频率相同，
        对侧存在偏移量相差 <= symmetry_hz 的峰值才确认有效，否则标记
        confirmed=False (疑似噪声)

    返回:
        dict: {'lower': {...}, 'upper': {...}}
        每个频移带包含: noise_floor_db(噪声底, dB),
                       peak_freqs(峰值频率, 按能量降序), peak_db(峰值能量, dB),
                       vib_freqs(振动频率 = |峰值频率 - 目标拍频|),
                       n_above(高于噪声底+SNR门限的频率点数), confirmed(对称性确认)
    """
    freq_axis = spectrogram_data['freq_axis']
    if energy is None:
        energy, _ = carrier_aligned_energy(spectrogram_data, center_freq)
    energy_db = 10 * np.log10(energy + cfg.SPECTROGRAM_DB_FLOOR)

    bands = {
        'lower': freq_axis < center_freq - guard_hz,   # 下频移带
        'upper': freq_axis > center_freq + guard_hz,   # 上频移带
    }

    result = {}
    for name, mask in bands.items():
        sub_db = energy_db[mask]
        if sub_db.size == 0:
            result[name] = {'noise_floor_db': np.nan, 'peak_freqs': np.array([]),
                            'peak_db': np.array([]), 'vib_freqs': np.array([]),
                            'match_mask': np.array([], dtype=bool),
                            'n_above': 0, 'confirmed': False}
            continue
        noise_floor_db = float(np.median(sub_db))   # 本带噪声底
        peaks, _ = signal.find_peaks(sub_db, height=noise_floor_db + min_snr_db)
        order = np.argsort(sub_db[peaks])[::-1]     # 按能量从高到低排序
        result[name] = {
            'noise_floor_db': noise_floor_db,
            'peak_freqs': freq_axis[mask][peaks][order],
            'peak_db': sub_db[peaks][order],
            'vib_freqs': np.abs(freq_axis[mask][peaks][order] - center_freq),
            'n_above': int(np.sum(sub_db >= noise_floor_db + min_snr_db)),
            'confirmed': False,
        }

    # 对称性交叉校验: 真实单频振动的边带在上/下频带对称出现（振动频率相同），
    # 为每个峰值单独标记是否在对侧频移带找到对称峰值
    for name, other in (('lower', 'upper'), ('upper', 'lower')):
        band, other_band = result[name], result[other]
        n = band['peak_freqs'].size
        band['match_mask'] = np.zeros(n, dtype=bool)
        if n and other_band['peak_freqs'].size:
            band['match_mask'] = np.any(
                np.abs(other_band['vib_freqs'][None, :] - band['vib_freqs'][:, None])
                <= symmetry_hz, axis=1)
        band['confirmed'] = bool(n and band['match_mask'][0])
    return result


# ========== Stage 4: 信号复原 — 振动速度/位移定量测量 ==========

def recover_instantaneous_frequency(spectrogram_data, center_freq,
                                    smooth_window=cfg.FREQ_SMOOTH_WINDOW):
    """
    提取瞬时频率偏移 Δf(t) = 峰值频率 - 目标拍频 (Doppler 频移)

    参数:
        spectrogram_data: calculate_spectrogram() 返回的字典
        center_freq: 目标拍频 (Hz)，零速度参考频率
        smooth_window: 滑动中值滤波窗口 (奇数，1=不平滑)

    返回:
        time_axis: 时间轴 (s)，每个时间块的中心时刻
        freq_shift: 瞬时频移 Δf(t) (Hz)，正值为目标靠近，负值为远离
        peak_freqs: 瞬时峰值频率 (Hz)
        peak_amps: 瞬时峰值 PSD 值
    """
    peak_freqs, peak_amps = extract_peak_frequency(spectrogram_data)
    time_axis = spectrogram_data['time_axis']

    # 滑动中值滤波去除野值 (可选)
    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1  # 确保奇数
    if smooth_window > 1:
        peak_freqs = signal.medfilt(peak_freqs, kernel_size=smooth_window)

    # Doppler 频移: Δf = f_peak - f_center
    # 正 Δf: 目标靠近 (反射光频率升高); 负 Δf: 目标远离
    freq_shift = peak_freqs - center_freq

    return time_axis, freq_shift, peak_freqs, peak_amps


def recover_velocity(freq_shift, wavelength=cfg.LASER_WAVELENGTH_M):
    """
    Doppler 频移 → 振动速度

    公式 (Michelson 干涉仪, 双程 Doppler):
        v = λ × Δf / 2
    其中:
        v  : 振动速度 (m/s), 正值为目标靠近
        λ  : 激光波长 (m)
        Δf : Doppler 频移 (Hz)

    参数:
        freq_shift: 瞬时频移 Δf(t) (Hz), 1D 数组
        wavelength: 激光波长 (m), 默认 1064 nm

    返回:
        velocity: 振动速度 (m/s), 1D 数组
        velocity_mm_s: 振动速度 (mm/s), 1D 数组
    """
    velocity = wavelength * freq_shift / 2.0
    velocity_mm_s = velocity * 1e3
    return velocity, velocity_mm_s


def recover_displacement(velocity, time_axis, detrend=cfg.DISPLACEMENT_DETREND):
    """
    速度积分 → 振动位移

    公式: d(t) = ∫₀ᵗ v(τ) dτ
    数值积分: 梯形法则 (cumulative trapezoidal)

    参数:
        velocity: 振动速度 (m/s), 1D 数组
        time_axis: 时间轴 (s), 1D 数组
        detrend: 是否在积分前去除速度的线性趋势 (消除缓慢漂移导致的位移发散)

    返回:
        displacement: 振动位移 (m), 1D 数组
        displacement_um: 振动位移 (μm), 1D 数组
        displacement_nm: 振动位移 (nm), 1D 数组
    """
    from scipy.integrate import cumulative_trapezoid

    v = velocity.copy()
    if detrend and len(v) > 2:
        v = signal.detrend(v)

    # 梯形积分: d[i] = ∫₀^{t[i]} v(τ) dτ
    # cumulative_trapezoid 返回长度 n-1, 前补 0
    disp = cumulative_trapezoid(v, time_axis, initial=0.0)

    return disp, disp * 1e6, disp * 1e9


def reconstruct_signal(spectrogram_data, center_freq, wavelength=cfg.LASER_WAVELENGTH_M,
                       smooth_window=cfg.FREQ_SMOOTH_WINDOW,
                       detrend=cfg.DISPLACEMENT_DETREND):
    """
    完整信号复原流程: 瞬时频率 → 速度 → 位移

    返回:
        dict: 包含 time_axis, freq_shift, peak_freqs, peak_amps,
              velocity (m/s), velocity_mm_s, displacement (m),
              displacement_um, displacement_nm
    """
    time_axis, freq_shift, peak_freqs, peak_amps = recover_instantaneous_frequency(
        spectrogram_data, center_freq, smooth_window)

    velocity, velocity_mm_s = recover_velocity(freq_shift, wavelength)

    displacement, displacement_um, displacement_nm = recover_displacement(
        velocity, time_axis, detrend)

    print(f"\n{'='*60}")
    print("Stage 4: 信号复原 — 振动定量测量")
    print(f"{'='*60}")
    print(f"  激光波长: {wavelength*1e9:.0f} nm")
    print(f"  Doppler 系数: λ/2 = {wavelength/2*1e6:.4f} μm/Hz")
    print(f"  瞬时频移: 均值={freq_shift.mean():.1f} Hz, 标准差={freq_shift.std():.1f} Hz, "
          f"范围=[{freq_shift.min():.1f}, {freq_shift.max():.1f}] Hz")
    print(f"  振动速度: 均值={velocity_mm_s.mean():.4f} mm/s, 标准差={velocity_mm_s.std():.4f} mm/s, "
          f"峰峰值={np.ptp(velocity_mm_s):.4f} mm/s")
    print(f"  振动位移: 均值={displacement_um.mean():.4f} μm, 标准差={displacement_um.std():.4f} μm, "
          f"峰峰值={np.ptp(displacement_um):.4f} μm")
    print(f"  振动位移: 峰峰值={np.ptp(displacement_nm):.1f} nm")
    print(f"{'='*60}")

    return {
        'time_axis': time_axis,
        'freq_shift': freq_shift,
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps,
        'velocity': velocity,
        'velocity_mm_s': velocity_mm_s,
        'displacement': displacement,
        'displacement_um': displacement_um,
        'displacement_nm': displacement_nm,
    }


def plot_reconstructed(recon, save_path=None):
    """
    绘制复原的振动信号 (4 行子图):
      第 1 行: 瞬时频率偏移 Δf(t) (Hz)
      第 2 行: 振动速度 v(t) (mm/s)
      第 3 行: 振动位移 d(t) (μm)
      第 4 行: 瞬时峰值 PSD (dB)

    参数:
        recon: reconstruct_signal() 返回的字典
        save_path: 图片保存路径 (不含扩展名)
    """
    time_axis = recon['time_axis']
    decimate = cfg.SIGNAL_PLOT_DECIMATE

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 第 1 行: 瞬时频移
    ax = axes[0]
    ax.plot(time_axis[::decimate], recon['freq_shift'][::decimate],
            linewidth=cfg.LINE_WIDTH, color='tab:blue')
    ax.set_ylabel('Freq Shift Δf (Hz)')
    ax.set_title('Instantaneous Doppler Frequency Shift')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    # 第 2 行: 振动速度
    ax = axes[1]
    ax.plot(time_axis[::decimate], recon['velocity_mm_s'][::decimate],
            linewidth=cfg.LINE_WIDTH, color='tab:orange')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Recovered Vibration Velocity')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    # 第 3 行: 振动位移
    ax = axes[2]
    ax.plot(time_axis[::decimate], recon['displacement_um'][::decimate],
            linewidth=cfg.LINE_WIDTH, color='tab:green')
    ax.set_ylabel('Displacement (μm)')
    ax.set_title('Recovered Vibration Displacement')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    # 第 4 行: 瞬时峰值 PSD
    ax = axes[3]
    peak_db = 10 * np.log10(recon['peak_amps'] + cfg.SPECTROGRAM_DB_FLOOR)
    ax.plot(time_axis[::decimate], peak_db[::decimate],
            linewidth=cfg.LINE_WIDTH, color='tab:red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Peak PSD (dB)')
    ax.set_title('Instantaneous Peak PSD')
    ax.grid(True, alpha=cfg.GRID_ALPHA)

    plt.tight_layout()

    if save_path:
        fig_path = f'{save_path}_reconstructed{cfg.SAVE_SUFFIX}'
        plt.savefig(fig_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'Reconstructed signal figure saved to: {fig_path}')

    return fig


def _recon_stats_lines(recon, center_freq):
    """生成复原信号统计行，供 collect_statistics 和 txt 输出共用"""
    lines = []
    sep = "=" * 60
    lines += ["", sep, "Stage 4: 信号复原 — 振动定量测量", sep]
    lines.append(f"激光波长: {cfg.LASER_WAVELENGTH_M*1e9:.0f} nm")
    lines.append(f"Doppler 系数 (λ/2): {cfg.LASER_WAVELENGTH_M/2*1e6:.4f} μm/Hz")
    lines.append(f"目标拍频 (零速度参考): {_fmt_freq(center_freq)}")
    lines.append(f"瞬时频移 Δf: 均值={recon['freq_shift'].mean():.2f} Hz, "
                 f"标准差={recon['freq_shift'].std():.2f} Hz, "
                 f"范围=[{recon['freq_shift'].min():.2f}, {recon['freq_shift'].max():.2f}] Hz, "
                 f"峰峰值={np.ptp(recon['freq_shift']):.2f} Hz")
    lines.append(f"振动速度 v: 均值={recon['velocity_mm_s'].mean():.4f} mm/s, "
                 f"标准差={recon['velocity_mm_s'].std():.4f} mm/s, "
                 f"范围=[{recon['velocity_mm_s'].min():.4f}, {recon['velocity_mm_s'].max():.4f}] mm/s, "
                 f"峰峰值={np.ptp(recon['velocity_mm_s']):.4f} mm/s")
    lines.append(f"振动位移 d: 均值={recon['displacement_um'].mean():.4f} μm, "
                 f"标准差={recon['displacement_um'].std():.4f} μm, "
                 f"范围=[{recon['displacement_um'].min():.4f}, {recon['displacement_um'].max():.4f}] μm, "
                 f"峰峰值={np.ptp(recon['displacement_um']):.4f} μm "
                 f"({np.ptp(recon['displacement_nm']):.1f} nm)")
    lines.append(sep)
    return lines


def _mean_std(data, chunk=cfg.MEAN_STD_CHUNK):
    """分块计算均值和标准差，避免 np.std 产生 O(n) 临时数组导致内存不足"""
    data = data.ravel()
    n = data.size
    mean = float(np.mean(data))
    ss = 0.0
    for start in range(0, n, chunk):
        diff = data[start:start + chunk].astype(np.float64)
        diff -= mean
        ss += float(np.dot(diff, diff))
    return mean, (ss / n) ** 0.5


def collect_statistics(bin_file, result, voltage_data, channels, fs,
                       center_freq, span, spectrogram_data, spectrum_peaks,
                       sideband_peaks, recon=None):
    """汇总数据统计与频谱分析统计，返回文本行列表（供控制台打印与 txt 保存共用）"""
    lines = []
    sep = "=" * 60
    lines += [sep, "数据统计信息", sep]
    lines.append(f"文件: {bin_file}")
    lines.append(f"采样点数: {result['num_samples']:,}")
    lines.append(f"采样率: {fs:,.0f} Hz ({fs/1e6:g} MHz)")
    lines.append(f"通道数: {len(channels)}")

    v = voltage_data.reshape(-1, 1) if voltage_data.ndim == 1 else voltage_data
    mean_all, std_all = _mean_std(v)
    lines.append(f"整体: 均值={mean_all:.4f}V, 标准差={std_all:.4f}V, "
                 f"最小值={v.min():.4f}V, 最大值={v.max():.4f}V, 峰峰值={np.ptp(v):.4f}V")

    for i, ch in enumerate(channels):
        d = v[:, i]
        mean, std = _mean_std(d)
        lines.append(f"  {ch:8s}: 均值={mean:8.4f}V, 标准差={std:8.4f}V, "
                     f"最小值={d.min():8.4f}V, 最大值={d.max():8.4f}V, 峰峰值={np.ptp(d):8.4f}V")

    lines += ["", sep, "频谱分析统计 (全时段)", sep]
    lines.append(f"目标拍频(本振-信号频差): {_fmt_freq(center_freq)}")
    lines.append(f"频移范围: [{_fmt_freq(center_freq - span)}, "
                 f"{_fmt_freq(center_freq + span)}] (±{span/1000:.1f} kHz)")
    lines.append(f"时间块长度: {cfg.TIME_BLOCK_DURATION*1000:.2f} ms, "
                 f"重叠率: {cfg.OVERLAP_RATIO*100:.0f}%, "
                 f"时间块数: {spectrogram_data['n_blocks']:,}, "
                 f"分析频点数: {spectrogram_data['n_freqs']:,}")
    freq_res_native = fs / (cfg.TIME_BLOCK_DURATION * fs)
    freq_res_eff = fs / (cfg.TIME_BLOCK_DURATION * fs * cfg.FFT_ZERO_PAD_FACTOR)
    lines.append(f"频谱分辨率: 原生 {freq_res_native:.1f} Hz, "
                 f"零填充 {cfg.FFT_ZERO_PAD_FACTOR}x → 有效 {freq_res_eff:.1f} Hz")

    peak_freqs_block, _ = extract_peak_frequency(spectrogram_data)
    drift_std_hz = float(peak_freqs_block.std())
    interp_note = ("（低于有效分辨率，由峰值二次插值估计）" if drift_std_hz < freq_res_eff else "")
    lines.append(f"载波瞬时频率(逐时间块峰值): 平均={_fmt_freq(peak_freqs_block.mean())}, "
                 f"漂移标准差={drift_std_hz/1000:.3f} kHz{interp_note}, "
                 f"漂移范围=[{_fmt_freq(peak_freqs_block.min())}, {_fmt_freq(peak_freqs_block.max())}]")

    lines += ["",
              f"全时段能量谱峰值: {spectrum_peaks['max_db']:.2f} dB",
              f"提取阈值 (峰值-{cfg.SPECTRUM_THRESHOLD_DB}dB): {spectrum_peaks['threshold_db']:.2f} dB",
              f"高于阈值的频率点数: {spectrum_peaks['n_above']:,}",
              f"高于阈值的峰值点 (共 {len(spectrum_peaks['peak_freqs'])} 个):"]
    lines.append(f"  {'序号':>4s}  {'频率':>12s}  {'能量值 (dB)':>12s}")
    for i, (f, p) in enumerate(zip(spectrum_peaks['peak_freqs'],
                                   spectrum_peaks['peak_energy_db'])):
        lines.append(f"  {i + 1:>4d}  {_fmt_freq(f):>12s}  {p:>12.2f}")

    # 振动频移分析: 振动信号以瞬时频移形式分布在目标拍频两侧，
    # 峰值频率与目标拍频的偏移量即为振动频率
    lines += ["", "振动频移分析 (目标拍频两侧 ±FREQ_SPAN)", "-" * 60]
    lines.append(f"目标拍频: {_fmt_freq(center_freq)}, 频移跨度: ±{span/1000:.1f} kHz")
    lines.append(f"载波对齐: 逐时间块锁定瞬时载波(搜索范围 ±{cfg.CARRIER_SEARCH_HZ/1000:.1f} kHz)后积分, "
                 f"消除载波漂移展宽; 对齐后拍频排除带宽: ±{cfg.CARRIER_GUARD_HZ/1000:.2f} kHz")
    lines.append(f"侧边带判定: 峰值须高于本带噪声底(中位数) {cfg.SIDEBAND_MIN_SNR_DB} dB, "
                 f"且对侧频移带存在对称峰值(振动频率容差 ±{cfg.SIDEBAND_SYMMETRY_HZ} Hz)才确认有效")
    lines.append(f"振动频率 = |峰值频率 - 目标拍频| (即频移量, 单频振动下等于振动频率); "
                 f"合理性上限: {cfg.VIB_FREQ_MAX_HZ/1000:.1f} kHz (超过此值标记为 [可疑] 噪声/干扰)")
    for name, label in (("lower", "下频移带"), ("upper", "上频移带")):
        band = sideband_peaks.get(name)
        if band is None or band["peak_freqs"].size == 0:
            nf = band["noise_floor_db"] if band else float('nan')
            lines.append(f"{label}: 未检测到显著峰值 (噪声底 {nf:.2f} dB)")
            continue
        f0, p0, v0 = band["peak_freqs"][0], band["peak_db"][0], band["vib_freqs"][0]
        status = "已确认" if band["confirmed"] else "未确认(对侧无对称峰值, 疑似噪声)"
        suspicious = " [可疑] 振动频率异常大，可能为噪声或干扰" if v0 > cfg.VIB_FREQ_MAX_HZ else ""
        lines.append(f"{label}: 峰值 {_fmt_freq(f0)} ({p0:.2f} dB, 高于噪声底 {p0 - band['noise_floor_db']:.1f} dB), "
                     f"振动频率 {v0/1000:.3f} kHz{suspicious}, 对称性: {status}, "
                     f"高于门限的点数 {band["n_above"]}")
        for pf, pd, pv, pm in zip(band["peak_freqs"][1:6], band["peak_db"][1:6],
                                  band["vib_freqs"][1:6], band["match_mask"][1:6]):
            suspicious2 = " [可疑]" if pv > cfg.VIB_FREQ_MAX_HZ else ""
            status2 = "" if pm else " [未确认] 对侧无对称峰值，疑似噪声"
            lines.append(f"    - {_fmt_freq(pf)} ({pd:.2f} dB), 振动频率 {pv/1000:.3f} kHz{suspicious2}{status2}")
    lines.append(sep)
    if recon is not None:
        lines += _recon_stats_lines(recon, center_freq)
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


def verify_conversion():
    """
    验证转换是否正确
    使用协议中的示例值进行验证
    """
    print("=" * 60)
    print("转换公式验证 (根据 PCIe8922M 协议)")
    print("=" * 60)

    # 协议示例数据 (±5V 量程)
    test_cases = [
        (0xFFF, "正满度", 4095, 5000.0),
        (0xFFE, "正满度-1LSB", 4094, 4997.56),
        (0x801, "中间值+1LSB", 2049, 3.66),
        (0x800, "中间值(零点)", 2048, 0.0),
        (0x7FF, "中间值-1LSB", 2047, -3.66),
        (0x001, "负满度+1LSB", 1, -4997.56),
        (0x000, "负满度", 0, -5000.0),
    ]

    range_span = cfg.VERIFY_RANGE_SPAN    # ±5V = 10000mV
    range_offset = cfg.VERIFY_RANGE_OFFSET

    print(f"\n量程: ±5V")
    print(f"公式: Volt(mV) = ({range_span}/4096) * code - {range_offset}")
    print(f"      Volt(V)  = Volt(mV) / 1000")
    print("\n" + "-" * 70)
    print(f"{'码值(hex)':<12} {'描述':<15} {'code':<8} {'理论值(mV)':<12} {'计算值(V)':<12} {'误差(V)':<12}")
    print("-" * 70)

    for hex_val, desc, code, expected_mv in test_cases:
        calculated_mv = (range_span / cfg.ADC_RESOLUTION) * code - range_offset
        calculated_v = calculated_mv / 1000.0
        expected_v = expected_mv / 1000.0
        error = calculated_v - expected_v
        print(f"0x{hex_val:03X}     {desc:<15} {code:<8} {expected_mv:<12.3f} {calculated_v:<12.6f} {error:<12.6f}")

    print("=" * 60)


# ========== 主程序 ==========
def load_data(file_path):
    """按扩展名读取数据文件 (bin/csv)，返回统一格式的 dict"""
    file_type = os.path.splitext(file_path)[1].lower()
    if file_type == '.csv':
        return read_csv_to_voltage(file_path, sampling_rate=cfg.DEFAULT_SAMPLE_RATE)
    if file_type == '.bin':
        return read_bin_to_voltage(file_path)
    raise ValueError(f"不支持的文件格式: {file_type}")


def run_analysis(bin_file):
    """完整分析流程: 读取数据 -> 全时段时频分析 -> 绘图 -> 统计信息保存到 txt"""
    print(f"处理文件: {os.path.basename(bin_file)}")
    print("-" * 60)
    save_path = os.path.join(os.path.dirname(bin_file), os.path.splitext(os.path.basename(bin_file))[0])
    timestr = time.strftime("%Y%m%d%H%M%S")
    save_path_suffix = f'{save_path}_{timestr}_'

    # 1. 读取数据
    result = load_data(bin_file)
    voltage = result['voltage_data']
    fs = result['sampling_rate']
    channels = result.get('channels') or [f'CH{i+1}' for i in range(voltage.shape[1])]

    # 2. 解交错并选择分析通道（默认第一个通道）
    channel_count = result['channel_count']
    if voltage.ndim == 1 and channel_count > 1:
        data = voltage.reshape(-1, channel_count)[:, 0]
    else:
        data = voltage if voltage.ndim == 1 else voltage[:, 0]
    print(f"\n[成功] 读取完成: 采样点数={result['num_samples']:,}, 采样率={fs/1e6:g} MHz, "
          f"通道数={len(channels)}, 数据形状={voltage.shape}")
    print(f"分析通道: {channels[0]}")

    # 3. 全时段时频分析（不截取数据），以 TARGET_FREQ(本振光与信号光的频差)
    #    为中心频率，分析范围 TARGET_FREQ ± FREQ_SPAN（参考频谱仪显示）
    spectrogram_data = calculate_spectrogram(
        data, fs=fs,
        time_block_duration=cfg.TIME_BLOCK_DURATION,
        center_freq=cfg.TARGET_FREQ,
        span=cfg.FREQ_SPAN,
        window=cfg.WINDOW_TYPE,
        overlap=cfg.OVERLAP_RATIO)

    # 4. 载波对齐能量谱: 逐时间块锁定瞬时载波后积分（补偿载波漂移展宽）
    aligned_energy, _ = carrier_aligned_energy(spectrogram_data, cfg.TARGET_FREQ,
                                               cfg.CARRIER_SEARCH_HZ)

    # 5. 提取高于阈值的频谱峰值点（频率 + 能量值，基于载波对齐能量谱）
    spectrum_peaks = extract_spectrum_peaks(spectrogram_data, energy=aligned_energy,
                                            threshold_db=cfg.SPECTRUM_THRESHOLD_DB)

    # 6. 提取目标拍频两侧频移带中的振动峰值（振动频率 = |峰值频率 - 目标拍频|，
    #    噪声底门限 + 上下频带对称性校验）
    sideband_peaks = extract_sideband_peaks(spectrogram_data, cfg.TARGET_FREQ,
                                            energy=aligned_energy,
                                            guard_hz=cfg.CARRIER_GUARD_HZ,
                                            min_snr_db=cfg.SIDEBAND_MIN_SNR_DB,
                                            symmetry_hz=cfg.SIDEBAND_SYMMETRY_HZ)

    # 7. 绘制合并图（波形 + 全时段频谱(含阈值峰值标注) + 时频瀑布图）
    plot_combined(data, fs, spectrogram_data, cfg.TARGET_FREQ, cfg.FREQ_SPAN,
                  spectrum_peaks=spectrum_peaks, sideband_peaks=sideband_peaks,
                  spectrum_energy=aligned_energy,
                  title=os.path.basename(bin_file), save_path=save_path_suffix)

    # 8. Stage 4: 信号复原 — 瞬时频率 → 振动速度 → 振动位移
    recon = reconstruct_signal(spectrogram_data, cfg.TARGET_FREQ,
                               wavelength=cfg.LASER_WAVELENGTH_M,
                               smooth_window=cfg.FREQ_SMOOTH_WINDOW,
                               detrend=cfg.DISPLACEMENT_DETREND)

    # 9. 绘制复原的振动信号 (频移/速度/位移/PSD)
    plot_reconstructed(recon, save_path=save_path_suffix)

    # 10. 汇总统计信息并保存到 txt
    save_statistics_txt(save_path_suffix,
                        collect_statistics(bin_file, result, voltage, channels, fs,
                                           cfg.TARGET_FREQ, cfg.FREQ_SPAN,
                                           spectrogram_data, spectrum_peaks,
                                           sideband_peaks, recon=recon))

    print(f"\n处理完成！所有结果已保存到: {save_path}")


if __name__ == "__main__":
    try:
        run_analysis(cfg.DEFAULT_BIN_FILE)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()