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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from tqdm import tqdm
from scipy import signal
from scipy.fft import fft, fftfreq

# ========== 转换参数获取 ==========
def _get_info_from_dll(bin_file_path):
    """方式1: 通过 DLL API 获取文件头大小与转换参数(与 ReadAutoSavedFile-Binary.py 相同)。

    仅适用于 ArtScope_AutoSaveFile API 保存的文件。
    返回 None 表示 DLL 无法识别该文件(如 ACTS1000 界面保存的 bin)。
    """
    import ctypes
    from ART_SCOPE_Lib.functions import Functions
    from ART_SCOPE_Lib.constants import ArtScope_wfmInfo

    header_size = ctypes.c_int32(0)
    wfm_info = ArtScope_wfmInfo()
    error_code = Functions.ArtScope_GetInfoFromAutoSaveFile(bin_file_path, header_size, wfm_info)
    if error_code < 0:
        return None

    channel_count = wfm_info.channelCount

    # 获取量程范围 (单位: mV)
    # 根据协议: ±5V 或 ±1V
    range_values = np.array(wfm_info.rangevalue[:channel_count], dtype=np.float64)

    # 根据量程确定转换参数
    # 协议公式: Volt(mV) = (range_span / 4096) * code - range_offset
    # 其中: range_span = 2 * range_max, range_offset = range_max
    # 对于 ±5V: range_span = 10000mV, range_offset = 5000mV
    # 对于 ±1V: range_span = 2000mV, range_offset = 1000mV
    range_span = range_values * 2   # 满量程范围
    range_offset = range_values     # 偏移量

    return {
        "header_bytes": header_size.value,
        "channel_count": channel_count,
        "range_span": range_span,        # mV (10000 或 2000)
        "range_offset": range_offset,    # mV (5000 或 1000)
        "w_max_lsb": int(wfm_info.wMaxLSB),  # 12bit: 0x0FFF
        "sample_rate": 100_000_000.0,    # wfmInfo 不含采样率，按 100MHz 计
        "range_mv": range_values,        # 量程 (mV)
        "channel_range": np.array([f"±{int(rv)}V" for rv in range_values / 1000])
    }


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


CHUNK_SAMPLES = 10_000_000   # 分块转换，避免一次性占用过多内存


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
    info = _get_info_from_dll(bin_file_path)
    if info is None:
        print("DLL 无法识别该文件(非 ArtScope_AutoSaveFile API 保存)，从 _header.txt 解析参数")
        info = _get_info_from_header_txt(bin_file_path)

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

    raw_data = np.memmap(bin_file_path, dtype=np.uint16, mode='r',
                         offset=header_bytes, shape=(total_samples,))

    # ========== 3. 码值转电压 (修正为协议公式) ==========
    n_use = total_samples - total_samples % channel_count   # 丢弃不足一组的尾部数据
    if n_use == 0:
        raise ValueError("没有有效数据!")

    # 预分配内存 (mV)
    voltage_data_mv = np.empty(n_use, dtype=np.float32)

    chunk = CHUNK_SAMPLES - CHUNK_SAMPLES % channel_count
    code_min, code_max = np.inf, -np.inf

    for start in range(0, n_use, chunk):
        end = min(start + chunk, n_use)
        block = raw_data[start:end].reshape(-1, channel_count)   # 每行一组交错通道

        # 屏蔽高位标志位 (12-bit 有效数据)
        codes = block & w_max_lsb

        code_min = min(code_min, int(codes.min()))
        code_max = max(code_max, int(codes.max()))

        # ===== 修正的转换公式 =====
        # 协议公式: Volt(mV) = (range_span / 4096) * code - range_offset
        # range_span: 满量程范围 (mV), range_offset: 零点偏移 (mV)
        # ±5V: Volt(mV) = (10000.0/4096) * code - 5000.00
        # ±1V: Volt(mV) = (2000.0/4096) * code - 1000.00
        voltage_data_mv[start:end] = (
            codes * (range_span[None, :] / 4096.0) - range_offset[None, :]
        ).ravel()

        print(f"  转换进度: {end}/{n_use}")

    # mV -> V
    voltage_data = voltage_data_mv * 1e-3

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


def read_csv_to_voltage(file_path, chunk_size=50000, sampling_rate=None):
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
            # 公式: Volt(V) = (code - 2048) * 10 / 4096
            voltage = (rawdata - 2048) * 10 / 4096

            current_chunk_size = len(voltage)
            voltage_data[row_counter:row_counter + current_chunk_size, :] = voltage

            row_counter += current_chunk_size
            pbar.update(current_chunk_size)

            del chunk, rawdata, voltage
            if row_counter % (chunk_size * 10) == 0:
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

def calculate_spectrogram(data, fs, time_block_duration, target_freq, span,
                         window='hann', overlap=0.5, detrend='constant'):
    """
    计算数据的频谱图（时频分析）

    参数:
        data: 输入数据 (1D numpy数组)
        fs: 采样率 (Hz)
        time_block_duration: 每个时间块的长度 (秒)
        target_freq: 目标频率 (Hz)
        span: 频率范围 (Hz)
        window: 窗函数类型
        overlap: 重叠比例 (0-1)
        detrend: 去趋势方法

    返回:
        dict: 包含频谱图数据、时间轴、频率轴等信息
    """
    samples_per_block = int(time_block_duration * fs)
    step = int(samples_per_block * (1 - overlap))
    n_blocks = (len(data) - samples_per_block) // step + 1

    freq_axis = fftfreq(samples_per_block, 1/fs)[:samples_per_block // 2]
    freq_mask = (freq_axis >= target_freq - span/2) & (freq_axis <= target_freq + span/2)
    freq_selected = freq_axis[freq_mask]
    n_freqs = np.sum(freq_mask)

    print(f"\n频谱分析参数:")
    print(f"  - 时间块长度: {time_block_duration*1000:.2f} ms ({samples_per_block} 采样点)")
    print(f"  - 重叠率: {overlap*100:.0f}%")
    print(f"  - 时间块数: {n_blocks:,}")
    print(f"  - 目标频率: {target_freq/1000:.1f} kHz")
    print(f"  - 频率范围: [{target_freq - span/2:.0f}, {target_freq + span/2:.0f}] Hz")
    print(f"  - 频率点数: {n_freqs}")

    spectrogram = np.zeros((n_blocks, n_freqs), dtype=np.float32)

    # 选择窗函数
    windows = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'rectangular': np.ones
    }
    win_func = windows.get(window, np.hanning)
    win = win_func(samples_per_block)
    win = win / np.sqrt(np.mean(win**2))

    print("\n计算频谱图...")
    with tqdm(total=n_blocks, desc="处理进度") as pbar:
        for i in range(n_blocks):
            start = i * step
            end = start + samples_per_block
            block = data[start:end]

            if detrend == 'constant':
                block = block - np.mean(block)
            elif detrend == 'linear':
                block = signal.detrend(block)

            block = block * win
            spectrum = fft(block)
            psd = np.abs(spectrum) ** 2 / (fs * samples_per_block)
            psd = psd[:samples_per_block//2]

            spectrogram[i, :] = psd[freq_mask]
            pbar.update(1)

    time_axis = np.arange(n_blocks) * step / fs + samples_per_block / (2 * fs)

    return {
        'spectrogram': spectrogram,
        'time_axis': time_axis,
        'freq_axis': freq_selected,
        'freq_mask': freq_mask,
        'n_blocks': n_blocks,
        'n_freqs': n_freqs
    }


def plot_combined(voltage, count, fs, spectrogram_data, target_freq, span,
                  title=None, save_path=None, n_plot=100000,
                  vmin=None, vmax=None, cmap='jet'):
    """
    绘制合并图:
      第一行: 时间-电压波形 | 全时段频谱
      第二行: 时频瀑布图
    """
    spectrogram = spectrogram_data['spectrogram']
    time_axis = spectrogram_data['time_axis']
    freq_axis = spectrogram_data['freq_axis']

    spectrogram_db = 10 * np.log10(spectrogram + 1e-12)

    if vmin is None:
        vmin = np.percentile(spectrogram_db, 5)
    if vmax is None:
        vmax = np.percentile(spectrogram_db, 95)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_tf = fig.add_subplot(gs[1, :])

    if title:
        fig.suptitle(title, fontsize=14)

    # 时间-电压波形
    t_start = 50100000
    t_end = 50200000  # s
    t_axis = np.arange(0, len(voltage)) / fs
    # ax_wave.plot(t_axis[t_start:t_end:n_plot], voltage[t_start:t_end:n_plot], linewidth=0.5)
    ax_wave.plot(t_axis[::n_plot], voltage[::n_plot], linewidth=0.5)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Voltage (V)")
    ax_wave.set_title(f"SampleRate {fs/1e6:.0f} MHz, every {n_plot} samples ({len(t_axis):,} pts)")
    ax_wave.grid(True, alpha=0.3)

    # 全时段频谱（能量随时间积分）
    energy_spectrum = np.sum(spectrogram, axis=0)
    energy_spectrum_db = 10 * np.log10(energy_spectrum + 1e-12)
    ax_spec.plot(freq_axis, energy_spectrum_db, linewidth=0.5, color='blue')
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Integrated Energy (dB)')
    ax_spec.set_title(f'Full-Period Spectrum (Target: {target_freq/1000:.1f} kHz)')
    ax_spec.grid(True, alpha=0.3)
    ax_spec.axvline(x=target_freq, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 时频瀑布图
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    im = ax_tf.imshow(spectrogram_db.T, aspect='auto', origin='lower',
                      extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_tf.set_xlabel('Time (s)')
    ax_tf.set_ylabel('Frequency (Hz)')
    ax_tf.set_title(f'Spectrogram (Target: {target_freq/1000:.1f} kHz, Span: {span/1000:.1f} kHz)')
    ax_tf.grid(True, alpha=0.2, linestyle='--')
    ax_tf.axhline(y=target_freq, color='white', linestyle='--', linewidth=1, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax_tf, label='Power Spectral Density (dB)')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}_analysis.jpg', dpi=300, bbox_inches='tight')
        print(f'Combined figure saved to: {save_path}_analysis.jpg')

    plt.show()
    return fig


def extract_peak_frequency(spectrogram_data):
    """
    提取每个时间块的最大频率点
    """
    spectrogram = spectrogram_data['spectrogram']
    freq_axis = spectrogram_data['freq_axis']

    peak_indices = np.argmax(spectrogram, axis=1)
    peak_freqs = freq_axis[peak_indices]
    peak_amps = np.max(spectrogram, axis=1)

    return peak_freqs, peak_amps


def _mean_std(data, chunk=5_000_000):
    """分块计算均值和标准差，避免 np.std 产生 O(n) 临时数组导致内存不足"""
    n = data.size
    mean = float(np.mean(data))
    ss = 0.0
    for start in range(0, n, chunk):
        diff = data[start:start + chunk].astype(np.float64)
        diff -= mean
        ss += float(np.dot(diff, diff))
    return mean, (ss / n) ** 0.5


def print_data_statistics(voltage_data, channels=None):
    """打印数据统计信息"""
    if voltage_data.ndim == 1:
        voltage_data = voltage_data.reshape(-1, 1)

    n_channels = voltage_data.shape[1]
    if channels is None:
        channels = [f'CH{i+1}' for i in range(n_channels)]

    print("\n" + "="*60)
    print("数据统计信息")
    print("="*60)

    for i, ch in enumerate(channels):
        data = voltage_data[:, i]
        mean, std = _mean_std(data)
        print(f"{ch:8s}: 均值={mean:8.4f}V, 标准差={std:8.4f}V, "
              f"最小值={np.min(data):8.4f}V, 最大值={np.max(data):8.4f}V, "
              f"峰峰值={np.ptp(data):8.4f}V")
    print("="*60)


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

    range_span = 10000.0  # ±5V = 10000mV
    range_offset = 5000.0

    # print(f"\n量程: ±5V")
    # print(f"公式: Volt(mV) = ({range_span}/4096) * code - {range_offset}")
    # print(f"      Volt(V)  = Volt(mV) / 1000")
    # print("\n" + "-" * 70)
    # print(f"{'码值(hex)':<12} {'描述':<15} {'code':<8} {'理论值(mV)':<12} {'计算值(V)':<12} {'误差(V)':<12}")
    # print("-" * 70)

    for hex_val, desc, code, expected_mv in test_cases:
        calculated_mv = (range_span / 4096.0) * code - range_offset
        calculated_v = calculated_mv / 1000.0
        expected_v = expected_mv / 1000.0
        error = calculated_v - expected_v
        print(f"0x{hex_val:03X}     {desc:<15} {code:<8} {expected_mv:<12.3f} {calculated_v:<12.6f} {error:<12.6f}")

    print("=" * 60)


# ========== 主程序 ==========
if __name__ == "__main__":
    # # 先验证转换公式
    # verify_conversion()

    # 文件路径
    bin_file = r"E:\system\data\1550\ACTS1000_data_100Msps_1s_10kS_20260821_20260821135631696_0.bin"

    # 时频分析参数
    TIME_BLOCK = 0.001     # 时间块长度 s
    CENTER_FREQ = 0         # 中心频率 Hz
    TARGET_FREQ = 10000      # 目标频率 Hz
    SPAN = TARGET_FREQ * 10            # 频率范围 Hz
    OVERLAP = 0.5            # 重叠率：50%
    WINDOW = 'hann'          # 窗函数
    MAX_SAMPLES = 10_000_000 # 最大分析样本数
    N_PLOT = 1000            # 时频图，取样间隔

    file_path = os.path.dirname(bin_file)
    save_name = os.path.splitext(os.path.basename(bin_file))[0]
    save_path = os.path.join(file_path, save_name)

    try:
        file_type = os.path.splitext(bin_file)[1].lower()
        print(f"处理文件: {os.path.basename(bin_file)}")
        print("-" * 60)

        # ========== 1. 读取数据 ==========
        if file_type == '.csv':
            result = read_csv_to_voltage(bin_file, chunk_size=1000000, sampling_rate=100000000)
        elif file_type == '.bin':
            result = read_bin_to_voltage(bin_file)
        else:
            raise ValueError(f"不支持的文件格式: {file_type}")

        voltage = result['voltage_data']
        count = result['num_samples']
        fs = result['sampling_rate']
        channels = result.get('channels') or [f'CH{i+1}' for i in range(voltage.shape[1] if voltage.ndim > 1 else 1)]

        print(f"\n[成功] 读取完成:")
        print(f"  - 采样点数: {count:,}")
        print(f"  - 采样率: {fs/1e6:.0f} MHz")
        print(f"  - 通道数: {len(channels)}")
        print(f"  - 数据形状: {voltage.shape}")

        # 打印统计信息
        print_data_statistics(voltage, channels)

        # ========== 2. 选择分析通道（默认第一个通道） ==========
        if voltage.ndim == 1:
            data_to_plot = voltage
        else:
            data_to_plot = voltage[:, 0]
            print(f"\n分析通道: {channels[0]}")

        # ========== 3. 时频分析 ==========
        # 限制分析数据长度（避免内存溢出）
        if len(data_to_plot) > MAX_SAMPLES:
            print(f"\n数据太长，截取前 {MAX_SAMPLES:,} 个点进行分析")
            data_to_analyze = data_to_plot[:MAX_SAMPLES]
        else:
            data_to_analyze = data_to_plot

        # 计算频谱图
        spectrogram_data = calculate_spectrogram(
            data_to_analyze,
            fs=fs,
            time_block_duration=TIME_BLOCK,
            target_freq=TARGET_FREQ,
            span=SPAN,
            window=WINDOW,
            overlap=OVERLAP
        )

        # 绘制合并图（波形 + 全时段频谱 + 时频瀑布图）
        plot_combined(data_to_plot, count, fs, spectrogram_data,
                      TARGET_FREQ, SPAN, n_plot=N_PLOT,
                      title=os.path.basename(bin_file), save_path=save_path)

        # ========== 4. 提取峰值频率 ==========
        peak_freqs, peak_amps = extract_peak_frequency(spectrogram_data)

        print(f"\n峰值频率统计:")
        print(f"  - 平均频率: {np.mean(peak_freqs):.2f} Hz")
        print(f"  - 频率标准差: {np.std(peak_freqs):.2f} Hz")
        print(f"  - 频率范围: [{np.min(peak_freqs):.2f}, {np.max(peak_freqs):.2f}] Hz")

        print(f"\n处理完成！所有结果已保存到: {save_path}")

    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()