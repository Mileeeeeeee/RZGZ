"""
ts_signal_fft.py — 时频分析工具
================================
读取 ADC 二进制数据文件，执行 FFT 时频分析，生成振动分析图 (Vibration.jpg)。

主要功能:
  1. 读取 bin/csv 数据文件，转换为电压值
  2. FFT 时频分析 (时间块长度为 2^n，最长 2M 点)
  3. 振动分析图 (4行紧凑型): 抽样时域波形 / 时间-频率-能量瀑布图
     (各频点减时间中值的偏差量) / 有效频点能量-时间曲线 (异常时段定位) /
     频率-能量频谱 (检测参数找峰标注, 梳状干涉自动剔除)
  4. 导出原始 ADC 码值为 CSV 文件
"""

import os
import re
import time
from types import SimpleNamespace

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

# ================= 版本化参数 (class 模块: 公用参数合并, 差异参数区分) =================
# params.base():        所有版本公用参数 (文件路径 / 分析窗口 / 分块 / 输出 / 绘图)
# params.pcie_8912() / pcie_1840() / pcie_1840l(): 各采集卡差异化参数 (bin 解析)
# params.params_fft() / params_dc(): 各分析模式差异化参数 (振动频谱 / 光辐射直流), 派生量在方法内统一计算
# get_params_combine(): 公用 + 采集卡 + 分析模式 依次合并 (后者覆盖前者); 修改参数无需改动主逻辑代码。

class params():
    def base(self):
        """公用参数：
        数据读入路径
        数据分块处理参数：使用np.memmap防止数据过大，处理报错
        数据输出参数配置： # 解析后的电压数据保存为csv
        画图参数
        其他
        """
        return dict(
            # 数据读入路径
            DEFAULT_BIN_FILE = r"E:\system\data\1064_PV\2026092201\daqyh_sn2026092201_125000000_c1_20000_int16_20260922_100738_487_8300kw_15mms.bin.part0",
            # 分析时间段: 只分析 [ANALYSIS_START_S, ANALYSIS_START_S + DURATION_S) 内数据, None = 到文件末尾
            ANALYSIS_START_S = 0,
            ANALYSIS_DURATION_S= None,
            # 数据分块处理参数 (各分析模式内按窗口点数派生覆盖)
            MEAN_STD_CHUNK = 2_000_000,
            CHUNK_SAMPLES = 2_000_000,
            # 数据输出参数配置: 解析后的电压数据保存为csv
            EXPORT_CSV=False,
            # 画图参数
            N_PLOT = 1000,                   # 时域波形图抽样间隔: 每 N_PLOT 个点绘制 1 个点
            FIGURE_SIZE = (36, 12),         # 分析图尺寸 (宽, 高) 英寸, 3行紧凑型布局
            FIGURE_DPI = 300,               # 保存图片的 DPI
            DEFAULT_CMAP = 'jet',           # 瀑布图 colormap
            LINE_WIDTH = 0.5,               # 曲线默认线宽
            GRID_ALPHA = 0.3,               # 网格透明度
            SPECTRUM_ANNOTATION_FONT_SIZE = 6,   # 频谱峰值标注字体大小
            SPECTRUM_PEAK_COLOR = 'red',         # 频谱图检测峰值标注颜色
            STATS_SUFFIX = 'statistics.txt',     # 统计信息 txt 文件后缀
            SAVE_SUFFIX = 'Vibration.jpg',       # 保存图片文件名后缀
        )

    def pcie_8912(self):
        """500M数据采集卡
        bin文件后缀格式 *_0.bin
        参数从 * _header.txt中读取
        其他
        """
        return dict(
            BIN_FORMAT='acts1000',    # 解析模块: read_bin_to_voltage (参数按 _header.txt 读取)
            # bin 读取与码值转电压: uint16 码值, 掩码保留有效位, 按各通道量程线性转换
            ACTS_BIN_DTYPE='uint16',   # bin 文件数据类型 (2 字节/点)
            ACTS_BIN_HEADER_BYTES=0,   # bin 文件头部偏移 (字节)
            ADC_RESOLUTION=4096,       # ADC 量化级数 2^12 (码值转电压)
        )

    def pcie_1840(self):
        """125M数据采集卡
        bin文件后缀格式 *.bin.part0
        chanEnableCount: 1
        SampleRate:125_000_000.0
        0rangeMaxValue(mV):10_000.0
        0rangeMinValue(mV):-10_000.0
        resolution:16
        其他
        """
        return dict(
            BIN_FORMAT='pcie_1840',          # 解析模块: read_bin_pcie1840 (int16, XOR 0x8000, ±10V)
            PCI_SAMPLE_RATE=125_000_000,     # 文件名不含采样率时的默认采样率 (Hz)
            PCI_BIN_HEADER_BYTES=0,          # bin 文件头部偏移 (字节)
            PCI_BIN_DTYPE='int16',           # bin 文件数据类型 (2 字节/点)
            PCI_RESOLUTION=16,               # ADC 量化位数 (resolution)
            PCI_CODE_XOR=0x8000,             # 偏移二进制码翻转符号位 (还原二进制补码)
            PCI_RANGE_MAX_V=10.0,            # 量程上限 (0rangeMaxValue: 10_000 mV)
            PCI_RANGE_MIN_V=-10.0,           # 量程下限 (0rangeMinValue: -10_000 mV)
        )

    def pcie_1840l(self):
        """80M数据采集卡
        bin文件后缀格式 *.bin.part0
        chanEnableCount: 3
        SampleRate:5_000_000.0
        0rangeMaxValue(mV):10_000.0
        0rangeMinValue(mV):-10_000.0
        resolution:16
        其他
        """
        return dict(
            BIN_FORMAT='pcie_1840',
            PCI_SAMPLE_RATE=5_000_000,
            PCI_BIN_HEADER_BYTES=0,
            PCI_BIN_DTYPE='int16',
            PCI_RESOLUTION=16,
            PCI_CODE_XOR=0x8000,
            PCI_RANGE_MAX_V=10.0,
            PCI_RANGE_MIN_V=-10.0,
        )

    def params_fft(self):
        """1路数据，振动信号分析
        local_oscillator_center_freq: 40_000_000.0 # 本振频率，单位Hz
        fft_window_size:  2^21  # 每帧处理数据长度
        fft_step：16_000  # 当前帧数据长度，傅里叶变换数据重叠处理方法：fft_window_size = 前一帧尾数据长度 + 当前帧数据长度
        band_start: 100.0 # 检测频率起始点
        band_width: 105_000.0 # 检测带宽， band_start + band_width 就是最大检测频率
        measure_min_magnitude: 1.0  # 检测最小幅度，去除没有信号时，杂波影响
        magnitude_peak_base_ratio: 1.2 # 检测频率幅度与底噪幅度的最小比值
        magnitude_peak_base_diff: 0.8 # 检测频率幅度与底噪幅度的最小差值
        其他
        """
        p = dict(
            # 时频分析: 目标拍频 (本振光与信号光频差, 频谱分析中心频率)
            TARGET_FREQ = 40_000_000,   # local_oscillator_center_freq (Hz)
            # FFT_ORDER=21 → 2^21=2,097,152 点/块 → 时间分辨率 ~21ms, 频率分辨率 ~48Hz
            # FFT_ORDER=20 → 2^20=1,048,576 点/块 → 时间分辨率 ~10ms, 频率分辨率 ~95Hz
            # FFT_ORDER=19 → 2^19=524,288 点/块 → 时间分辨率 ~5ms, 频率分辨率 ~191Hz
            FFT_ORDER = 19,
            WINDOW_TYPE = 'hann',       # FFT 窗函数: 'hann', 'hamming', 'blackman', 'rectangular'
            DETREND_METHOD = 'constant',  # FFT 前去趋势: 'constant' (减均值), 'linear', None
            SPECTROGRAM_DB_FLOOR = 1e-12,         # dB 转换下限保护值，避免 log10(0)
            SPECTROGRAM_MAX_DISPLAY_ROWS = 100,  # 瀑布图渲染的最大频率行数: 超出时按行分组取最大值聚合
            # 瀑布图颜色范围: 按有效频点能量的分位数取 [下限, 上限] (下限≈底噪, 上限≈最强信号),
            #   避免全局 autoscale 把动态范围压扁 (整图同色)
            SPECTROGRAM_VMIN_PCT = 50,     # 颜色下限分位数 (%): 底噪附近, 其下渲染为深色
            SPECTROGRAM_VMAX_PCT = 99.9,   # 颜色上限分位数 (%): 最强信号附近, 其上图饱和
            WATERFALL_SPAN_HZ = None,      # 瀑布图纵轴显示频移范围 (Hz), 如 100_000 = ±100 kHz; None = 完整检测频带
            # 检测参数列表: [起始频率, 终止频率, 检测最小幅度(dB), 峰-底噪最小比值(dB), 峰-底噪最小差值(dB)]
            #   三项均为峰高出频带底噪(中值)的余量(dB), 须同时满足 → 有效阈值 = 三项最大值 (CFAR 式相对检测)
            DETECT_PARAMETERS=[
                [200, 1_000, 6, 1.2, 0.8],
            ],
            PEAK_DISTANCE_BINS = 3,   # 找峰最小间隔 (单位: 频率分辨率 bin)
            # 梳状干涉判别 (电源/时钟谐波): 等间隔 + 全时段持续 → 从振动分析中剔除
            COMB_DETECT_ENABLE = True,     # 启用梳状干涉自动判别与剔除
            COMB_MIN_TEETH = 3,            # 判为梳状的最少齿数 (检测峰个数)
            COMB_DUTY_MIN = 0.8,           # 干涉时间占用率下限: 齿能量超阈的时间块比例 (振动为间歇, 占用率低)
            COMB_ALIGN_TOL_BINS = 20,       # 齿频率对齐容差 (频率分辨率 bin 数)
        )
        # 派生参数: 时间窗口 = N_FFT / 采样率, 频率分辨率 = 采样率 / N_FFT
        p['N_FFT'] = 1 << p['FFT_ORDER']                    # FFT 窗口点数 (fs=125M 时 ≈ 17ms/帧)
        p['FFT_STEP'] = p['N_FFT'] // 5                     # 帧移 (80% 重叠) = 前一帧尾数据长度 + 当前帧数据长度
        p['FREQ_SPAN'] = max(f1 for _, f1, *_ in p['DETECT_PARAMETERS'])  # 分析频移范围, 由检测频带派生
        p['MEAN_STD_CHUNK'] = p['N_FFT']                    # 时域分析与频域分析使用同样的窗口点数
        p['CHUNK_SAMPLES'] = p['N_FFT']
        return p

    def params_dc(self):
        """3路数据，光辐射信号分析
        dc_window_time： 0.5 每帧处理数据长度
        其他
        """
        return dict(
            DC_WINDOW_TIME=0.5,     # 每帧处理数据长度 (s)
        )


def get_params_combine():
    """参数合并: 公用参数 + 采集卡参数 + 分析参数 (dict 依次合并, 后者覆盖前者)"""
    p = params()
    combine = {}
    for section in (p.base, p.pcie_1840, p.params_fft):
        combine.update(section())
    return SimpleNamespace(**combine)

cfg = get_params_combine()
# ========== 工具函数 ==========

def _fmt_freq(hz):
    """按频谱仪显示习惯格式化频率: >= 1 MHz 用 MHz，否则用 kHz"""
    return f"{hz/1e6:.3f} MHz" if abs(hz) >= 1e6 else f"{hz/1e3:.3f} kHz"


def _fmt_hz(hz):
    """紧凑格式化频率: >=1MHz 用 MHz, >=1kHz 用 kHz, 否则 Hz"""
    if abs(hz) >= 1e6:
        return f"{hz/1e6:g} MHz"
    if abs(hz) >= 1e3:
        return f"{hz/1e3:g} kHz"
    return f"{hz:g} Hz"


def _fmt_shift(hz):
    """紧凑格式化相对中心频率的频移 (带符号)"""
    return ('+' if hz >= 0 else '-') + _fmt_hz(abs(hz))


def _mean_std(data, chunk=None):
    """分块计算均值和标准差，避免 np.std 产生 O(n) 临时数组导致内存不足"""
    if chunk is None:
        chunk = cfg.MEAN_STD_CHUNK
    data = data.ravel()
    n = data.size
    mean = float(np.mean(data))
    ss = 0.0
    for start in range(0, n, chunk):
        diff = data[start:start + chunk].astype(np.float64)
        diff -= mean
        ss += float(np.dot(diff, diff))
    return mean, (ss / n) ** 0.5


# ========== 文件头解析 ==========

def _get_info_from_header_txt(bin_file_path):
    """从同名 _header.txt 解析 bin 文件的参数 (ACTS1000 界面保存的 bin 无文件头)。"""
    stem = os.path.splitext(bin_file_path)[0]
    stem = re.sub(r"_\d{17}(_\d+)?$", "", stem)
    header_file = stem + "_header.txt"
    if not os.path.exists(header_file):
        raise FileNotFoundError(f"找不到头部参数文件: {header_file}")

    params = {}
    with open(header_file, encoding="utf-8") as fp:
        for line in fp:
            key, _, value = line.strip().partition(":")
            params[key] = value

    channel_count = int(params["chanEnableCount"])
    resolution = int(params["resolution"])
    w_max_lsb = (1 << resolution) - 1

    range_span = np.empty(channel_count, dtype=np.float64)
    range_offset = np.empty(channel_count, dtype=np.float64)
    range_mv = np.empty(channel_count, dtype=np.float64)

    for ch in range(channel_count):
        range_max = float(params[f"{ch}rangeMaxValue"])
        range_min = float(params[f"{ch}rangeMinValue"])
        range_span[ch] = range_max - range_min
        range_offset[ch] = range_max
        range_mv[ch] = range_max

    return {
        "channel_count": channel_count,
        "range_span": range_span,
        "range_offset": range_offset,
        "w_max_lsb": w_max_lsb,
        "sample_rate": float(params["SampleRate"]),
        "range_mv": range_mv,
        "channel_range": np.array([f"±{int(rv / 1000)}V" for rv in range_mv])
    }


# ========== 分析窗口 ==========

def _analysis_window_indices(fs, total):
    """按 cfg.ANALYSIS_START_S / cfg.ANALYSIS_DURATION_S 计算分析窗口的索引区间 [i0, i1)。"""
    i0 = int(round(cfg.ANALYSIS_START_S * fs))
    if cfg.ANALYSIS_DURATION_S is None:
        i1 = total
    else:
        i1 = i0 + int(round(cfg.ANALYSIS_DURATION_S * fs))
    i0 = max(0, min(i0, total))
    i1 = max(i0, min(i1, total))
    if i1 <= i0:
        raise ValueError(
            f"分析时间段 [{cfg.ANALYSIS_START_S}, "
            f"{'文件末尾' if cfg.ANALYSIS_DURATION_S is None else cfg.ANALYSIS_START_S + cfg.ANALYSIS_DURATION_S}] "
            f"内无有效数据 (数据共 {total:,} 点, 时长 {total / fs:.3f} s)，"
            f"请检查 ANALYSIS_START_S / ANALYSIS_DURATION_S")
    return i0, i1


# ========== 数据读取 ==========

def read_bin_to_voltage(bin_file_path):
    """
    读取 bin 文件并转换为电压值，同时返回原始 ADC 码值 (用于 CSV 导出)。

    转换公式 (PCIe8922M 协议):
        Volt(mV) = (range_span / ADC_RESOLUTION) * (code & wMaxLSB) - range_offset
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"文件不存在: {bin_file_path}")

    info = _get_info_from_header_txt(bin_file_path)
    print("从 _header.txt 解析参数")

    header_bytes = cfg.ACTS_BIN_HEADER_BYTES
    channel_count = info["channel_count"]
    range_span = info["range_span"]
    range_offset = info["range_offset"]
    w_max_lsb = info["w_max_lsb"]
    itemsize = np.dtype(cfg.ACTS_BIN_DTYPE).itemsize

    print("=" * 60)
    print(f"bin 数据读取 ({cfg.ACTS_BIN_DTYPE}, {itemsize}字节/点)")
    print("=" * 60)
    print(f"通道数: {channel_count}, 量程: {info.get('channel_range', ['未知'])}")

    file_size = os.path.getsize(bin_file_path)
    total_samples = round((file_size - header_bytes) / itemsize)
    fs = info["sample_rate"]
    total_groups = total_samples // channel_count
    print(f"文件大小: {file_size / (1024 ** 2):.1f} MB, 每通道 {total_groups:,} 点, "
          f"时长 {total_groups / fs:.3f} s")

    g0, g1 = _analysis_window_indices(fs, total_groups)
    win_i0, win_i1 = g0 * channel_count, g1 * channel_count
    n_use = win_i1 - win_i0
    print(f"分析窗口: [{g0 / fs:.3f}, {g1 / fs:.3f}] s, {n_use:,} 点")

    raw_data = np.memmap(bin_file_path, dtype=cfg.ACTS_BIN_DTYPE, mode='r',
                         offset=header_bytes + win_i0 * itemsize, shape=(n_use,))

    # 码值转电压
    voltage_data = np.empty(n_use, dtype=np.float32)
    scale_mv = (range_span / cfg.ADC_RESOLUTION).astype(np.float32)
    offset_mv = range_offset.astype(np.float32)
    chunk = cfg.CHUNK_SAMPLES - cfg.CHUNK_SAMPLES % channel_count

    # 保存原始码值用于 CSV 导出
    raw_codes = np.empty((n_use // channel_count, channel_count), dtype=np.uint16)

    for start in range(0, n_use, chunk):
        end = min(start + chunk, n_use)
        block = raw_data[start:end].reshape(-1, channel_count)
        codes = block & w_max_lsb
        raw_codes[start // channel_count:end // channel_count] = codes.astype(np.uint16)
        voltage_data[start:end] = (
            codes.astype(np.float32) * scale_mv[None, :] - offset_mv[None, :]
        ).ravel()
        print(f"  转换进度: {end}/{n_use}")

    voltage_data *= 1e-3  # mV -> V

    mean_v, std_v = _mean_std(voltage_data)
    print(f"电压范围: [{voltage_data.min():.4f}, {voltage_data.max():.4f}] V, "
          f"均值={mean_v:.4f} V, 标准差={std_v:.4f} V")

    return {
        'voltage_data': voltage_data,
        'num_samples': n_use,
        'sampling_rate': fs,
        'channels': info.get('channel_range', [f'CH{i + 1}' for i in range(channel_count)]),
        'shape': (n_use, channel_count),
        'window_start_s': g0 / fs,
        'window_duration_s': (g1 - g0) / fs,
        'range_span': range_span,
        'range_offset': range_offset,
        'w_max_lsb': w_max_lsb,
        'channel_count': channel_count,
        'raw_codes': raw_codes,  # 原始 ADC 码值 (n_groups, n_channels)
    }


# ========== PCIe-1840L 新采集卡 bin 解析 ==========

def _detect_pcie_sample_rate(bin_file_path, default):
    """从文件名 (daqyh_snXXXX_5000000_...) 提取采样率, 失败时用默认值。"""
    m = re.search(r'sn\d+?_(\d+?)_', os.path.basename(bin_file_path))
    if m:
        rate = float(m.group(1))
        if rate in (5_000_000.0, 125_000_000.0):
            return rate
    return default


def read_bin_pcie1840(bin_file_path):
    """
    读取 PCIe-1840L 采集卡 16bit bin 文件 (.bin.part0/1/2) 并转换为电压值。

    数据格式与转换参数由 cfg 提供 (PCI_BIN_DTYPE/PCI_RESOLUTION/PCI_CODE_XOR/
    PCI_RANGE_MAX_V/PCI_RANGE_MIN_V): 偏移二进制码与 PCI_CODE_XOR 异或还原为
    二进制补码, 再乘 (量程跨度)/2^分辨率 (V/LSB) 得到电压。
    """
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"文件不存在: {bin_file_path}")

    fs = _detect_pcie_sample_rate(bin_file_path, cfg.PCI_SAMPLE_RATE)
    file_size = os.path.getsize(bin_file_path)
    itemsize = np.dtype(cfg.PCI_BIN_DTYPE).itemsize
    total_samples = (file_size - cfg.PCI_BIN_HEADER_BYTES) // itemsize
    if total_samples <= 0:
        raise ValueError(f"文件过小或头部偏移过大 ({file_size} 字节, "
                         f"头部偏移 {cfg.PCI_BIN_HEADER_BYTES} 字节): {bin_file_path}")

    i0, i1 = _analysis_window_indices(fs, total_samples)
    n_use = i1 - i0

    print("=" * 60)
    print(f"bin 数据读取 (PCIe-1840L, {cfg.PCI_BIN_DTYPE}, XOR 0x{cfg.PCI_CODE_XOR:04X})")
    print("=" * 60)
    print(f"采样率: {fs/1e6:g} MHz, 总点数: {total_samples:,}, 时长 {total_samples/fs:.3f} s")
    print(f"分析窗口: [{i0/fs:.3f}, {i1/fs:.3f}] s, {n_use:,} 点")

    raw = np.memmap(bin_file_path, dtype=cfg.PCI_BIN_DTYPE, mode='r',
                    offset=cfg.PCI_BIN_HEADER_BYTES + i0 * itemsize, shape=(n_use,))

    voltage_data = np.empty(n_use, dtype=np.float32)
    raw_codes = np.empty((n_use, 1), dtype=np.uint16)
    code_xor = np.uint16(cfg.PCI_CODE_XOR)
    scale_v = np.float32((cfg.PCI_RANGE_MAX_V - cfg.PCI_RANGE_MIN_V) / (1 << cfg.PCI_RESOLUTION))
    chunk = cfg.CHUNK_SAMPLES

    for start in range(0, n_use, chunk):
        end = min(start + chunk, n_use)
        block = raw[start:end]
        # 偏移二进制码: 异或翻转符号位还原补码; 按位等价改写为 uint16 视图, 避免 int16(0x8000) 溢出
        voltage_data[start:end] = ((block.view(np.uint16) ^ code_xor).view(np.int16)).astype(np.float32) * scale_v
        raw_codes[start:end, 0] = block.view(np.uint16)
        print(f"  转换进度: {end}/{n_use}")

    mean_v, std_v = _mean_std(voltage_data)
    print(f"电压范围: [{voltage_data.min():.4f}, {voltage_data.max():.4f}] V, "
          f"均值={mean_v:.4f} V, 标准差={std_v:.4f} V")

    return {
        'voltage_data': voltage_data,
        'num_samples': n_use,
        'sampling_rate': fs,
        'channels': [f"±{cfg.PCI_RANGE_MAX_V:g}V"],
        'shape': (n_use, 1),
        'window_start_s': i0 / fs,
        'window_duration_s': n_use / fs,
        'range_span': np.array([cfg.PCI_RANGE_MAX_V - cfg.PCI_RANGE_MIN_V]),
        'range_offset': np.array([0.0]),
        'w_max_lsb': (1 << cfg.PCI_RESOLUTION) - 1,
        'channel_count': 1,
        'raw_codes': raw_codes,  # 原始 16bit 码值位模式 (n, 1)
    }


def load_data(file_path):
    """按 cfg.BIN_FORMAT 选择 bin 解析模块 (保留旧卡解析, 可随时切换)。"""
    if re.search(r'\.bin(\.part\d+)?$', file_path):
        if cfg.BIN_FORMAT == 'pcie_1840' or (
                cfg.BIN_FORMAT == 'auto' and re.search(r'\.part\d+$', file_path)):
            return read_bin_pcie1840(file_path)
        return read_bin_to_voltage(file_path)
    raise ValueError(f"不支持的文件格式: {file_path}")


# ========== CSV 导出 ==========

def export_rawdata_csv(result, save_path):
    """将原始 ADC 码值 (uint16) 导出为 CSV 文件，包含 time_s 列。"""
    # fs = result['sampling_rate']
    raw_codes = result['raw_codes']
    n_groups = raw_codes.shape[0]
    channel_count = raw_codes.shape[1] if raw_codes.ndim > 1 else 1

    # time_s = np.arange(n_groups, dtype=np.float64) / fs

    # data = {'time_s': time_s}
    data = {}
    if channel_count == 1:
        data['raw'] = raw_codes.ravel()
    else:
        for ch in range(channel_count):
            data[f'CH{ch+1}'] = raw_codes[:, ch]

    df = pd.DataFrame(data)
    csv_path = f'{save_path}_rawdata.csv'
    df.to_csv(csv_path, index=False)
    print(f"原始码值已导出: {csv_path} ({n_groups:,} 行)")
    return csv_path


# ========== 时频分析 ==========

def calculate_spectrogram(data, fs, n_fft=None, center_freq=cfg.TARGET_FREQ,
                          span=cfg.FREQ_SPAN, window=cfg.WINDOW_TYPE,
                          step=None, overlap=None, detrend=cfg.DETREND_METHOD,
                          time_block_duration=None):
    """
    计算频谱图 (时频分析)

    参数:
        data: 输入数据 (1D numpy数组)
        fs: 采样率 (Hz)
        n_fft: FFT 块长度 (采样点数)，默认取 cfg.N_FFT (2^FFT_ORDER)
        center_freq: 目标拍频 (Hz)
        span: 频移范围 (Hz)
        window: 窗函数类型
        step: 相邻时间块起始位置间隔 (采样点数)，默认取 cfg.FFT_STEP
        overlap: 相邻时间块重叠比例 (0-1)，与 step 同时给出时以 step 为准
        detrend: 去趋势方法
        time_block_duration: (兼容旧接口) 时间块长度 (s)，传入时转换为 n_fft

    返回:
        dict: spectrogram, time_axis, freq_axis, n_blocks, n_freqs
    """
    if time_block_duration is not None:
        n_fft = int(round(time_block_duration * fs))
    if n_fft is None:
        n_fft = cfg.N_FFT
    if step is None:
        if overlap is not None:
            step = max(1, int(round(n_fft * (1 - overlap))))
        else:
            step = cfg.FFT_STEP

    samples_per_block = n_fft
    n_blocks = (len(data) - samples_per_block) // step + 1
    if n_blocks < 1:
        raise ValueError(
            f"数据长度 ({len(data)}) 不足以构成一个时间块 ({samples_per_block:,} 采样点)")

    freq_axis = rfftfreq(samples_per_block, 1/fs)
    freq_mask = (freq_axis >= center_freq - span) & (freq_axis <= center_freq + span)
    freq_selected = freq_axis[freq_mask]
    n_freqs = freq_selected.size
    if n_freqs == 0:
        raise ValueError(
            f"频率范围 [{center_freq - span:.0f}, {center_freq + span:.0f}] Hz 内无频率点，"
            f"请检查 TARGET_FREQ/FREQ_SPAN 与采样率 (fs={fs:.0f} Hz, Nyquist={fs/2:.0f} Hz)")

    freq_resolution = fs / samples_per_block
    time_block_duration = samples_per_block / fs
    print(f"\n频谱分析参数:")
    print(f"  FFT 块长度: {samples_per_block:,} 点")
    print(f"  时间块长度: {time_block_duration*1000:.2f} ms")
    print(f"  频率分辨率: {freq_resolution:.1f} Hz")
    print(f"  块步进: {step:,} 点 (重叠率 {(1 - step/samples_per_block)*100:.0f}%)")
    print(f"  时间块数: {n_blocks:,}")
    print(f"  目标拍频: {_fmt_freq(center_freq)}, 频移范围: ±{span/1000:.1f} kHz")
    print(f"  分析频点数: {n_freqs}")

    # 窗函数（归一化到单位 RMS，float32 以加速 FFT）
    windows = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'rectangular': np.ones
    }
    win = windows.get(window, np.hanning)(samples_per_block)
    win = (win / np.sqrt(np.mean(win ** 2))).astype(np.float32)

    # 有效数据门限: voltage > 0 为有效采样点, 无效采样点置零不参与 FFT
    valid_mask = data > 0

    # 滑动窗口视图 + 分块向量化 FFT
    block_views = sliding_window_view(data, samples_per_block)[::step]
    mask_views = sliding_window_view(valid_mask, samples_per_block)[::step]
    chunk_blocks = max(1, int(512 * 1e6 // (16 * samples_per_block)))
    spectrogram = np.empty((n_blocks, n_freqs), dtype=np.float32)
    scale = np.float32(2.0 / (fs * samples_per_block))
    workers = min(os.cpu_count() or 1, 16)   # 多线程 FFT (pocketfft 复用 FFT 计划)

    print("\n计算频谱图...")
    with tqdm(total=n_blocks, desc="处理进度") as pbar:
        for start in range(0, n_blocks, chunk_blocks):
            end = min(start + chunk_blocks, n_blocks)
            # 仅拷贝一次, 之后原地去趋势/加窗, rfft(overwrite_x=True) 不再二次拷贝
            block = np.ascontiguousarray(block_views[start:end])

            if detrend == 'constant':
                block -= block.mean(axis=1, keepdims=True)
            elif detrend == 'linear':
                block = signal.detrend(block, axis=1).astype(np.float32)

            block *= win
            block *= mask_views[start:end]   # 无效采样点 (voltage<=0) 置零
            psd = np.abs(rfft(block, n=samples_per_block, axis=1,
                              workers=workers, overwrite_x=True))
            psd *= psd
            psd *= scale
            spectrogram[start:end] = psd[:, freq_mask]
            pbar.update(end - start)

    time_axis = np.arange(n_blocks) * step / fs + samples_per_block / (2 * fs)

    return {
        'spectrogram': spectrogram,
        'time_axis': time_axis,
        'freq_axis': freq_selected,
        'freq_mask': freq_mask,
        'n_blocks': n_blocks,
        'n_freqs': n_freqs,
        'n_fft': samples_per_block,
        'freq_resolution': freq_resolution,
        # 幅值换算: PSD峰值 = A^2*G^2*N/(2*fs) (G=sum(win)/N 为窗相干增益)
        #   → 正弦幅值 A = amp_scale * sqrt(PSD峰值), 用于检测最小幅度(V) 阈值
        'amp_scale': np.float32(np.sqrt(2.0 * fs / samples_per_block) * (samples_per_block / win.sum())),
    }


# ========== 绘图 ==========

def _setup_freq_axis_scale(ax, freq_axis, center_freq):
    """设置频率轴刻度格式 (显示为相对中心频率的频移)"""
    # 瀑布图纵轴数据本身已是频移 (freq - center_freq), 直接格式化, 不可再减 center_freq
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_shift(v)))


def detect_spectrum_peaks(env_db, env_v, freq_axis, center_freq, detect_params,
                          verbose=True):
    """按检测参数列表在频谱包络上找峰 (上下边带镜像对称检测)。

    detect_params 每项: [起始频率, 终止频率, 检测最小幅度(dB), 峰-底噪最小比值(dB), 峰-底噪最小差值(dB)],
    起始/终止频率为相对目标拍频的偏移量; 峰须同时满足三项阈值 (均为高出频带底噪中值的余量 dB)
    及找峰最小间隔 (PEAK_DISTANCE_BINS)。
    镜像对称: 振动边带物理对称, 任一侧通过的峰, 其在另一侧 ±tol 内的镜像峰一并返回
    (弱侧镜像可能单独不达阈值, 仍标注以保持上下边带成对)。
    返回 [(freq_hz, offset_hz, peak_db, peak_v), ...], 按能量降序。
    verbose=True 时向控制台打印每频带诊断 (底噪/候选数/最强候选及拒绝原因), 便于整定阈值。
    """
    peaks_out = []
    dist = max(1, int(cfg.PEAK_DISTANCE_BINS))
    bin_hz = float(freq_axis[1] - freq_axis[0]) if freq_axis.size > 1 else 1.0
    tol_hz = dist * bin_hz          # 镜像配对容差 (找峰最小间隔对应的频率宽度)
    for start_f, end_f, min_amp_db, min_ratio_db, min_diff_db in detect_params:
        band_cands = {1: [], -1: []}
        for sign, side in ((1, '上边带'), (-1, '下边带')):
            f_lo = center_freq + sign * min(start_f, end_f)
            f_hi = center_freq + sign * max(start_f, end_f)
            mask = (freq_axis >= f_lo) & (freq_axis <= f_hi)
            if mask.sum() < 3:
                continue
            band_freq = freq_axis[mask]
            band_db = env_db[mask]
            band_v = env_v[mask]
            floor_db = float(np.median(band_db))   # 底噪: 频带内包络中值
            idx, _ = signal.find_peaks(band_db, distance=dist)
            n_pass = 0
            cand = []
            for i in idx:
                margin = float(band_db[i] - floor_db)
                fails = []
                if margin < min_amp_db:
                    fails.append(f"高出底噪{margin:.1f}dB<最小幅度{min_amp_db:g}dB")
                if margin < min_ratio_db:
                    fails.append(f"高出底噪{margin:.1f}dB<比值{min_ratio_db:g}dB")
                if margin < min_diff_db:
                    fails.append(f"高出底噪{margin:.1f}dB<差值{min_diff_db:g}dB")
                cand.append((float(band_db[i]), float(band_freq[i]), float(band_v[i]), fails))
                if not fails:
                    n_pass += 1
            band_cands[sign] = cand
            cand.sort(key=lambda c: -c[0])
            line = (f"  [{_fmt_hz(start_f)}–{_fmt_hz(end_f)}] {side}: 底噪 {floor_db:.1f} dB, "
                    f"候选 {len(cand)} 峰, 通过 {n_pass}")
            if cand:
                db, f, v, fails = cand[0]
                if fails:
                    line += (f"; 最强候选 {_fmt_shift(f - center_freq)} {db:.1f} dB / {v:.3g} V "
                             f"被拒: {'、'.join(fails)}")
                else:
                    line += f"; 最强候选 {_fmt_shift(f - center_freq)} {db:.1f} dB / {v:.3g} V (通过)"
            if verbose:
                print(line)
        # 镜像对称标注: 任一侧通过的峰, 另一侧 ±tol 内的镜像峰一并标注
        for sign in (1, -1):
            other = -sign
            for db, f, v, fails in band_cands[sign]:
                if fails:
                    continue
                peaks_out.append((f, f - center_freq, db, v))
                o_abs = abs(f - center_freq)
                o_cands = band_cands[other]
                if not o_cands:
                    continue
                j = int(np.argmin([abs(abs(cf - center_freq) - o_abs)
                                   for _, cf, _, _ in o_cands]))
                cdb, cf, cv, _ = o_cands[j]
                if abs(abs(cf - center_freq) - o_abs) <= tol_hz:
                    mirror = (cf, cf - center_freq, cdb, cv)
                    if mirror not in peaks_out:
                        peaks_out.append(mirror)
    peaks_out.sort(key=lambda p: -p[2])
    return peaks_out


def detect_valid_frequencies(env_db, freq_axis, center_freq, detect_params):
    """按检测参数判定有效频率 (上下边带对称)。

    detect_params 每项: [起始频率, 终止频率, 检测最小幅度(dB), 峰-底噪最小比值(dB), 峰-底噪最小差值(dB)]。
    频点须同时满足: 包络-底噪(dB) >= 检测最小幅度(dB),
    且 >= 峰-底噪最小比值(dB), 且 >= 峰-底噪最小差值(dB);
    底噪 = 检测频带内包络中值。三项均为相对底噪的余量 → 有效阈值 = 三项最大值。
    返回 (row_valid, band_details):
      row_valid: 与 env_db 等长的有效频率 bool 掩码
      band_details: [(频段, 边带, 底噪dB, 最小幅度dB, 最小比值dB, 最小差值dB, 有效频点数, 频带频点数)]
    """
    row_valid = np.zeros(env_db.shape[0], dtype=bool)
    band_details = []
    for start_f, end_f, min_amp_db, min_ratio_db, min_diff_db in detect_params:
        for sign, side in ((1, '上边带'), (-1, '下边带')):
            f_lo = center_freq + sign * min(start_f, end_f)
            f_hi = center_freq + sign * max(start_f, end_f)
            mask = (freq_axis >= f_lo) & (freq_axis <= f_hi)
            if mask.sum() < 3:
                continue
            band_db = env_db[mask]
            floor_db = float(np.median(band_db))   # 底噪: 频带内包络中值
            margin_db = band_db - floor_db
            valid = (margin_db >= min_amp_db) & (margin_db >= min_ratio_db) & (margin_db >= min_diff_db)
            row_valid[mask] |= valid
            band_details.append((f'{_fmt_hz(start_f)}–{_fmt_hz(end_f)}', side,
                                 floor_db, min_amp_db, min_ratio_db, min_diff_db,
                                 int(valid.sum()), int(mask.sum())))
    return row_valid, band_details


# ========== 梳状干涉判别 (电源/时钟谐波) ==========

def _comb_fundamental(offsets_abs, tol_hz):
    """在峰偏移绝对值中搜索等间隔基频 f0: 使最多峰落在 n*f0 ± tol 的候选。

    峰偏移先按 tol 量化为整数 bin; 候选 f0 由峰间距及其 1/2..1/4 生成
    (谐波只出现偶次阶时检出的 f0 可能是真实基频的整数倍, 不影响梳状判定)。
    返回 (f0_hz, 对齐峰数); 对齐峰数不足 COMB_MIN_TEETH 时返回 (None, 0)。
    """
    offs = np.asarray(sorted({round(float(o) / tol_hz) for o in offsets_abs}),
                      dtype=np.int64)
    if offs.size < cfg.COMB_MIN_TEETH:
        return None, 0
    best_f0, best_n = None, 0
    for i in range(offs.size - 1):
        for j in range(i + 1, offs.size):
            d = int(offs[j] - offs[i])
            for k in (1, 2, 3, 4):
                f0b = max(1, round(d / k))
                if f0b < 2:          # 基频低于 2 个容差 bin 无梳状意义
                    continue
                n_hit = 0
                for b in offs:
                    n = max(1, round(int(b) / f0b))
                    if abs(int(b) - n * f0b) <= 1:
                        n_hit += 1
                if n_hit > best_n:
                    best_n, best_f0 = n_hit, f0b * tol_hz
    if best_n < cfg.COMB_MIN_TEETH:
        return None, 0
    return best_f0, best_n


def detect_comb_interference(spectrogram_db, freq_axis, center_freq, env_db, peaks):
    """判别检测峰是否构成梳状干涉 (电源/时钟谐波), 成立则从振动分析剔除。

    判据 (须同时满足):
      1. 等间隔: 存在基频 f0 使 >= COMB_MIN_TEETH 个峰位于 n*f0 ± COMB_ALIGN_TOL_BINS;
      2. 全时段持续: 各齿时间占用率中值 >= COMB_DUTY_MIN
         (时间块能量 > 底噪+有效阈值 的比例; 干涉持续存在, 振动为间歇性)。
    返回 (comb_mask, info): comb_mask 与 freq_axis 等长 (True = 干涉齿);
    info 为 None (非梳状, 全部保留) 或 dict(f0, n_teeth, duty)。"""
    zeros = np.zeros(freq_axis.shape, dtype=bool)
    if not cfg.COMB_DETECT_ENABLE or len(peaks) < cfg.COMB_MIN_TEETH:
        return zeros, None
    tol_hz = cfg.COMB_ALIGN_TOL_BINS * float(freq_axis[1] - freq_axis[0])
    f0, n_teeth = _comb_fundamental([abs(p[1]) for p in peaks], tol_hz)
    if f0 is None:
        return zeros, None
    # 持续性检验: 各齿时间占用率 (干涉 = 全时段持续; 振动 = 间歇)
    floor_db = float(np.median(env_db))
    thr_db = max(max(a, r, d) for _, _, a, r, d in cfg.DETECT_PARAMETERS)
    shift = np.abs(freq_axis - center_freq)
    n_max = int(shift.max() // f0)
    duties = []
    for n in range(1, n_max + 1):
        i = int(np.argmin(np.abs(shift - n * f0)))
        if abs(float(shift[i]) - n * f0) <= tol_hz:
            duties.append(float(np.mean(spectrogram_db[:, i] > floor_db + thr_db)))
    if not duties or float(np.median(duties)) < cfg.COMB_DUTY_MIN:
        return zeros, None
    comb_mask = np.zeros(freq_axis.shape, dtype=bool)
    for n in range(1, n_max + 1):
        comb_mask |= np.abs(shift - n * f0) <= tol_hz
    return comb_mask, {'f0': float(f0), 'n_teeth': int(n_teeth),
                       'duty': float(np.median(duties))}


def plot_combined(voltage, fs, spectrogram_data, center_freq, span,
                  title=None, save_path=None, n_plot=cfg.N_PLOT,
                  vmin=None, vmax=None, cmap=cfg.DEFAULT_CMAP, t0=0.0):
    """
    绘制振动分析图 (Vibration.jpg), 4 行紧凑型布局:
      第一行: 时间-电压波形 (抽样显示, 仅有效数据 voltage>0, 横轴为绝对时间)
      第二行: 时间-频率瀑布图 (各频点减时间中值 → 相对时变基线偏差, 亮色 = 能量异常升高)
      第三行: 有效频点能量-时间曲线 (每条曲线对应一个检测频点, 定位异常振动时段)
      第四行: 频率-能量频谱 (全时段包络, 标注有效频率值)

    有效频点判定: 频点全时段包络高出频带底噪的余量(dB) 同时满足
    检测最小幅度(dB) / 峰-底噪最小比值(dB) / 峰-底噪最小差值(dB) (底噪 = 频带内包络中值)。
    梳状干涉 (电源/时钟谐波, 等间隔且全时段持续) 自动从有效频点与谱峰标注中剔除。
    t0: 分析窗口起始绝对时间 (s), 各时间横轴 = t0 + 窗口内相对时间。
    """
    spectrogram = spectrogram_data['spectrogram']
    time_axis = spectrogram_data['time_axis']
    freq_axis = spectrogram_data['freq_axis']

    # 转 dB
    spectrogram_db = 10 * np.log10(spectrogram + np.float32(cfg.SPECTROGRAM_DB_FLOOR),
                                   dtype=np.float32)

    env_psd = spectrogram.max(axis=0)          # 各频点全时段包络 (PSD)
    env_db = spectrogram_db.max(axis=0)        # 各频点全时段包络 (dB)
    env_v = spectrogram_data['amp_scale'] * np.sqrt(env_psd)   # 包络电压幅值 (V)
    # 有效频率行: 包络高出底噪的余量同时满足检测参数三项阈值
    row_valid, _ = detect_valid_frequencies(env_db, freq_axis, center_freq,
                                            cfg.DETECT_PARAMETERS)
    # 谱峰检测 (第四行标注 + 梳状判别共用, 避免重复计算)
    peaks = detect_spectrum_peaks(env_db, env_v, freq_axis, center_freq,
                                  cfg.DETECT_PARAMETERS)
    # 梳状干涉判别: 等间隔 + 全时段持续 → 判为电源/时钟谐波, 从振动分析剔除
    comb_mask, comb_info = detect_comb_interference(
        spectrogram_db, freq_axis, center_freq, env_db, peaks)
    if comb_info is not None:
        row_valid &= ~comb_mask
        keep = []
        for p in peaks:
            k = int(np.argmin(np.abs(freq_axis - p[0])))
            if not comb_mask[k]:
                keep.append(p)
        peaks = keep
        print(f"  梳状干涉: 基频 {_fmt_hz(comb_info['f0'])}, {comb_info['n_teeth']} 齿, "
              f"时间占用率 {100 * comb_info['duty']:.0f}% → 判为电源/时钟谐波, "
              f"剔除 {int(comb_mask.sum()):,} 频点")
    else:
        print("  梳状干涉: 未检出, 全部有效频率保留")
    has_valid = bool(row_valid.any())

    # 瀑布图: 各频点减去自身时间中值 → 相对时变基线的偏差量
    # (持续存在的谱线/干涉背景平坦化为 ≈0 dB, 间歇或突发的振动能量显示为亮色)
    spec_dev = spectrogram_db - np.median(spectrogram_db, axis=0, keepdims=True)
    row_show = np.ones(row_valid.shape, dtype=bool)
    show_span = span
    # 可选: 瀑布图纵轴只显示 ±WATERFALL_SPAN_HZ 范围 (检测仍用完整频带)
    if cfg.WATERFALL_SPAN_HZ is not None:
        crop = np.abs(freq_axis - center_freq) <= cfg.WATERFALL_SPAN_HZ
        if crop.any():
            row_show = crop
            show_span = cfg.WATERFALL_SPAN_HZ
        else:
            print("  瀑布图显示范围超出频带, 保持完整频带显示")
    spectrogram_masked = spec_dev[:, row_show]

    # 颜色范围: 按瀑布图偏差分位数取 [背景, 异常], 避免全局 autoscale 压扁动态范围
    if vmin is None or vmax is None:
        lo, hi = np.percentile(spectrogram_masked,
                               [cfg.SPECTROGRAM_VMIN_PCT, cfg.SPECTROGRAM_VMAX_PCT])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig = plt.figure(figsize=cfg.FIGURE_SIZE)
    # 第 2 行右侧留窄条给瀑布图 colorbar, 各行绘图区同宽
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.6, 1.0, 1.4], width_ratios=[1, 0.03])

    if title:
        fig.suptitle(title, fontsize=12)

    # ---- 第一行: 时间-电压波形 (抽样显示, 仅有效数据 voltage>0) ----
    ax_wave = fig.add_subplot(gs[0, :])
    t_axis = t0 + np.arange(0, len(voltage), n_plot, dtype=np.float64) / fs
    v_plot = np.where(voltage > 0, voltage, np.nan)   # 无效数据 (voltage<=0) 不显示
    ax_wave.plot(t_axis, v_plot[::n_plot], linewidth=cfg.LINE_WIDTH)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Voltage (V)")
    ax_wave.set_title(f"SampleRate {fs/1e6:.0f} MHz, {len(voltage):,} pts, "
                      f"分析窗口 [{t0:.3f}, {t0 + len(voltage)/fs:.3f}] s")
    ax_wave.grid(True, alpha=cfg.GRID_ALPHA)

    # ---- 第二行: 时间-频率瀑布图 (各频点减时间中值 → 相对时变基线偏差, 亮色 = 能量异常升高) ----
    ax_tf = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    freq_show = freq_axis[row_show]
    spec_show = spectrogram_masked

    # 频点数过多时按行分组取最大值聚合, 限制渲染规模
    if freq_show.size > cfg.SPECTROGRAM_MAX_DISPLAY_ROWS:
        group = int(np.ceil(freq_show.size / cfg.SPECTROGRAM_MAX_DISPLAY_ROWS))
        starts = np.arange(0, freq_show.size, group)
        spec_show = np.fmax.reduceat(spec_show, starts, axis=1)
        freq_show = freq_show[starts]

    # 纵坐标用相对拍频的频移; 横轴为绝对时间
    shift_show = freq_show - center_freq
    # rasterized=True: 瀑布图单元数巨大, 直接栅格化渲染 (输出本就是位图)
    mesh = ax_tf.pcolormesh(t0 + time_axis, shift_show, spec_show.T,
                            shading='nearest', cmap=cmap,
                            vmin=vmin, vmax=vmax, rasterized=True)
    _setup_freq_axis_scale(ax_tf, freq_axis, center_freq)
    ax_tf.set_xlabel('Time (s)')
    ax_tf.set_ylabel('频移 (Hz)')
    ax_tf.set_title(f'时频瀑布图 (偏差量, 纵坐标: 相对拍频频移 ±{show_span/1000:.0f} kHz, '
                    f'亮色 = 能量相对自身时间基线异常升高)')
    ax_tf.grid(True, alpha=0.2, linestyle='--')
    ax_tf.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # 检测频带分界线 (上下边带对称, 无文字标注保持紧凑)
    for fb0, fb1, *_ in cfg.DETECT_PARAMETERS:
        for edge in (fb0, fb1):
            for sign in (1, -1):
                s = sign * edge
                if freq_axis[0] - center_freq <= s <= freq_axis[-1] - center_freq:
                    ax_tf.axhline(y=s, color='white', linestyle=':', linewidth=0.8, alpha=0.5)

    cbar = fig.colorbar(mesh, cax=cax, label='能量变化 (dB)')

    # ---- 第三行: 有效频点能量-时间曲线 (定位异常振动出现的时刻) ----
    ax_tr = fig.add_subplot(gs[2, :])
    if has_valid:
        # 频带底噪 (与检测一致: 频带内包络中值), 曲线归一化为 峰-底噪(dB) 便于同图比较
        floors = {}
        thr_all = 0.0
        for bi, (f0, f1, min_amp_db, min_ratio_db, min_diff_db) in enumerate(cfg.DETECT_PARAMETERS):
            thr_all = max(thr_all, min_amp_db, min_ratio_db, min_diff_db)
            for sgn in (1, -1):
                m = (freq_axis >= center_freq + sgn * min(f0, f1)) & \
                    (freq_axis <= center_freq + sgn * max(f0, f1))
                if m.sum() >= 3:
                    floors[(sgn, bi)] = float(np.median(env_db[m]))
        idx_valid = np.flatnonzero(row_valid)
        if idx_valid.size > 30:   # 曲线过多时按包络峰值保留前 30 条
            idx_valid = idx_valid[np.argsort(env_db[idx_valid])[-30:]]
            idx_valid.sort()
        for k in idx_valid:
            off = float(freq_axis[k]) - center_freq
            sgn = 1 if off >= 0 else -1
            fl = None
            for bi, (f0, f1, *_p) in enumerate(cfg.DETECT_PARAMETERS):
                if min(f0, f1) <= abs(off) <= max(f0, f1):
                    fl = floors.get((sgn, bi))
                    if fl is not None:
                        break
            if fl is None:
                fl = float(np.median(env_db))
            ax_tr.plot(t0 + time_axis, spectrogram_db[:, k] - fl, linewidth=0.5, alpha=0.75)
        ax_tr.axhline(y=thr_all, color='red', linestyle=':', linewidth=0.8,
                      label=f'阈值 {thr_all:g} dB')
        ax_tr.legend(fontsize=6, loc='upper right')
    else:
        ax_tr.text(0.5, 0.5, '无有效频点', transform=ax_tr.transAxes,
                   ha='center', va='center', color='gray')
    ax_tr.set_xlim(t0, t0 + len(voltage) / fs)
    ax_tr.set_xlabel('Time (s)')
    ax_tr.set_ylabel('峰-底噪 (dB)')
    ax_tr.set_title('有效频点能量-时间曲线 (超过红色阈值线 = 异常振动时段)')
    ax_tr.grid(True, alpha=cfg.GRID_ALPHA)

    # ---- 第四行: 频率-能量频谱 (默认蓝色, 标注有效频率值) ----
    ax_spec = fig.add_subplot(gs[3, :])
    ax_spec.plot(freq_axis, env_db, color='tab:blue', linewidth=cfg.LINE_WIDTH)

    # 标注有效频率值 (相对拍频偏移; peaks 已在开头检测并剔除梳状干涉)
    for f, offset, db_v, v_v in peaks:
        ax_spec.plot(f, db_v, 'o', color=cfg.SPECTRUM_PEAK_COLOR, markersize=4, zorder=5)
        sign = '+' if offset >= 0 else '-'
        # 标注有效频率值 (相对拍频偏移)
        ax_spec.annotate(f'{sign}{_fmt_hz(abs(offset))}',
                         xy=(f, db_v), xytext=(0, 5),
                         textcoords='offset points', ha='center',
                         fontsize=cfg.SPECTRUM_ANNOTATION_FONT_SIZE,
                         color=cfg.SPECTRUM_PEAK_COLOR)

    ax_spec.axvline(x=center_freq, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('PSD Envelope (dB)')
    ax_spec.set_title(f'频谱 (检测峰值 {len(peaks)} 个)')
    ax_spec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_freq(v)))
    ax_spec.grid(True, alpha=cfg.GRID_ALPHA)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig_path = f'{save_path}{cfg.SAVE_SUFFIX}'
        plt.savefig(fig_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f'振动分析图已保存: {fig_path}')

    return fig


# ========== 统计信息 ==========

def collect_statistics(bin_file, result, fs, spectrogram_data):
    """汇总统计信息，返回文本行列表"""
    lines = []
    sep = "=" * 60
    lines += [sep, "数据统计信息", sep]
    lines.append(f"文件: {bin_file}")
    lines.append(f"采样点数: {result['num_samples']:,}")
    lines.append(f"采样率: {fs:,.0f} Hz ({fs/1e6:g} MHz)")
    lines.append(f"通道数: {result.get('channel_count', 1)}")
    lines.append(f"分析窗口: 起始 {result.get('window_start_s', 0.0):.3f} s, "
                 f"时长 {result['num_samples'] / fs:.3f} s")

    v = result['voltage_data']
    v_valid = v[v > 0]   # 有效数据: voltage > 0
    if v_valid.size:
        mean_v, std_v = _mean_std(v_valid)
        lines.append(f"电压(有效 voltage>0): 均值={mean_v:.4f}V, 标准差={std_v:.4f}V, "
                     f"范围=[{v_valid.min():.4f}, {v_valid.max():.4f}]V, "
                     f"有效采样 {v_valid.size:,}/{v.size:,} ({100 * v_valid.size / v.size:.1f}%)")
    else:
        lines.append("电压(有效 voltage>0): 无有效采样")

    lines += ["", sep, "频谱分析统计", sep]
    lines.append(f"FFT 窗口点数: 2^{cfg.FFT_ORDER} = {cfg.N_FFT:,} 点")
    lines.append(f"时间窗口: {spectrogram_data['n_fft'] / fs * 1000:.2f} ms "
                 f"(= {cfg.N_FFT:,} 点 / {fs/1e6:g} MHz)")
    lines.append(f"频率分辨率: {spectrogram_data['freq_resolution']:.1f} Hz")
    lines.append(f"目标拍频: {_fmt_freq(cfg.TARGET_FREQ)}")
    lines.append(f"频移范围: [{_fmt_freq(cfg.TARGET_FREQ - cfg.FREQ_SPAN)}, "
                 f"{_fmt_freq(cfg.TARGET_FREQ + cfg.FREQ_SPAN)}] (±{cfg.FREQ_SPAN/1000:.1f} kHz)")
    lines.append(f"时间块数: {spectrogram_data['n_blocks']:,}")
    lines.append(f"分析频点数: {spectrogram_data['n_freqs']:,}")

    # 检测参数
    lines += ["", "检测参数 (振动频率 = |谱线频率 - 目标拍频|, 上下边带对称检测)", "-" * 60]
    for f0, f1, min_amp_db, min_ratio_db, min_diff_db in cfg.DETECT_PARAMETERS:
        lines.append(f"  [{_fmt_hz(f0)} – {_fmt_hz(f1)}], 检测最小幅度 {min_amp_db:g} dB, "
                     f"峰-底噪最小比值 {min_ratio_db:g} dB, 峰-底噪最小差值 {min_diff_db:g} dB")

    # 有效频率判定 (峰-底噪余量: 最小幅度/比值/差值 均达标)
    spec = spectrogram_data['spectrogram']
    spec_db = 10 * np.log10(spec + cfg.SPECTROGRAM_DB_FLOOR)
    env_db = spec_db.max(axis=0)
    row_valid, band_details = detect_valid_frequencies(
        env_db, spectrogram_data['freq_axis'], cfg.TARGET_FREQ, cfg.DETECT_PARAMETERS)
    lines += ["", "有效频率判定 (包络高出底噪余量同时满足 最小幅度/比值/差值)", "-" * 60]
    for band, side, floor_db, min_amp_db, min_ratio_db, min_diff_db, n_v, n_t in band_details:
        lines.append(f"  [{band}] {side}: 底噪 {floor_db:.1f} dB, 最小幅度 {min_amp_db:g} dB, "
                     f"最小比值 {min_ratio_db:g} dB, 最小差值 {min_diff_db:g} dB, "
                     f"有效 {n_v}/{n_t} 频点")
    # 梳状干涉判别 (与绘图一致; verbose=False 避免重复打印谱峰检测过程)
    env_v_st = spectrogram_data['amp_scale'] * np.sqrt(spec.max(axis=0))
    peaks_st = detect_spectrum_peaks(env_db, env_v_st, spectrogram_data['freq_axis'],
                                     cfg.TARGET_FREQ, cfg.DETECT_PARAMETERS, verbose=False)
    comb_mask_st, comb_info_st = detect_comb_interference(
        spec_db, spectrogram_data['freq_axis'], cfg.TARGET_FREQ, env_db, peaks_st)
    n_excl = int((row_valid & comb_mask_st).sum()) if comb_info_st is not None else 0
    if comb_info_st is not None:
        row_valid &= ~comb_mask_st
    n_valid = int(row_valid.sum())
    lines.append(f"有效频率合计: {n_valid:,} / {row_valid.size:,} 频点 "
                 f"({100 * n_valid / row_valid.size:.1f}%)")

    lines += ["", "梳状干涉判别 (等间隔 + 全时段持续 → 判为电源/时钟谐波)", "-" * 60]
    if comb_info_st is not None:
        lines.append(f"  基频 {_fmt_hz(comb_info_st['f0'])}, {comb_info_st['n_teeth']} 齿, "
                     f"时间占用率 {100 * comb_info_st['duty']:.0f}%")
        lines.append(f"  判定: 电源/时钟谐波, 从振动分析剔除 {n_excl:,} 个有效频点")
    else:
        lines.append("  未检出梳状干涉, 全部有效频率保留")

    lines.append(sep)
    return lines


# ========== 主程序 ==========

def run_analysis(bin_file):
    """完整分析流程: 读取数据 → 导出 CSV → FFT → 绘图 → 统计信息"""
    print(f"处理文件: {os.path.basename(bin_file)}")
    print("-" * 60)
    save_path = os.path.join(os.path.dirname(bin_file), os.path.splitext(os.path.basename(bin_file))[0])
    timestr = time.strftime("%Y%m%d%H%M%S")
    save_path_suffix = f'{save_path}_{timestr}_'

    # 1. 读取数据
    result = load_data(bin_file)
    voltage = result['voltage_data']
    fs = result['sampling_rate']
    channels = result.get('channels') or [f'CH{i+1}' for i in range(voltage.shape[1] if voltage.ndim > 1 else 1)]

    # 2. 解交错并选择分析通道（默认第一个通道）
    channel_count = result.get('channel_count', 1)
    if voltage.ndim == 1 and channel_count > 1:
        data = voltage.reshape(-1, channel_count)[:, 0]
    else:
        data = voltage if voltage.ndim == 1 else voltage[:, 0]
    print(f"\n[成功] 读取完成: 采样点数={result['num_samples']:,}, 采样率={fs/1e6:g} MHz, "
          f"通道数={len(channels)}")
    print(f"分析通道: {channels[0]}")

    # 3. 导出原始码值 CSV
    if cfg.EXPORT_CSV and 'raw_codes' in result:
        export_rawdata_csv(result, save_path_suffix.rstrip('_'))

    # 4. 时频分析
    spectrogram_data = calculate_spectrogram(
        data, fs=fs, n_fft=cfg.N_FFT,
        center_freq=cfg.TARGET_FREQ, span=cfg.FREQ_SPAN,
        window=cfg.WINDOW_TYPE, step=cfg.FFT_STEP)

    # 5. 绘制分析图
    plot_combined(data, fs, spectrogram_data, cfg.TARGET_FREQ, cfg.FREQ_SPAN,
                  title=os.path.basename(bin_file), save_path=save_path_suffix,
                  t0=result.get('window_start_s', 0.0))

    # 6. 统计信息
    stats_lines = collect_statistics(bin_file, result, fs, spectrogram_data)
    print("\n".join(stats_lines))

    print(f"\n处理完成！结果已保存到: {save_path}")


if __name__ == "__main__":
    try:
        run_analysis(cfg.DEFAULT_BIN_FILE)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
