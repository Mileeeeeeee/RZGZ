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

def cfg():
    """
    ts_signal_fft 配置文件
    =====================
    所有可调参数集中管理，修改参数无需改动主逻辑代码。
    """

    # =============================================================================
    # 默认文件路径
    # =============================================================================

    DEFAULT_BIN_FILE = r"E:\system\data\1550_40M\ACTS1000_data_100Msps_1s_10kS_20260821_20260821135631696_0.bin"

    # =============================================================================
    # 分析时间段
    # =============================================================================

    # 只分析文件内 [ANALYSIS_START_S, ANALYSIS_START_S + ANALYSIS_DURATION_S) 时间段的数据
    # (文件内绝对时间, 秒)。读取阶段即裁剪: 未选中的数据不做码值转换、不占内存。
    # ANALYSIS_DURATION_S = None 表示分析到文件末尾。
    ANALYSIS_START_S = 0.0  # 分析窗口起始时间 (s, 文件内绝对时间)
    ANALYSIS_DURATION_S = 1  # 分析窗口长度 (s); None = 分析到文件末尾

    # =============================================================================
    # 数据转换 - ADC 码值转电压参数
    # =============================================================================

    ADC_RESOLUTION = 4096  # ADC分辨率：2^12 = 4096 级量化
    CODE_OFFSET = ADC_RESOLUTION // 2  # ADC中点码值偏移
    VOLTAGE_SCALE = 10  # 满量程电压范围 (V)，对应 ±5V 量程

    # =============================================================================
    # 采样率
    # =============================================================================

    DEFAULT_SAMPLE_RATE = 100_000_000  # 采样率 (Hz)

    # =============================================================================
    # 时频分析参数
    # =============================================================================

    # FFT 时间块长度为 2 的幂次: n_fft = 2^FFT_ORDER
    # fs=100MHz,
    # FFT_ORDER=21 → 2^21=2,097,152 点/块 → 时间分辨率 ~21ms, 频率分辨率 ~48Hz
    # FFT_ORDER=20 → 2^20=1,048,576 点/块 → 时间分辨率 ~10ms, 频率分辨率 ~95Hz
    # FFT_ORDER=19 → 2^19=524,288 点/块 → 时间分辨率 ~5ms, 频率分辨率 ~191Hz
    FFT_ORDER = 21  # FFT 块长度指数 (n_fft = 2^FFT_ORDER)

    # 由 FFT_ORDER 派生:
    N_FFT = 1 << FFT_ORDER  # FFT 块长度 (采样点数)
    FFT_STEP = int((N_FFT * 0.80) // 100000 * 100000)  # 当前块长度，剩余长度取前一块
    TIME_BLOCK_DURATION = N_FFT / DEFAULT_SAMPLE_RATE  # 时间块长度 (s)

    TARGET_FREQ = 40_000_000  # 目标拍频 (Hz): 本振光与信号光的频差，即频谱分析中心频率
    FREQ_SPAN = 1_000_000  # 频移范围 (Hz): 分析范围为 [TARGET_FREQ - FREQ_SPAN, TARGET_FREQ + FREQ_SPAN]

    # OVERLAP_RATIO = 0.9  # 相邻时间块之间的重叠比例 (0~1)
    WINDOW_TYPE = 'hann'  # FFT 窗函数类型: 'hann', 'hamming', 'blackman', 'rectangular'
    DETREND_METHOD = 'constant'  # FFT 前去趋势: 'constant' (减均值), 'linear', None
    SPECTROGRAM_DB_FLOOR = 1e-12  # dB 转换下限保护值，避免 log10(0)

    # 时频瀑布图能量阈值: 低于 (峰值 - 该值 dB) 的频谱单元渲染为白色/透明
    SPECTROGRAM_ENERGY_THRESHOLD_DB = 30  # 能量阈值 (dB): 峰值以下该 dB 数以内才渲染

    # 瀑布图渲染的最大频率行数: 有效频率行数超出时按行分组取最大值聚合 (限制渲染规模)
    SPECTROGRAM_MAX_DISPLAY_ROWS = 4000

    # =============================================================================
    # 频段划分 (Hz): 以目标拍频 (40 MHz) 为中心, 由窄到宽逐级放大
    # 振动频率 = |谱线频率 - 目标拍频|
    #   低频段 ±1 kHz:   热变形引起的宏观位移、熔池固有振荡频率能量异常
    #   中频段 ±100 kHz: 特征频带能量突增、瞬态冲击
    #   高频段 ±1 MHz:   瞬时速度突变、宽带冲击
    # fmin=50 统一排除载波泄漏附近 (≈0–50 Hz) 的谱线
    # =============================================================================

    FREQ_BANDS = {
        'low': {'label': '低频段', 'fmin': 100, 'fmax': 1_000},
        'mid': {'label': '中频段', 'fmin': 1_000, 'fmax': 100_000},
        'high': {'label': '高频段', 'fmin': 100_000, 'fmax': 1_000_000},
    }

    # 低频漂移窄带 (Hz): 用于监测 50Hz 附近的缓慢漂移
    DRIFT_BAND_HZ = (45, 55)

    # =============================================================================
    # CSV 导出参数
    # =============================================================================

    EXPORT_CSV = False  # 是否将原始 ADC 码值 (uint16) 导出为 CSV 文件

    # =============================================================================
    # 分块处理参数 (内存控制)
    # =============================================================================

    MEAN_STD_CHUNK = 5_000_000  # 均值/标准差计算分块大小
    CHUNK_SAMPLES = 10_000_000  # bin 文件分块转换采样点数

    # =============================================================================
    # 绘图参数
    # =============================================================================

    N_PLOT = 100  # 时域波形图采样间隔：每 N_PLOT 个点绘制 1 个点
    FIGURE_SIZE = (24, 15)  # 振动分析图尺寸 (宽, 高) 英寸, 3行×3列布局
    FIGURE_DPI = 300  # 保存图片的 DPI
    DEFAULT_CMAP = 'jet'  # 时频瀑布图 colormap
    LINE_WIDTH = 0.5  # 曲线默认线宽
    GRID_ALPHA = 0.3  # 网格透明度

    SPECTRUM_ANNOTATION_FONT_SIZE = 6  # 频谱峰值标注字体大小
    SPECTRUM_PEAK_COLOR = 'red'  # 频段频谱图有效频点标注颜色
    SPECTRUM_PEAK_MAX_ANNOT = 8  # 每个频段最多标注的有效频点数 (按能量取前 N)
    STATS_SUFFIX = 'statistics.txt'  # 统计信息 txt 文件后缀

    SAVE_SUFFIX = 'Vibration.jpg'  # 保存图片文件名后缀


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
    # 以目标拍频为中心，±span 范围进行频谱分析（振动瞬时频移区域）
    freq_mask = (freq_axis >= center_freq - span) & (freq_axis <= center_freq + span)
    freq_selected = freq_axis[freq_mask]
    n_freqs = freq_selected.size
    if n_freqs == 0:
        raise ValueError(
            f"频率范围 [{center_freq - span:.0f}, {center_freq + span:.0f}] Hz 内无频率点，"
            f"请检查 TARGET_FREQ/FREQ_SPAN 与采样率 (fs={fs:.0f} Hz, Nyquist={fs/2:.0f} Hz)")

    print(f"\n频谱分析参数:")
    print(f"  - 时间块长度: {time_block_duration*1000:.2f} ms ({samples_per_block} 采样点)")
    print(f"  - 重叠率: {overlap*100:.0f}%")
    print(f"  - 时间块数: {n_blocks:,}")
    print(f"  - 目标拍频(中心频率): {_fmt_freq(center_freq)}")
    print(f"  - 频移范围: [{_fmt_freq(center_freq - span)}, {_fmt_freq(center_freq + span)}] (±{span/1000:.1f} kHz)")
    print(f"  - 频率点数: {n_freqs}")

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
    chunk_blocks = max(1, int(cfg.SPECTROGRAM_CHUNK_MB * 1e6 // (16 * samples_per_block)))
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
            psd = np.abs(rfft(block, axis=1))          # (chunk, N//2+1) float32
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
                  spectrum_peaks=None, sideband_peaks=None):
    """
    绘制合并图:
      第一行: 时间-电压波形 | 全时段频谱 (标注高于阈值的峰值点及能量值)
      第二行: 时频瀑布图

    center_freq: 目标拍频 (Hz)，本振光与信号光的频差，即频谱分析中心频率
                 （参考频谱仪显示）
    spectrum_peaks: extract_spectrum_peaks() 返回的字典，若提供则在全时段频谱图上
                    标注高于阈值的峰值点（频率 + 能量值）并绘制阈值参考线
    sideband_peaks: extract_sideband_peaks() 返回的字典，若提供则在全时段频谱图上
                    标注目标拍频两侧的频移带振动峰值（峰值频率 + 振动频率 + 能量值）
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

    # 全时段频谱（能量随时间积分）
    energy_spectrum = np.sum(spectrogram, axis=0)
    energy_spectrum_db = 10 * np.log10(energy_spectrum + cfg.SPECTROGRAM_DB_FLOOR)
    ax_spec.plot(freq_axis, energy_spectrum_db,
                 linewidth=cfg.LINE_WIDTH, color=cfg.SPECTRUM_LINE_COLOR)
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Integrated Energy (dB)')
    ax_spec.set_title(f'Full-Period Spectrum (Target: {_fmt_freq(center_freq)}, span ±{span/1000:.0f} kHz)')
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

    # 标注载波两侧的侧边带振动峰值（振动频率 = 峰值频率与载波频率的偏移量）
    if sideband_peaks:
        for name, band in sideband_peaks.items():
            if not band:
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
    提取每个时间块的最大频率点
    """
    spectrogram = spectrogram_data['spectrogram']
    freq_axis = spectrogram_data['freq_axis']

    peak_indices = np.argmax(spectrogram, axis=1)
    peak_freqs = freq_axis[peak_indices]
    peak_amps = np.max(spectrogram, axis=1)

    return peak_freqs, peak_amps


def extract_spectrum_peaks(spectrogram_data, threshold_db=cfg.SPECTRUM_THRESHOLD_DB):
    """
    在全时段能量谱上提取高于阈值的峰值点（频率 + 能量值）

    能量谱 = 各时间块 PSD 沿时间求和（全时段能量积分）
    阈值   = 能量谱峰值 - threshold_db (dB)

    返回:
        dict: peak_freqs(峰值频率), peak_energy_db(峰值能量值, dB),
              threshold_db(绝对阈值, dB), max_db(能量谱峰值, dB),
              n_above(高于阈值的频率点总数)
    """
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


def extract_sideband_peaks(spectrogram_data, center_freq,
                           guard_hz=cfg.CARRIER_GUARD_HZ,
                           threshold_db=cfg.SPECTRUM_THRESHOLD_DB):
    """
    在目标拍频两侧的频移带中提取振动信号峰值

    目标拍频 = center_freq（本振光与信号光的频差），振动信号以瞬时频移形式分布在
    [center_freq - span, center_freq + span]。排除目标拍频附近 ±guard_hz 的频率点后，
    分别在下频移带(低于目标拍频)和上频移带(高于目标拍频)的全时段能量谱上提取峰值；
    峰值频率与目标拍频的偏移量即为振动频率。

    返回:
        dict: {'lower': {...}, 'upper': {...}}
        每个频移带包含: peak_freqs(峰值频率, 按能量降序), peak_db(峰值能量, dB),
                       vib_freqs(振动频率 = |峰值频率 - 目标拍频|),
                       n_above(高于阈值的频率点数)；无有效数据时为 None
    """
    freq_axis = spectrogram_data['freq_axis']
    energy = np.sum(spectrogram_data['spectrogram'], axis=0)
    energy_db = 10 * np.log10(energy + cfg.SPECTROGRAM_DB_FLOOR)

    bands = {
        'lower': freq_axis < center_freq - guard_hz,   # 下频移带
        'upper': freq_axis > center_freq + guard_hz,   # 上频移带
    }

    result = {}
    for name, mask in bands.items():
        sub_db = energy_db[mask]
        if sub_db.size == 0:
            result[name] = None
            continue
        max_db = float(sub_db.max())
        peaks, _ = signal.find_peaks(sub_db, height=max_db - threshold_db)
        if peaks.size == 0:
            result[name] = None
            continue
        order = np.argsort(sub_db[peaks])[::-1]   # 按能量从高到低排序
        result[name] = {
            'peak_freqs': freq_axis[mask][peaks][order],
            'peak_db': sub_db[peaks][order],
            'vib_freqs': np.abs(freq_axis[mask][peaks][order] - center_freq),
            'n_above': int(np.sum(sub_db >= max_db - threshold_db)),
        }
    return result


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
                       sideband_peaks):
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
                 f"频率点数: {spectrogram_data['n_freqs']:,}")

    peak_freqs_block, _ = extract_peak_frequency(spectrogram_data)
    lines.append(f"逐时间块峰值频率: 平均={_fmt_freq(peak_freqs_block.mean())}, "
                 f"标准差={peak_freqs_block.std()/1000:.3f} kHz, "
                 f"范围=[{_fmt_freq(peak_freqs_block.min())}, {_fmt_freq(peak_freqs_block.max())}]")

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
    lines.append(f"目标拍频: {_fmt_freq(center_freq)}, "
                 f"频移跨度: ±{span/1000:.1f} kHz, "
                 f"拍频排除带宽: ±{cfg.CARRIER_GUARD_HZ/1000:.2f} kHz")
    for name, label in (('lower', '下频移带'), ('upper', '上频移带')):
        band = sideband_peaks.get(name)
        if not band:
            lines.append(f"{label}: 未检测到峰值")
            continue
        f0, p0, v0 = band['peak_freqs'][0], band['peak_db'][0], band['vib_freqs'][0]
        lines.append(f"{label}: 峰值 {_fmt_freq(f0)} ({p0:.2f} dB), "
                     f"振动频率 {v0/1000:.3f} kHz, 高于阈值的点数 {band['n_above']}")
        for pf, pd, pv in zip(band['peak_freqs'][1:6], band['peak_db'][1:6], band['vib_freqs'][1:6]):
            lines.append(f"    - {_fmt_freq(pf)} ({pd:.2f} dB), 振动频率 {pv/1000:.3f} kHz")
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

    # 4. 提取高于阈值的频谱峰值点（频率 + 能量值）
    spectrum_peaks = extract_spectrum_peaks(spectrogram_data,
                                            threshold_db=cfg.SPECTRUM_THRESHOLD_DB)

    # 5. 提取目标拍频两侧频移带中的振动峰值（振动频率 = |峰值频率 - 目标拍频|）
    sideband_peaks = extract_sideband_peaks(spectrogram_data, cfg.TARGET_FREQ,
                                            guard_hz=cfg.CARRIER_GUARD_HZ,
                                            threshold_db=cfg.SPECTRUM_THRESHOLD_DB)

    # 6. 绘制合并图（波形 + 全时段频谱(含阈值峰值标注) + 时频瀑布图）
    plot_combined(data, fs, spectrogram_data, cfg.TARGET_FREQ, cfg.FREQ_SPAN,
                  spectrum_peaks=spectrum_peaks, sideband_peaks=sideband_peaks,
                  title=os.path.basename(bin_file), save_path=save_path_suffix)

    # 7. 汇总统计信息并保存到 txt
    save_statistics_txt(save_path_suffix,
                        collect_statistics(bin_file, result, voltage, channels, fs,
                                           cfg.TARGET_FREQ, cfg.FREQ_SPAN,
                                           spectrogram_data, spectrum_peaks,
                                           sideband_peaks))

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