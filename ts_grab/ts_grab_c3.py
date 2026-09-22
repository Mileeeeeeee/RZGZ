""" 相干信号处理 - 激光焊接振动分析
输入：bin格式数据 int16，多通道交错存储 (解析按 PCI_CHANNEL, 文件名 _cN_ 自动覆盖; 分析只取前 channel_num 通道)
输出：
  阶段一: 三通道时域统计分析 (均值/RMS/峰峰值/标准差) statistics.txt
  每通道统计图 JPG (6层: 抽样数据/滑窗均值/滑窗标准差/滑窗RMS/滑窗峰峰值/有效频点能量-时间曲线)
  后续阶段: 所有通道合并波形JPG + 每通道CSV(时间,电压) + 滑动窗口特征CSV
参数：集中在文件内 params 类 (base/pcie_1840l/params_dc 合并)；文件命名参考 ts_signal_fft.py
作者：LYB
日期：2026-09-17
"""
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, periodogram

# 中文标注字体支持 (Windows 常见中文字体，缺失则回退默认字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
            DEFAULT_BIN_FILE = r"E:\system\data\1064_PV\2026091501\daqyh_sn000000000_5000000_c4_10000_10000_10000_10000_int16_20260915_155731_633.bin.part0",
            # 分析时间段: 只分析 [ANALYSIS_START_S, ANALYSIS_START_S + DURATION_S) 内数据, None = 到文件末尾
            ANALYSIS_START_S = 2.12,
            ANALYSIS_DURATION_S = 7.48,
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
            PCI_CHANNEL = 4,
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
                [1_000, 1_000_000, 1, 1.2, 0.8],
            ],
            PEAK_DISTANCE_BINS = 3,   # 找峰最小间隔 (单位: 频率分辨率 bin)
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
            DC_WINDOW_TIME=0.05,     # 每帧处理数据长度 (s)
            channel_num = 3,  # 数据分析通道数量: 只分析前 N 个通道, 其余省略
            channel_names  = ['CH0 可见光 (Vis)', 'CH1 1064nm反射 (Ref)', 'CH2 红外 (IR)'],

            # 特征提取参数
            feat_window_s = 0.02,  # 滑动窗口长度 (s)
            feat_step_s = 0.01,  # 滑动步进 (s)
            ir_lowpass_hz = 1000.0,  # 红外低通截止 (Hz)
            vis_bandpass_hz = (10.0, 500e3),  # 可见光带通 (Hz)
            ref_bandpass_hz = (10.0, 500e3),  # 1064nm带通 (Hz)
            ref_keyhole_band = (200.0, 800.0),  # 小孔振荡特征频段 (Hz)

            # 频率分析参数 (时频图: 窗口长度 = feat_window_s, 步进 = feat_step_s)
            freq_window_type = 'hann',       # 频谱窗函数
            freq_band_hz = (0, 2_500_000),    # 分析频带 (Hz), (0, fs/2) 为全频带
            freq_db_floor = 1e-12,           # dB 转换下限保护值, 避免 log10(0)
            freq_detect_thr_db = 20,        # 有效频点判定: 全时段包络高出频带底噪(中值)的最小余量 (dB)
            freq_detect_max_curves = 30,     # 有效频点能量-时间曲线最大条数 (按包络峰值取前 N)
            freq_max_display_rows = 2000,    # 频率行数上限: 超出时按行分组取最大值聚合
        )

def get_params_combine():
    """参数合并: 公用参数 + 采集卡参数 + 分析参数 (dict 依次合并, 后者覆盖前者)"""
    p = params()
    combine = {}
    for section in (p.base, p.pcie_1840l, p.params_dc):
        combine.update(section())
    combine['fs'] = combine['PCI_SAMPLE_RATE']   # 采样率别名 (Hz)
    # bin解析通道数自动识别: 文件名含 _cN_ (如 _c4_ = 4通道), 否则用 PCI_CHANNEL 默认值
    m = re.search(r'_c(\d+)_', os.path.basename(combine['DEFAULT_BIN_FILE']))
    if m:
        combine['PCI_CHANNEL'] = int(m.group(1))
    return SimpleNamespace(**combine)

cfg = get_params_combine()

def read_bin_file(bin_file, offset=0, count=-1):
    """用内存映射读取bin文件，避免一次性载入整个文件占用内存"""
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"文件不存在: {bin_file}")

    data = np.memmap(bin_file, dtype=np.int16, mode='r', offset=offset)
    if count >= 0:
        data = data[:count]
    print(data.shape, f"from {offset} to {offset + data.nbytes}")
    return data

def to_voltage(chunk):
    """int16原始码转电压，使用float32节省内存"""
    scalar = np.int16(-32768)
    data_v = (chunk ^ scalar).astype(np.float32) * np.float32(20.0 / 65536.0)
    return data_v

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
            f'分析时间段 [{cfg.ANALYSIS_START_S}, '
            f"{'文件末尾' if cfg.ANALYSIS_DURATION_S is None else cfg.ANALYSIS_START_S + cfg.ANALYSIS_DURATION_S}] "
            f'内无有效数据 (数据共 {total:,} 点, 时长 {total / fs:.3f} s)，'
            f'请检查 ANALYSIS_START_S / ANALYSIS_DURATION_S')
    return i0, i1

def _channel_stats(sig, chunk):
    """分块计算单通道时域统计量，避免 np.std/np.mean 产生 O(n) 临时数组导致内存不足。
    返回 (均值, 标准差, RMS, 最小值, 最大值)，单位 V。"""
    n = sig.size
    mean = float(np.mean(sig, dtype=np.float64))
    ss = 0.0   # 距平平方和 → 标准差
    sq = 0.0   # 平方和 → RMS
    vmin, vmax = np.inf, -np.inf
    for start in range(0, n, chunk):
        x = sig[start:start + chunk].astype(np.float64)
        d = x - mean
        ss += float(np.dot(d, d))
        sq += float(np.dot(x, x))
        vmin = min(vmin, float(x.min()))
        vmax = max(vmax, float(x.max()))
    return mean, (ss / n) ** 0.5, (sq / n) ** 0.5, vmin, vmax

def collect_time_stats(bin_file, data, fs, channel_num, channel_names):
    """阶段一: 三通道时域统计分析 (均值/RMS/峰峰值/标准差)，返回统计文本行列表。"""
    i0, i1 = _analysis_window_indices(fs, data.shape[0])
    sep = '=' * 60
    lines = [sep, '时域统计分析 (阶段一)', sep]
    lines.append(f'文件: {bin_file}')
    lines.append(f'采样率: {fs:,.0f} Hz ({fs / 1e6:g} MHz)')
    lines.append(f'通道数: {channel_num}')
    lines.append(f'数据总时长: {data.shape[0] / fs:.3f} s, 每通道 {data.shape[0]:,} 点')
    lines.append(f'分析窗口: [{i0 / fs:.3f} s, {i1 / fs:.3f} s), 时长 {(i1 - i0) / fs:.3f} s, 每通道 {i1 - i0:,} 点')
    lines.append('-' * 60)
    for ch in range(channel_num):
        name = channel_names[ch] if ch < len(channel_names) else f'CH{ch}'
        sig = to_voltage(data[i0:i1, ch])
        mean, std, rms, vmin, vmax = _channel_stats(sig, cfg.MEAN_STD_CHUNK)
        lines.append(f'{name}: 均值={mean:.6f} V, 标准差={std:.6f} V, RMS={rms:.6f} V, '
                     f'峰峰值={vmax - vmin:.6f} V, 范围=[{vmin:.6f}, {vmax:.6f}] V')
    return lines

def _sliding_stats(sig, win, step):
    """滑窗计算均值/标准差/RMS/峰峰值，返回 (t_center, mean, std, rms, p2p)。
    t_center 为各窗口中心索引 (采样点)，其余各量单位 V。"""
    n = sig.size
    n_w = (n - win) // step + 1
    t_center = np.zeros(n_w, dtype=np.float64)
    mean = np.zeros(n_w, dtype=np.float64)
    std = np.zeros(n_w, dtype=np.float64)
    rms = np.zeros(n_w, dtype=np.float64)
    p2p = np.zeros(n_w, dtype=np.float64)
    for i in range(n_w):
        s = i * step
        e = s + win
        w = sig[s:e].astype(np.float64)
        t_center[i] = (s + e) / 2.0
        mean[i] = w.mean()
        std[i] = w.std()
        rms[i] = np.sqrt(np.mean(w ** 2))
        p2p[i] = w.max() - w.min()
    return t_center, mean, std, rms, p2p

def _spectrogram(sig, fs, win, step):
    """按 feat_window_s 滑窗计算时频能量矩阵 (dB)。
    返回 (t_center, freq_show, spec_db):
      t_center [n_w]: 窗口中心时刻 (s, 相对信号起点)
      freq_show [n_rows]: 频率行 (Hz, 超出 freq_max_display_rows 时按最大值聚合)
      spec_db [n_w, n_rows]: 能量 (dB)"""
    n = sig.size
    n_w = (n - win) // step + 1
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band_mask = (freqs >= cfg.freq_band_hz[0]) & (freqs <= cfg.freq_band_hz[1])
    f_band = freqs[band_mask]
    # 频点数过多时按行分组取最大值聚合, 限制矩阵与渲染规模
    group = max(1, int(np.ceil(f_band.size / cfg.freq_max_display_rows)))
    starts = np.arange(0, f_band.size, group)
    ends = np.append(starts[1:], f_band.size)
    mid = starts + (ends - starts) // 2   # 行代表频率: 组中点
    spec = np.zeros((n_w, starts.size), dtype=np.float32)
    t_center = np.zeros(n_w, dtype=np.float64)
    for i in range(n_w):
        s = i * step
        e = s + win
        w = sig[s:e].astype(np.float64)
        _, psd = periodogram(w, fs, window=cfg.freq_window_type)
        spec[i] = np.fmax.reduceat(psd[band_mask], starts).astype(np.float32)
        t_center[i] = (s + e) / 2.0 / fs
    spec_db = 10.0 * np.log10(spec + cfg.freq_db_floor)
    return t_center, f_band[mid], spec_db

def _detect_valid_freqs(spec_db):
    """有效频点判定 (参考 ts_signal_fft.py 第三行):
    频点全时段包络高出频带底噪(中值)的余量 >= freq_detect_thr_db。
    返回 (有效频率行索引, 底噪 dB), 索引按包络峰值从高到低。"""
    env_db = spec_db.max(axis=0)
    floor = float(np.median(env_db))
    idx = np.flatnonzero(env_db - floor >= cfg.freq_detect_thr_db)
    return idx[np.argsort(env_db[idx])[::-1]], floor

def plot_channel_time_stats(data, out_prefix, fs, channel_num, channel_names):
    """每通道分别绘制统计图并保存
    (6层: 1层抽样数据, 2层滑窗均值, 3层滑窗标准差, 4层滑窗RMS, 5层滑窗峰峰值,
     6层有效频点能量-时间曲线, 参考 ts_signal_fft.py 第三行)。"""
    i0, i1 = _analysis_window_indices(fs, data.shape[0])
    win = min(int(cfg.feat_window_s * fs), i1 - i0)
    step = max(1, int(cfg.feat_step_s * fs))
    titles = ['抽样数据', '滑窗均值', '滑窗标准差', '滑窗RMS', '滑窗峰峰值']

    for ch in range(channel_num):
        name = channel_names[ch] if ch < len(channel_names) else f'CH{ch}'
        sig = to_voltage(data[i0:i1, ch])

        # 1层: 抽样数据 (每 N_PLOT 点取 1 点)
        idx = np.arange(0, sig.size, cfg.N_PLOT)
        t_samp = (i0 + idx) / fs

        # 2~5层: 按 feat_window_s / feat_step_s 滑窗统计
        t_c, mean, std, rms, p2p = _sliding_stats(sig, win, step)
        t_c = (i0 + t_c) / fs

        # 6层: 有效频点能量-时间曲线 (窗口长度 = feat_window_s)
        t_tf, freq_show, spec_db = _spectrogram(sig, fs, win, step)
        t_tf = (i0 / fs) + t_tf

        fig, axes = plt.subplots(6, 1,
                                 figsize=(cfg.FIGURE_SIZE[0], cfg.FIGURE_SIZE[1] * 2),
                                 sharex=True)
        for ax in axes:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))   # 时间轴网格 0.1 s 间隔
        series = [(t_samp, sig[idx]), (t_c, mean), (t_c, std), (t_c, rms), (t_c, p2p)]
        for ax, title, (tx, y) in zip(axes[:5], titles, series):
            ax.plot(tx, y, lw=cfg.LINE_WIDTH)
            ax.set_title(title, fontsize=9, loc='left')
            ax.set_ylabel('V')
            ax.grid(True, alpha=cfg.GRID_ALPHA)
        if sig[idx].min() < 0:
            axes[0].set_ylim(bottom=0)   # 第1层抽样数据: 最小值 < 0 时下限取 0, 否则自动调整

        # 6层: 有效频点能量-时间曲线 (参考 ts_signal_fft.py 第三行)
        #   每条曲线对应一个有效频点, 纵轴 = 峰-底噪 (dB), 超过红色阈值 = 能量异常时段
        ax_tf = axes[5]
        idx_valid, floor = _detect_valid_freqs(spec_db)
        if idx_valid.size > cfg.freq_detect_max_curves:   # 曲线过多时按包络峰值保留前 N 条
            idx_valid = idx_valid[:cfg.freq_detect_max_curves]
        idx_valid = np.sort(idx_valid)                    # 按频率升序绘制
        if idx_valid.size:
            for k in idx_valid:
                ax_tf.plot(t_tf, spec_db[:, k] - floor, lw=cfg.LINE_WIDTH, alpha=0.75,
                           label=f'{freq_show[k] / 1000:g} kHz')
        else:
            ax_tf.text(0.5, 0.5, '无有效频点', transform=ax_tf.transAxes,
                       ha='center', va='center', color='gray')
        ax_tf.axhline(y=cfg.freq_detect_thr_db, color='red', linestyle=':',
                      linewidth=0.8, label=f'阈值 {cfg.freq_detect_thr_db:g} dB')
        ax_tf.legend(fontsize=6, loc='upper right', ncol=2)
        ax_tf.set_title('时频图 (有效频点能量-时间曲线, 超过红色阈值 = 能量异常)',
                        fontsize=9, loc='left')
        ax_tf.set_ylabel('峰-底噪 (dB)')
        ax_tf.grid(True, alpha=cfg.GRID_ALPHA)

        axes[0].set_xlim(i0 / fs, i1 / fs)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{name} 统计与时频 (窗口 {cfg.feat_window_s:g} s, 步进 {cfg.feat_step_s:g} s)',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig_path = f'{out_prefix}ch{ch}_time_stats.jpg'
        fig.savefig(fig_path, dpi=cfg.FIGURE_DPI)
        plt.close(fig)
        print(f'已保存 {name} 统计图: {fig_path}')

def butter_lowpass_filter(sig, cutoff, fs, order=4):
    """Butterworth 低通滤波，零相位"""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)

def butter_bandpass_filter(sig, low, high, fs, order=4):
    """Butterworth 带通滤波，零相位"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_features(data_v, cfg):
    """
    滑动窗口提取三通道物理特征。
    data_v: float32 电压矩阵，shape [N, 3]，列顺序 [IR, Vis, Ref]
    返回: (feats, t_center)
      feats shape [M, 10]: ir_mean, ir_std, ir_slope,
                            vis_rms, vis_kurtosis, vis_zcr,
                            ref_mean, ref_std, ref_loss_ratio, ref_psd_ratio
      t_center shape [M]: 每个窗口中心时刻 (s)
    """
    fs = cfg.fs
    win = int(cfg.feat_window_s * fs)
    step = int(cfg.feat_step_s * fs)
    N = data_v.shape[0]

    print('正在滤波...')
    ir_f  = butter_lowpass_filter(data_v[:, 0].astype(np.float64),
                                  cfg.ir_lowpass_hz, fs)
    vis_f = butter_bandpass_filter(data_v[:, 1].astype(np.float64),
                                   cfg.vis_bandpass_hz[0], cfg.vis_bandpass_hz[1], fs)
    ref_f = butter_bandpass_filter(data_v[:, 2].astype(np.float64),
                                   cfg.ref_bandpass_hz[0], cfg.ref_bandpass_hz[1], fs)

    n_windows = (N - win) // step + 1
    feats = np.zeros((n_windows, 10), dtype=np.float64)
    t_center = np.zeros(n_windows, dtype=np.float64)
    nperseg = min(65536, win)

    print(f'正在提取特征，共 {n_windows} 个窗口...')
    for i in range(n_windows):
        s = i * step
        e = s + win
        t_center[i] = (s + e) / 2.0 / fs

        # --- CH0 红外：均值、标准差、线性斜率 ---
        ir_w = ir_f[s:e]
        ir_mean = ir_w.mean()
        ir_std  = ir_w.std()
        t_win = np.arange(win) / fs
        ir_slope = np.polyfit(t_win, ir_w, 1)[0]  # V/s

        # --- CH1 可见光：RMS、峭度、过零率 ---
        vis_w = vis_f[s:e]
        vis_rms = np.sqrt(np.mean(vis_w ** 2))
        vis_std = vis_w.std()
        if vis_std > 0:
            vis_kurt = np.mean(((vis_w - vis_w.mean()) / vis_std) ** 4)
        else:
            vis_kurt = 0.0
        vis_zcr = np.sum(np.diff(np.sign(vis_w)) != 0) / win

        # --- CH2 1064nm 反射：均值、标准差、丢失率、小孔振荡频段能量比 ---
        ref_w = ref_f[s:e]
        ref_mean = ref_w.mean()
        ref_std  = ref_w.std()
        threshold = ref_mean - 3.0 * ref_std
        ref_loss  = np.sum(ref_w < threshold) / win

        freq_r, psd_r = periodogram(ref_w, fs, window='hann', nfft=nperseg)
        band_mask = (freq_r >= cfg.ref_keyhole_band[0]) & (freq_r <= cfg.ref_keyhole_band[1])
        psd_total = psd_r.sum()
        ref_psd_ratio = psd_r[band_mask].sum() / psd_total if psd_total > 0 else 0.0

        feats[i] = [ir_mean, ir_std, ir_slope,
                    vis_rms, vis_kurt, vis_zcr,
                    ref_mean, ref_std, ref_loss, ref_psd_ratio]

        if (i + 1) % 50 == 0 or i == n_windows - 1:
            print(f'  特征提取进度: {i+1}/{n_windows}')

    return feats, t_center

def export_channel_csvs(data, out_prefix, fs, channel_num, chunk_samples, decimation):
    """按块遍历数据，同时写出各通道CSV文件 (time_s, voltage_V)"""
    n_samples = data.shape[0]
    files = []
    for ch in range(channel_num):
        f = open(f'{out_prefix}ch{ch}.csv', 'w')
        f.write('time_s,voltage_V\n')
        files.append(f)
    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        block = to_voltage(data[start:end])
        block = block[::decimation]  # 抽稀
        t = (start + np.arange(0, end - start, decimation)) / fs
        for ch, f in enumerate(files):
            np.savetxt(f, np.column_stack((t, block[:, ch])), fmt='%.9f,%.6f')
        print(f'CSV导出进度: {end}/{n_samples}')
    for f in files:
        f.close()

def plot_channels(data, out_path, fs, max_points, channel_names):
    """所有通道波形画在同一张画布上(纵向排列、共用时间轴)，数据量大时均匀抽稀"""
    n_samples, n_ch = data.shape
    step = max(1, n_samples // max_points)
    idx = np.arange(0, n_samples, step)
    t = idx / fs
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for ch, ax in enumerate(axes):
        signal = to_voltage(data[idx, ch])
        ax.plot(t, signal, lw=0.5)
        ax.set_ylabel('Voltage (V)')
        title = channel_names[ch] if ch < len(channel_names) else f'Channel {ch}'
        ax.set_title(title, fontsize=9, loc='left')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'已保存波形图: {out_path}')

def main():
    # 参数与文件路径导入
    fileIn = cfg.DEFAULT_BIN_FILE
    filename = os.path.basename(fileIn)

    # 输出文件统一命名: 数据目录/<主名>_<时间戳>_<描述>.ext (参考 ts_signal_fft.py)
    save_path = os.path.join(os.path.dirname(fileIn), os.path.splitext(filename)[0])
    timestr = time.strftime('%Y%m%d%H%M%S')
    out_prefix = f'{save_path}_{timestr}_'

    # 数据导入：memmap不占内存，reshape为 (每通道采样数, 采集通道数)
    raw_data = read_bin_file(fileIn, 0, -1)
    n_total = len(raw_data) // cfg.PCI_CHANNEL * cfg.PCI_CHANNEL
    data = raw_data[:n_total].reshape(-1, cfg.PCI_CHANNEL)
    data = data[:, :cfg.channel_num]   # 只分析前 channel_num 个通道, 其余省略
    print(f'每通道采样点数: {data.shape[0]:,} (采集通道 {cfg.PCI_CHANNEL}, 分析通道 {cfg.channel_num})')

    # 阶段一: 三通道时域统计分析 (均值/RMS/峰峰值/标准差)
    stats_lines = collect_time_stats(fileIn, data, cfg.fs,
                                     cfg.channel_num, cfg.channel_names)
    print('\n' + '\n'.join(stats_lines))

    stats_path = f'{out_prefix}{cfg.STATS_SUFFIX}'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines) + '\n')
    print(f'\n统计信息已保存: {stats_path}')

    # 每通道统计图 (6层: 抽样数据/滑窗均值/滑窗标准差/滑窗RMS/滑窗峰峰值/有效频点能量-时间曲线)
    plot_channel_time_stats(data, out_prefix, cfg.fs,
                            cfg.channel_num, cfg.channel_names)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f'\n[错误] 文件不存在: {e}')
    except Exception as e:
        print(f'\n[错误] {e}')
        import traceback
        traceback.print_exc()
