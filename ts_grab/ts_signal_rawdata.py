"""
读取自动保存的二进制数据文件(bin)并转换为电压值
移植自 ART-SCOPE SDK 示例 ReadAutoSavedFile-Binary.py

与示例一致的读取/转换逻辑:
- 通过 ArtScope_GetInfoFromAutoSaveFile 获取文件头大小和转换参数:
    fLsbs      : 每码值对应的幅度 (mV/code)
    rangevalue : 量程偏移 (mV)
    wMaxLSB    : 最大码值掩码 (12bit 为 0x0FFF)
    channelCount: 通道数 (多通道按 index % channelCount 交错存储)
- 每个采样点 2 字节 (uint16)
- 电压(mV) = fLsbs[index % channelCount] * (code & wMaxLSB) - rangevalue[index % channelCount]
- 码值高位(bit15~12)为标志位，转换时用 wMaxLSB 掩码屏蔽

兼容 ACTS1000 图形界面保存的文件:
DLL 不支持此类文件(返回 "The file type is incorrect")，
此时从同名 _header.txt 解析等价参数(chanEnableCount / SampleRate / SampleLen /
resolution / {ch}rangeMaxValue / {ch}rangeMinValue)。
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from tqdm import tqdm

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
    return {
        "header_bytes": header_size.value,
        "channel_count": channel_count,
        "f_lsbs": np.array(wfm_info.fLsbs[:channel_count], dtype=np.float64),      # mV/code
        "range_value": np.array(wfm_info.rangevalue[:channel_count], dtype=np.float64),  # mV
        "w_max_lsb": int(wfm_info.wMaxLSB),
        "sample_rate": 100_000_000.0,   # wfmInfo 不含采样率，按 100MHz 计
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

    # 按示例公式 V(mV) = fLsbs * code - rangevalue 反推等价参数:
    #   code=0        -> fLsbs*0 - rangevalue = rangeMin   => rangevalue = -rangeMin
    #   code=wMaxLSB  -> fLsbs*wMaxLSB - rangevalue ≈ rangeMax
    f_lsbs = np.empty(channel_count, dtype=np.float64)
    range_value = np.empty(channel_count, dtype=np.float64)
    for ch in range(channel_count):
        range_max = float(params[f"{ch}rangeMaxValue"])   # mV
        range_min = float(params[f"{ch}rangeMinValue"])   # mV
        f_lsbs[ch] = (range_max - range_min) / (w_max_lsb + 1)
        range_value[ch] = -range_min

    return {
        "header_bytes": 0,          # 界面保存的 bin 无文件头
        "channel_count": channel_count,
        "f_lsbs": f_lsbs,
        "range_value": range_value,
        "w_max_lsb": w_max_lsb,
        "sample_rate": float(params["SampleRate"]),
    }


CHUNK_SAMPLES = 10_000_000   # 分块转换，避免一次性占用过多内存


def read_bin_to_voltage(bin_file_path):
    """
    读取 bin 文件并转换为电压值

    参数:
        bin_file_path: bin文件路径

    返回:
        voltage_data: numpy.ndarray (float32)，电压值 (V)
        sample_count: 采样点数
        sample_rate : 采样率 (Hz)
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
    f_lsbs = info["f_lsbs"]
    range_value = info["range_value"]
    w_max_lsb = info["w_max_lsb"]

    print("=" * 60)
    print("bin 数据读取 (uint16, 2字节/点)")
    print("=" * 60)
    print(f"文件头大小: {header_bytes} 字节")
    print(f"通道数: {channel_count} (交错存储)")
    print(f"最大码值掩码: 0x{w_max_lsb:X}")
    print(f"各通道 fLsbs (mV/code): {f_lsbs}")
    print(f"各通道 rangevalue (mV): {range_value}")
    print("=" * 60)

    # ========== 2. 内存映射读取原始码值 ==========
    file_size = os.path.getsize(bin_file_path)
    total_samples = round((file_size - header_bytes) / 2)   # 每采样点2字节
    print(f"文件实际大小: {file_size / (1024**2):.1f} MB, 采样点数: {total_samples:,}")

    raw_data = np.memmap(bin_file_path,dtype=np.uint16,mode='r',offset=header_bytes,shape=(total_samples,),)

    # ========== 3. 码值转电压 (与示例公式一致，分块处理) ==========
    n_use = total_samples - total_samples % channel_count   # 丢弃不足一组的尾部数据
    if n_use == 0:
        raise ValueError("没有有效数据!")

    voltage_data = np.empty(n_use, dtype=np.float32)   # 单位 mV
    chunk = CHUNK_SAMPLES - CHUNK_SAMPLES % channel_count
    code_min, code_max = np.inf, -np.inf
    for start in range(0, n_use, chunk):
        end = min(start + chunk, n_use)
        block = raw_data[start:end].reshape(-1, channel_count)   # 每行一组交错通道
        codes = block & w_max_lsb                                 # 屏蔽高位标志位
        code_min = min(code_min, int(codes.min()))
        code_max = max(code_max, int(codes.max()))
        voltage_data[start:end] = (
            codes * f_lsbs[None, :] - range_value[None, :]
        ).ravel()
        print(f"  转换进度: {end}/{n_use}")

    voltage_data *= 1e-3   # mV -> V

    # ========== 4. 统计信息 ==========
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"有效数据点数: {n_use:,}")
    print(f"ADC码值范围: [{code_min}, {code_max}] (掩码后)")
    print(f"电压范围: [{voltage_data.min():.4f}, {voltage_data.max():.4f}] V")
    print(f"电压平均值: {voltage_data.mean():.4f} V")
    print(f"电压标准差: {voltage_data.std():.4f} V")
    print("=" * 60)

    return voltage_data, n_use, info["sample_rate"]


def read_csv_to_voltage(file_path, chunk_size=50000, sampling_rate=None):
    """
    分块读取大型CSV文件并转换为电压数据

    参数:
        file_path: CSV文件路径
        chunk_size: 每块的行数，根据内存调整
        sampling_rate: 采样率（Hz），如果提供则返回

    返回:
        voltage_data: numpy数组，形状为 (总行数, 通道数)
        num_samples: 数据总长度（行数）
        sampling_rate: 采样率（如果提供）
        channels: 通道名称列表
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
    # 使用float32节省内存，精度足够
    voltage_data = np.empty((total_rows, num_channels), dtype=np.float32)

    # 分块读取并处理
    print("开始处理数据...")
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    row_counter = 0
    with tqdm(total=total_rows, desc="处理进度", unit="行") as pbar:
        for chunk in chunk_reader:
            # 提取原始数据并转换
            rawdata = chunk.values.astype(np.float32)
            voltage = (rawdata - 2048) * 10 / 4096

            # 填充到预分配的数组中
            current_chunk_size = len(voltage)
            voltage_data[row_counter:row_counter + current_chunk_size, :] = voltage

            row_counter += current_chunk_size
            pbar.update(current_chunk_size)

            # 释放内存
            del chunk, rawdata, voltage
            if row_counter % (chunk_size * 10) == 0:
                gc.collect()

    print(f"数据读取完成！共 {row_counter:,} 行")

    # 返回结果
    result = {
        'voltage_data': voltage_data,
        'num_samples': row_counter,
        'channels': channels,
        'shape': voltage_data.shape
    }

    if sampling_rate is not None:
        result['sampling_rate'] = sampling_rate
        result['duration'] = row_counter / sampling_rate  # 持续时间（秒）
        print(f"采样率: {sampling_rate} Hz")
        print(f"数据时长: {row_counter / sampling_rate:.2f} 秒")

    return result

# ========== 主程序 ==========
if __name__ == "__main__":

    # 直接读取本地bin文件
    bin_file = r"E:\system\data\1550\ACTS1000_data_100Msps_1s_10kS_20260820_20260820164318108_0.csv"  # 修改为你的文件路径
    file_path = os.path.dirname(bin_file)
    save_name = os.path.splitext(os.path.basename(bin_file))[0]
    save_path = os.path.join(file_path, save_name)
    try:
        file_type = os.path.splitext(os.path.basename(bin_file))[1]
        if file_type == '.csv':
            result = read_csv_to_voltage(bin_file, chunk_size=1000000, sampling_rate=100000000)
            voltage = result['voltage_data']  # numpy数组
            count = result['num_samples']  # 数据长度
            fs = result['sampling_rate']  # 通道名称

        elif file_type == '.bin':
            voltage, count, fs = read_bin_to_voltage(bin_file)
        else:
            print('数据格式不符合 csv/bin')

        print(f"\n[成功] 读取成功: {count:,} 个电压值, 采样率 {fs/1e6:.0f} MHz")
        print(f"前10个电压值: {voltage[:10]}")

        # ========== 画图: 电压-时间曲线 ==========


        # 只为画图范围生成时间轴，避免为全部点分配大数组
        time_axis = np.arange(count) / fs  # 时间轴 (s)
        n_plot = 100_000       # 只画前*个点

        plt.figure(figsize=(12, 5))
        plt.plot(time_axis[:n_plot], voltage[:n_plot], linewidth=0.5)
        plt.xlabel("time (s)")
        plt.ylabel("voltage (V)")
        plt.title(f"SampleRate {fs/1e6:.0f} MHz, first {n_plot:,} samples")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.savefig(f'{save_path}.jpg')
        print(f'save figure to: {save_path}.jpg')

    except FileNotFoundError:
        print(f"\n[错误] 找不到文件 '{bin_file}' 或其头部参数文件")
        print("请修改 bin_file 变量为你的实际文件路径")
    except Exception as e:
        print(f"\n[错误] {e}")
