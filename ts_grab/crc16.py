"""CRC16 Modbus 校验 & 增益控制指令生成"""


def crc16_modbus(data: bytes) -> int:
    """
    CRC16 Modbus 算法
    多项式: 0x8005 (反转 0xA001)
    初始值: 0xFFFF
    结果异或: 0x0000
    输入输出反转: True
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_command(slave_id: int, func: int, reg_addr: int, data_val: int) -> bytes:
    """
    构建 Modbus RTU 指令帧 (8 字节)
    帧格式: [SLAVE][FUNC][ADDR_H][ADDR_L][DATA_H][DATA_L][CRC_L][CRC_H]
    CRC 计算覆盖前 6 字节，低字节在前
    """
    frame = bytes([
        slave_id & 0xFF,
        func & 0xFF,
        (reg_addr >> 8) & 0xFF,
        reg_addr & 0xFF,
        (data_val >> 8) & 0xFF,
        data_val & 0xFF,
    ])
    crc = crc16_modbus(frame)
    # Modbus RTU: CRC 低字节在前
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


# ── 增益档位对照表 ──
GAIN_TABLE = {
    0: ("×1",  0x0000),
    1: ("×2",  0x0001),
    2: ("×3",  0x0002),
    3: ("×5",  0x0003),
    4: ("×9",  0x0004),
    5: ("×15", 0x0005),
    6: ("×20", 0x0006),
}

# 寄存器地址
REG_SYSTEM_RESET   = 0x0000
REG_CH_A_GAIN_VOLTAGE = 0x0001
REG_CH_B_GAIN_VOLTAGE = 0x0002
REG_CH_C_GAIN_VOLTAGE = 0x0003
REG_CH_A_GAIN_STEP = 0x0004
REG_CH_B_GAIN_STEP = 0x0005
REG_CH_C_GAIN_STEP = 0x0006

SLAVE_ID = 0x01
FUNC_WRITE = 0x06


def set_gain_step(channel: str, step: int) -> bytes:
    """设置通道增益档位，返回完整指令帧"""
    addr_map = {"A": REG_CH_A_GAIN_STEP, "B": REG_CH_B_GAIN_STEP, "C": REG_CH_C_GAIN_STEP}
    if channel not in addr_map:
        raise ValueError(f"通道无效: {channel}，可选 A/B/C")
    if step not in GAIN_TABLE:
        raise ValueError(f"档位无效: {step}，可选 0~6")
    return build_command(SLAVE_ID, FUNC_WRITE, addr_map[channel], step)


def set_gain_voltage(channel: str, voltage_mv: int) -> bytes:
    """设置通道增益电压 (0~2047 mV)，返回完整指令帧"""
    addr_map = {"A": REG_CH_A_GAIN_VOLTAGE, "B": REG_CH_B_GAIN_VOLTAGE, "C": REG_CH_C_GAIN_VOLTAGE}
    if channel not in addr_map:
        raise ValueError(f"通道无效: {channel}，可选 A/B/C")
    if not (0 <= voltage_mv <= 2047):
        raise ValueError(f"电压超出范围: {voltage_mv}，范围 0~2047 mV")
    return build_command(SLAVE_ID, FUNC_WRITE, addr_map[channel], voltage_mv)


def system_reset() -> bytes:
    """系统软复位，写入特定值 8959 (0x22FF)"""
    return build_command(SLAVE_ID, FUNC_WRITE, REG_SYSTEM_RESET, 8959)


def format_hex(frame: bytes) -> str:
    """格式化输出十六进制指令"""
    return " ".join(f"{b:02X}" for b in frame)


if __name__ == "__main__":
    # 打印所有档位指令
    ch = "A"
    print(f"增益档位指令 (通道 {ch}):")
    for step, (label, _) in GAIN_TABLE.items():
        cmd = set_gain_step(ch, step)
        print(f"  档位{step} {label}: {format_hex(cmd)}")

    # print("\n" + "=" * 50)
    # print("增益电压指令 (通道 A, 1000mV):")
    # cmd = set_gain_voltage("A", 1000)
    # print(f"  {format_hex(cmd)}")
    #
    # print("\n" + "=" * 50)
    # print("系统软复位指令:")
    # cmd = system_reset()
    # print(f"  {format_hex(cmd)}")
    #
    # print("\n" + "=" * 50)
    # print("CRC 校验验证:")
    # # 手动验证一个已知帧
    # frame = bytes([0x01, 0x06, 0x00, 0x04, 0x00, 0x01])
    # crc = crc16_modbus(frame)
    # print(f"  数据: {format_hex(frame)}")
    # print(f"  CRC:  {crc:04X} (低字节: {crc & 0xFF:02X}, 高字节: {(crc >> 8) & 0xFF:02X})")
    # print(f"  完整帧: {format_hex(frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))}")