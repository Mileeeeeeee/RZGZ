import serial
import serial.tools.list_ports
import time
import struct
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

# ========== 1. 配置与数据类 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ChannelConfig:
    """通道配置数据类"""
    channel: str  # 'A', 'B', 'C'
    address: int  # Modbus寄存器地址
    gain_voltage: int = 0  # 增益电压 0~2047mV
    gain_gear: int = 0  # 增益档位 0~6


class DetectorController:
    """三路光电探测器增益控制器 (Modbus-RTU)"""

    # 寄存器地址映射 (来自协议文档)
    REG_SOFT_RESET = 0x0000
    REG_CHA_VOLTAGE = 0x0001
    REG_CHB_VOLTAGE = 0x0002
    REG_CHC_VOLTAGE = 0x0003
    REG_CHA_GEAR = 0x0004
    REG_CHB_GEAR = 0x0005
    REG_CHC_GEAR = 0x0006

    # 增益档位映射
    GEAR_MAP = {0: 1, 1: 2, 2: 3, 3: 5, 4: 9, 5: 15, 6: 20}

    # Modbus帧格式常量
    SLAVE_ID = 0x01
    FUNCTION_WRITE = 0x06  # 写单个寄存器
    RESET_VALUE = 8959  # 系统软重启特定值 (十进制)

    def __init__(self, port: str, baudrate: int = 19200, timeout: float = 1.0):
        """
        初始化控制器

        Args:
            port: 串口号，如 'COM3' 或 '/dev/ttyUSB0'
            baudrate: 波特率，默认19200
            timeout: 超时时间(秒)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False

    def connect(self) -> bool:
        """连接串口设备"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.is_connected = True
            logging.info(f"串口连接成功: {self.port} @ {self.baudrate}bps")
            return True
        except Exception as e:
            logging.error(f"串口连接失败: {e}")
            return False

    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            logging.info("串口已断开")

    # ========== 2. CRC16校验计算 (协议文档提供的算法) ==========
    @staticmethod
    def crc16(data: bytes) -> int:
        """
        Modbus CRC16 计算 (多项式: 0xA001)
        符合协议文档中的C代码实现
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

    # ========== 3. 构建Modbus帧 ==========
    def build_modbus_frame(self, register_addr: int, value: int) -> bytes:
        """
        构建Modbus-RTU写寄存器帧

        帧格式: SLAVE_ID | FUNC | ADDR_H | ADDR_L | DATA_H | DATA_L | CRC_L | CRC_H
        """
        # 数据部分 (寄存器地址 + 写入值)
        data = struct.pack('>B B H H',
                           self.SLAVE_ID,  # 从机地址
                           self.FUNCTION_WRITE,  # 功能码 0x06
                           register_addr,  # 寄存器地址 (16位)
                           value & 0xFFFF)  # 写入值 (16位)

        # 计算CRC
        crc = self.crc16(data)

        # 完整帧: 数据 + CRC低字节 + CRC高字节
        frame = data + struct.pack('<H', crc)  # '<H' = 小端字节序 (低字节在前)
        return frame

    # ========== 4. 发送指令并接收响应 ==========
    def send_command(self, register_addr: int, value: int, wait_response: bool = True) -> bool:
        """
        发送Modbus指令

        Args:
            register_addr: 寄存器地址
            value: 写入值 (0~65535)
            wait_response: 是否等待响应

        Returns:
            bool: 是否成功
        """
        if not self.is_connected or not self.ser:
            logging.error("设备未连接")
            return False

        # 构建帧
        frame = self.build_modbus_frame(register_addr, value)

        # 发送
        try:
            self.ser.write(frame)
            logging.debug(f"发送: {frame.hex().upper()}")

            if wait_response:
                # 等待响应 (标准Modbus响应长度 = 8字节)
                response = self.ser.read(8)
                if len(response) == 8:
                    logging.debug(f"响应: {response.hex().upper()}")
                    # 简单校验: 检查功能码是否一致 (实际应校验CRC和地址)
                    if response[1] == self.FUNCTION_WRITE:
                        return True
                    else:
                        logging.error(f"响应异常: 功能码不匹配")
                        return False
                else:
                    logging.error(f"响应超时或长度错误: {len(response)}字节")
                    return False
            return True

        except Exception as e:
            logging.error(f"发送指令异常: {e}")
            return False

    # ========== 5. 公共控制接口 ==========
    def set_gain_voltage(self, channel: str, voltage_mv: int) -> bool:
        """
        设置通道增益电压

        Args:
            channel: 'A', 'B', 'C'
            voltage_mv: 0~2047 mV

        Returns:
            bool: 是否成功
        """
        if not 0 <= voltage_mv <= 2047:
            logging.error(f"电压值 {voltage_mv} 超出范围 (0~2047)")
            return False

        addr_map = {
            'A': self.REG_CHA_VOLTAGE,
            'B': self.REG_CHB_VOLTAGE,
            'C': self.REG_CHC_VOLTAGE
        }
        addr = addr_map.get(channel.upper())
        if addr is None:
            logging.error(f"无效通道: {channel}")
            return False

        logging.info(f"设置通道{channel} 增益电压 = {voltage_mv}mV")
        return self.send_command(addr, voltage_mv)

    def set_gain_gear(self, channel: str, gear: int) -> bool:
        """
        设置通道增益档位

        Args:
            channel: 'A', 'B', 'C'
            gear: 0~6 (对应增益倍数: 1,2,3,5,9,15,20)

        Returns:
            bool: 是否成功
        """
        if not 0 <= gear <= 6:
            logging.error(f"档位 {gear} 超出范围 (0~6)")
            return False

        addr_map = {
            'A': self.REG_CHA_GEAR,
            'B': self.REG_CHB_GEAR,
            'C': self.REG_CHC_GEAR
        }
        addr = addr_map.get(channel.upper())
        if addr is None:
            logging.error(f"无效通道: {channel}")
            return False

        gain_value = self.GEAR_MAP.get(gear, 1)
        logging.info(f"设置通道{channel} 增益档位 = {gear} (×{gain_value})")
        return self.send_command(addr, gear)

    def set_channel_full(self, channel: str, voltage_mv: int, gear: int) -> bool:
        """
        完整配置单个通道 (电压 + 档位)

        Args:
            channel: 'A', 'B', 'C'
            voltage_mv: 0~2047 mV
            gear: 0~6

        Returns:
            bool: 全部成功返回True
        """
        success1 = self.set_gain_voltage(channel, voltage_mv)
        time.sleep(0.05)  # 短暂延时，确保前一条指令完成
        success2 = self.set_gain_gear(channel, gear)
        return success1 and success2

    def soft_reset(self) -> bool:
        """
        系统软重启 (写入特定值 8959 到寄存器 0x0000)

        Returns:
            bool: 是否成功
        """
        logging.warning("执行系统软重启...")
        return self.send_command(self.REG_SOFT_RESET, self.RESET_VALUE)

    # ========== 6. 工具函数 ==========
    @staticmethod
    def list_ports() -> List[str]:
        """列出所有可用串口"""
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def get_gain_gear_value(self, gear: int) -> int:
        """获取档位对应的实际增益倍数"""
        return self.GEAR_MAP.get(gear, 1)


# ========== 7. 使用示例 ==========
def main():
    """演示如何控制三路光电探测器"""

    # 1. 查找可用串口
    print("可用串口列表:", DetectorController.list_ports())

    # 2. 创建控制器实例 (修改为实际串口号)
    controller = DetectorController(port='COM3')  # Windows示例
    # controller = DetectorController(port='/dev/ttyUSB0')  # Linux示例

    # 3. 连接设备
    if not controller.connect():
        print("连接失败，请检查串口号和硬件连接")
        return

    try:
        # 4. 配置通道A: 增益电压1000mV, 档位3 (×5倍)
        print("\n--- 配置通道A ---")
        success = controller.set_channel_full('A', voltage_mv=1000, gear=3)
        print(f"通道A配置: {'成功' if success else '失败'}")

        # 5. 配置通道B: 增益电压500mV, 档位2 (×3倍)
        print("\n--- 配置通道B ---")
        success = controller.set_channel_full('B', voltage_mv=500, gear=2)
        print(f"通道B配置: {'成功' if success else '失败'}")

        # 6. 配置通道C: 增益电压1500mV, 档位5 (×15倍)
        print("\n--- 配置通道C ---")
        success = controller.set_channel_full('C', voltage_mv=1500, gear=5)
        print(f"通道C配置: {'成功' if success else '失败'}")

        # 7. 单独修改通道A的档位
        print("\n--- 单独修改通道A档位 ---")
        success = controller.set_gain_gear('A', gear=4)  # 改为×9倍
        print(f"通道A档位修改: {'成功' if success else '失败'}")

        # 8. (可选) 系统软重启
        # print("\n--- 执行软重启 ---")
        # controller.soft_reset()

    finally:
        # 9. 断开连接
        controller.disconnect()
        print("\n程序结束")


# ========== 8. 批量配置辅助函数 ==========
def batch_configure(port: str, configs: List[Tuple[str, int, int]]):
    """
    批量配置多个通道

    Args:
        port: 串口号
        configs: [(通道, 电压mV, 档位), ...]
    """
    controller = DetectorController(port)
    if not controller.connect():
        print(f"连接 {port} 失败")
        return

    try:
        for channel, voltage, gear in configs:
            print(f"配置通道{channel}: 电压={voltage}mV, 档位={gear}")
            success = controller.set_channel_full(channel, voltage, gear)
            print(f"  {'成功' if success else '失败'}")
            time.sleep(0.05)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()