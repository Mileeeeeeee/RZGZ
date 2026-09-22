from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from enum import Enum
import ctypes
from ctypes import *

class SampleMode(Enum):
    """
    acquisition mode
    """
    FINITE = 0      #: finite
    CONTINUOUS = 1  #: continuous

class CouplingType(Enum):
    """
    coupling
    """
    DC = 0  #: dc
    AC = 1  #: ac

class InputImpedance(Enum):
    """
    input impedance
    """
    AI_IMPEND_1M = 0  #: 1M
    AI_IMPEND_50 = 1  #: 50

class BandWidth(Enum):
    """
    bandwidth
    """
    BANDWIDTH_ALL = -1    #: full bandwidth
    BANDWIDTH_DEFAULT = 0 #: default bandwidth
    BANDWIDTH_100M = 1    #: 100MHz
    BANDWIDTH_20M = 2     #: 20MHz

class InputRange(Enum):
    """
    range
    """
    RANGE_20VPP = 0     #: ±10V
    RANGE_10VPP = 1     #: ±5V
    RANGE_2VPP = 2      #: ±1V
    RANGE_40VPP = 3     #: 5V/Div(40Vpp)
    RANGE_16VPP = 4     #: 2V/Div(16Vpp)
    RANGE_8VPP = 5      #: 1V/Div(8Vpp)
    RANGE_4VPP = 6      #: 500mV/Div(4Vpp)
    RANGE_1600MVPP = 7  #: 200mV/Div(1.6Vpp)
    RANGE_800MVPP = 8   #: 100mV/Div(800mVpp)
    RANGE_400MVPP = 9   #: 50mV/Div(400mVpp)
    RANGE_160MVPP = 10  #: 20mV/Div(160mVpp)
    RANGE_80MVPP = 11   #: 10mV/Div(80mVpp)
    RANGE_40MVPP = 12   #: 5mV/Div(40mVpp)

class TriggerSource(Enum):
    """
    trigger source
    """
    TRIGSRC_DTR = "VAL_TRIGGER_DTR"             #: external digital trigger
    TRIGSRC_ATR = "VAL_TRIGGER_ATR"             #: external analog trigger
    TRIGSRC_SYNC = "VAL_TRIGGER_SYNCTRIGGER"    #: synchronous trigger
    TRIGSRC_CH0 = "VAL_TRIGGER_CH0"             #: ch0 trigger
    TRIGSRC_CH1 = "VAL_TRIGGER_CH1"             #: ch1 trigger
    TRIGSRC_CH2 = "VAL_TRIGGER_CH2"             #: ch2 trigger
    TRIGSRC_CH3 = "VAL_TRIGGER_CH3"             #: ch3 trigger
    TRIGSRC_CH4 = "VAL_TRIGGER_CH4"             #: ch4 trigger
    TRIGSRC_CH5 = "VAL_TRIGGER_CH5"             #: ch5 trigger
    TRIGSRC_CH6 = "VAL_TRIGGER_CH6"             #: ch6 trigger
    TRIGSRC_CH7 = "VAL_TRIGGER_CH7"             #: ch7 trigger
    TRIGSRC_PFI = "VAL_TRIGGER_PFI"             #: PFI digital trigger(only 8582 8584 8586 8582A 8584A 8586A support)

class TriggerSyncSrc(Enum):
    """
    synchronous trigger source
    """
    TRIG_SYNC_0 = "VAL_SYNC_TRIGGER0"  #: synchronous trigger 0
    TRIG_SYNC_1 = "VAL_SYNC_TRIGGER1"  #: synchronous trigger 1
    TRIG_SYNC_2 = "VAL_SYNC_TRIGGER2"  #: synchronous trigger 2
    TRIG_SYNC_3 = "VAL_SYNC_TRIGGER3"  #: synchronous trigger 3
    TRIG_SYNC_4 = "VAL_SYNC_TRIGGER4"  #: synchronous trigger 4
    TRIG_SYNC_5 = "VAL_SYNC_TRIGGER5"  #: synchronous trigger 5
    TRIG_SYNC_6 = "VAL_SYNC_TRIGGER6"  #: synchronous trigger 6
    TRIG_SYNC_7 = "VAL_SYNC_TRIGGER7"  #: synchronous trigger 7

class TriggerSlope(Enum):
    """
    trigger direction
    """
    TRIGDIR_NEGATIVE = 0    #: negative
    TRIGDIR_POSITIVE = 1    #: positive
    TRIGDIR_NEGATPOSIT = 2  #: negative and positive

class TriggerPolarity(Enum):
    """
    trigger polarity
    """
    TRIGPOLAR_POSITIVE = 0  #: positive
    TRIGPOLAR_NEGATIVE = 1  #: negative

class TriggerCoupling(Enum):
    """
    trigger coupling
    """
    TRIGCOUP_DC = 0      #: dc
    TRIGCOUP_AC = 1      #: ac
    TRIGCOUP_HIG = 2     #: HF reject
    TRIGCOUP_LOW = 3     #: LF reject

class TriggerWindowType(Enum):
    """
    trigger window type
    """
    WINDOW_ENTERING = 0  #: in
    WINDOW_LEAVING = 1   #: out

class TriggerPulseDirection(Enum):
    """
    trigger pulse slope
    """
    TRIGDIR_PULSE_GREATER = 0   #: greater than the set pulse width
    TRIGDIR_PULSE_LESS = 1      #: lower than the set pulse width
    TRIGDIR_PULSE_EQUAL = 2     #: equal to the set pulse width
    TRIGDIR_PULSE_NEQUAL = 3    #: not equal to the set pulse width

class TriggerOutPolarity(Enum):
    """
    trigger out polarity
    """
    TRIGOUT_POLAR_NEGATIVE = 0  #: negative
    TRIGOUT_POLAR_POSITIVE = 1  #: positive

class InputClockSource(Enum):
    """
    input clock source
    """
    CLK_IN = "VAL_CLK_IN"                #: external clock(CLK_IN)(8582 8584 8586 don't support)
    CLK_ONBOARD = "VAL_CLK_INTERNAL"     #: on board clock
    CLK10M_EXT = "VAL_CLK10M_EXT"        #: external 10M clock(CLK_IN)(8582 8584 8586 don't support)
    CLK10M_PXI = "VAL_CLK10M_PXI"        #: PXIe_CLK10M(PXIE)/Master 10M(PCIE)
    CLK10M_MASTER = "VAL_CLK10M_MASTER"  #: Master 10M clock(PCIE or USB)
    CLK100M_PXIE = "VAL_CLK100M_PXIE"    #: PXIe_CLK100M(PCIE and USB don't support)

class ExportClockType(Enum):
    """
    export clock type
    """
    CLK_REF = 0     #: reference clock
    CLK_SAMPLE = 1  #: time base

class ARTSCOPE_STATUS_AD(Structure):
    """
    acquisition status

    Args:
        bADEanble(c_int32):AD enable or not =1 yes =0 no
        bTrigger(c_int32):AD trigger or not =1 trigger =0 not trigger
        bComplete(c_int32):AD acquisition done or not =1 yes =0 no(only finite mode support)
        bAheadTrig(c_int32):AD pre-trigger or not =1 yes =0 no
        lCanReadPoint(c_uint64):readable number(only finite mode support)
        lSavePoints(c_uint64):saved number
    """
    _fields_ = [("bADEanble", ctypes.c_int32),
                ("bTrigger", ctypes.c_int32),
                ("bComplete", ctypes.c_int32),
                ("bAheadTrig", ctypes.c_int32),
                ("lCanReadPoint", ctypes.c_uint64),
                ("lSavePoints", ctypes.c_uint64)]


class ArtScope_wfmInfo(Structure):
    """
    acquisition struct

    Args:
        fLsbs(c_double):each code value corresponding to the amplitude
        rangevalue(c_double):range
        wMaxLSB(c_uint32):maximum code
        TriggerMode(c_int32):trigger mode
        actualSamples(c_uint32):actual sample number
        pAvailSampsPoints(c_uint32):available points to read
        channelCount(c_int32):channel count
    """
    _fields_ = [("fLsbs", ctypes.c_double * 8),
                ("rangevalue", ctypes.c_double * 8),
                ("wMaxLSB", ctypes.c_uint32),
                ("TriggerMode", ctypes.c_int32),
                ("actualSamples", ctypes.c_uint32),
                ("pAvailSampsPoints", ctypes.c_uint32),
                ("channelCount", ctypes.c_int32)]


class DEVICE_EEP_INFO(Structure):
    """
    information about software：

    Args:
        nDLLVer(c_uint32):DLL version
        nSysVer(c_uint32):program version
        nFirmwareVer(c_uint32):firmware version
        nCaled(c_uint32):calibrate or not
        nCalDate(c_uint64):calibrate date
        nReserved(c_byte):reserve
    """
    _fields_ = [("nDLLVer", ctypes.c_uint32),
                ("nSysVer", ctypes.c_uint32),
                ("nFirmwareVer", ctypes.c_uint32),
                ("nCaled", ctypes.c_uint32),
                ("nCalDate", ctypes.c_uint64),
                ("nReserved", ctypes.c_byte*40)]