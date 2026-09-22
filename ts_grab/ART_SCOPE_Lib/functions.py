from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ctypes
import numpy
from ART_SCOPE_Lib.lib import lib_importer, wrapped_ndpointer, ctypes_byte_str
from ART_SCOPE_Lib.constants import *

class Functions(object):

    def ArtScope_init(resourceName, taskHandle):
        """
        create task

        Args:
            resourceName(c_char_p): task name
            taskHandle(c_void_p): task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_init
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str,
                              ctypes.POINTER(lib_importer.task_handle)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(resourceName, ctypes.byref(taskHandle))
        return error_code

    def ArtScope_Close(taskHandle):
        """
        close task

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_Close
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_ConfigureAcquisitionMode(taskHandle, acquisitionMode=SampleMode.FINITE):
        """
        set acquisition mode

        Args:
            taskHandle(c_void_p): task handle；
            acquisitionMode(c_int32): mode，=0:finite, =1:continuous
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureAcquisitionMode
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, acquisitionMode.value)
        return error_code

    def ArtScope_ConfigureVertical(taskHandle, channelList, range=InputRange.RANGE_2VPP, offset=0, coupling=CouplingType.DC, probeAttenuation=1, enabled=1):
        """
        set vertical parameters

        Args:
            taskHandle(c_void_p):task handle
            channelList(c_char_p):channel name list
            range(c_int32):range
            offset(c_int32):offset
            coupling(c_int32):coupling
            probeAttenuation(c_int32):not used
            enabled(c_uint16):channel enable or not
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureVertical
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_uint16]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, channelList, range.value, offset, coupling.value, probeAttenuation, enabled)
        return error_code

    def ArtScope_ConfigureChanCharacteristics(taskHandle, channelList, inputImpedance=InputImpedance.AI_IMPEND_1M, maxInputFrequency=BandWidth.BANDWIDTH_DEFAULT):
        """
        set channel parameters

        Args:
            taskHandle(c_void_p):task handle
            channelList(c_char_p):channel name list
            inputImpedance(c_int32):impedance
            maxInputFrequency(c_int32):bandwidth
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureChanCharacteristics
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, channelList, inputImpedance.value, maxInputFrequency.value)
        return error_code

    def ArtScope_ConfigureHorizontalTiming(taskHandle, minSampleRate, minNumPts, refPosition):
        """
        set horizontal parameters

        Args:
            taskHandle(c_void_p):task handle
            minSampleRate(c_double):sample rate
            minNumPts(c_int32):sample length
            refPosition(c_double):trigger position
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureHorizontalTiming
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_double,
                              ctypes.c_int32,
                              ctypes.c_double]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, minSampleRate, minNumPts, refPosition)
        return error_code

    def ArtScope_ConfigureTriggerEdge(taskHandle, triggerSource=TriggerSource.TRIGSRC_CH0, level=0.0, slope=TriggerSlope.TRIGDIR_NEGATIVE, triggerCoupling=TriggerCoupling.TRIGCOUP_DC, triggerCount=1, triggerSensitivity=50):
        """
        set trigger edge parameters

        Args:
            taskHandle(c_void_p):task handle
            triggerSource(c_char_p):trigger source
            level(c_double):trigger level
            slope(c_int32):trigger slope
            triggerCoupling(c_int32):trigger coupling
            triggerCount(c_int32):trigger count
            triggerSensitivity(c_int32):trigger sensitivity
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerEdge
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_double,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, triggerSource.value, level, slope.value, triggerCoupling.value, triggerCount, triggerSensitivity)
        return error_code

    def ArtScope_ConfigureTriggerWindow(taskHandle, triggerSource=TriggerSource.TRIGSRC_CH0, lowLevel=0.0, highLevel=0.0, windowMode=TriggerWindowType.WINDOW_ENTERING, triggerCoupling=TriggerCoupling.TRIGCOUP_DC, triggerCount=1, triggerSensitivity=50):
        """
        trigger window parameters

        Args:
            taskHandle(c_void_p):task handle
            triggerSource(c_char_p):trigger source
            lowLevel(c_double):trigger low level
            highLevel(c_double):trigger high level
            windowMode(c_int32):window type
            triggerCoupling(c_int32):trigger coupling
            triggerCount(c_int32):trigger count
            triggerSensitivity(c_int32):trigger sensitivity
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerWindow
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_double,
                              ctypes.c_double,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, triggerSource.value, lowLevel, highLevel, windowMode.value, triggerCoupling.value, triggerCount, triggerSensitivity)
        return error_code

    def ArtScope_ConfigureTriggerPulse(taskHandle, triggerSource=TriggerSource.TRIGSRC_CH0, level=0.0, polarity=TriggerPolarity.TRIGPOLAR_NEGATIVE, slope=TriggerSlope.TRIGDIR_NEGATIVE, width=4, triggerCoupling=TriggerCoupling.TRIGCOUP_DC, triggerCount=1, triggerSensitivity=50):
        """
        set trigger pulse parameters

        Args:
            taskHandle(c_void_p):task handle
            triggerSource(c_char_p):trigger source
            level(c_double):trigger level
            polarity(c_int32):trigger polarity
            slope(c_int32):trigger slope
            width(c_int32):pulse width
            triggerCoupling(c_int32):trigger coupling
            triggerCount(c_int32):trigger count
            triggerSensitivity(c_int32):trigger sensitivity
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerPulse
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_double,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, triggerSource.value, level, polarity.value, slope.value, width, triggerCoupling.value, triggerCount, triggerSensitivity)
        return error_code

    def ArtScope_ConfigureTriggerDigital(taskHandle, triggerSource=TriggerSource.TRIGSRC_CH0, slope=TriggerSlope.TRIGDIR_NEGATIVE, triggerCount=1, triggerSensitivity=50):
        """
        set trigger digital parameters

        Args:
            taskHandle(c_void_p):task handle
            triggerSource(c_char_p):trigger source
            slope(c_int32):trigger slope
            triggerCount(c_int32):trigger count
            triggerSensitivity(c_int32):trigger sensitivity
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerDigital
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_int32,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, triggerSource.value, slope.value, triggerCount, triggerSensitivity)
        return error_code

    def ArtScope_ConfigureTriggerHardwareDelay(taskHadle, delayPoints):
        """
        set hardware trigger parameters

        Args:
            taskHadle(c_void_p):task handle
            delayPoints(c_int32):delay points
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerHardwareDelay
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHadle, delayPoints)
        return error_code

    def ArtScope_ConfigureTriggerSoftWare(taskHandle):
        """
        set software trigger

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerSoftWare
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_ConfigureTriggerSyncSrc(taskHandle, triggerSyncSource=TriggerSyncSrc.TRIG_SYNC_0):
        """
        set synchronous trigger parameters

        Args:
            taskHandle(c_void_p):task handle
            triggerSyncSource(c_char_p):synchronous trigger source
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureTriggerSyncSrc
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, triggerSyncSource)
        return error_code

    def ArtScope_ConfigureClock(taskHandle, inputClockSource=InputClockSource.CLK_ONBOARD, freqDivision=1, masterEnabled=0):
        """
        set clock parameters

        Args:
            taskHandle(c_void_p):task handle
            inputClockSource(c_char_p):input clock source
            freqDivision(c_int32):frequency division coefficient,only inputClockSource is ARTSCOPE_VAL_CLK_IN,this parameter takes effect
            masterEnabled(c_uint32):multi card synchronization,set the main card
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ConfigureClock
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes_byte_str,
                              ctypes.c_int32,
                              ctypes.c_uint32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, inputClockSource.value, freqDivision, masterEnabled)
        return error_code

    def ArtScope_ExportClock(taskHandle, signal=ExportClockType.CLK_REF, signalIdentifier='', outputTerminal=''):
        """
        export clock

        Args:
            taskHandle(c_void_p):task handle
            signal(c_int32):export signal type
            signalIdentifier(c_char_p):Describes the signal being exported not used for now
            outputTerminal(c_char_p):not used for now
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ExportClock
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_int32,
                              ctypes_byte_str,
                              ctypes_byte_str]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, signal.value, signalIdentifier, outputTerminal)
        return error_code

    def ArtScope_ExportTrigger(taskHandle, trigOutWidth, trigOutPolarity=TriggerOutPolarity.TRIGOUT_POLAR_NEGATIVE):
        """
        export trigger

        Args:
            taskHandle(c_void_p):task handle
            trigOutWidth(c_int32):export trigger width
            trigOutPolarity(c_int32):export trigger polarity
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ExportTrigger
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_int32,
                              ctypes.c_int32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, trigOutWidth, trigOutPolarity.value)
        return error_code

    def ArtScope_AutoSaveFile(taskHandle, bIsAutoSave, chFileName):
        """
        auto save file

        Args:
            taskHandle(c_void_p):task handle
            bIsAutoSave(c_uint32):if 1,save file,if 0,not save
            chFileName(c_char_p):path of the save file
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_AutoSaveFile
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_uint32,
                              ctypes_byte_str]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, bIsAutoSave, chFileName)
        return error_code

    def ArtScope_InitiateAcquisition(taskHandle):
        """
        Initialize Acquisition

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_InitiateAcquisition
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_StartAcquisition(taskHandle):
        """
        // Start Acquisition

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_StartAcquisition
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_SendSoftwareTrigger(taskHandle):
        """
        send software trigger

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_SendSoftwareTrigger
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_AcquisitionStatus(taskHandle, status):
        """
        get acquisition status

        Args:
            taskHandle (c_void_p):task handle
            status (ARTSCOPE_STATUS_AD):status
        Returns:
            int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_AcquisitionStatus
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ARTSCOPE_STATUS_AD)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, status)
        return error_code

    def ArtScope_FetchBinary8(taskHandle, timeout, numSamples, wfmBinary, wfmInfo):
        """
        read binary data

        Args:
            taskHandle(c_void_p):task handle
            timeout(c_double):timeout
            numSamples(c_uint32):read length
            wfmBinary(uint8):data buffer
            wfmInfo(ArtScope_wfmInfo):information for analyze
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_FetchBinary8
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_double,
                              ctypes.c_uint32,
                              wrapped_ndpointer(dtype=numpy.uint8, flags=('C', 'W')),
                              ctypes.POINTER(ArtScope_wfmInfo)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, timeout, numSamples, wfmBinary, ctypes.byref(wfmInfo))
        return error_code

    def ArtScope_FetchBinary16(taskHandle, timeout, numSamples, wfmBinary, wfmInfo):
        """
        read binary data

        Args:
            taskHandle(c_void_p):task handle
            timeout(c_double):timeout
            numSamples(c_uint32):read length
            wfmBinary(uint16):data buffer
            wfmInfo(ArtScope_wfmInfo):information for analyze
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_FetchBinary16
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_double,
                              ctypes.c_uint32,
                              wrapped_ndpointer(dtype=numpy.uint16, flags=('C','W')),
                              ctypes.POINTER(ArtScope_wfmInfo)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, timeout, numSamples, wfmBinary, ctypes.byref(wfmInfo))
        return error_code

    def ArtScope_FetchVoltage(taskHandle, timeout, numSamples, wfmVoltage, wfmInfo):
        """
        read voltage data

        Args:
            taskHandle(c_void_p):task handle
            timeout(c_double):timeout
            numSamples(c_uint32):read length
            wfmVoltage(double):data buffer
            wfmInfo(ArtScope_wfmInfo):information for analyze
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_FetchVoltage
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.c_double,
                              ctypes.c_uint32,
                              wrapped_ndpointer(dtype=numpy.double, flags=('C','W')),
                              ctypes.POINTER(ArtScope_wfmInfo)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, timeout, numSamples, wfmVoltage, ctypes.byref(wfmInfo))
        return error_code

    def ArtScope_StopAcquisition(taskHandle):
        """
        stop acquisition

        Args:
            taskHandle:task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_StopAcquisition
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_ReleaseAcquisition(taskHandle):
        """
        Release Acquisition

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ReleaseAcquisition
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_ActualNumWfms(taskHandle, numWfms):
        """
        get acquisition channel count

        Args:
            taskHandle(c_void_p):task handle
            numWfms(c_uint32):channel count
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ActualNumWfms
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_uint32)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, ctypes.byref(numWfms))
        return error_code

    def ArtScope_ActualRecordLength(taskHandle, actualRecordLength):
        """
       get actual sample length

        Args:
            taskHandle(c_void_p):task handle
            actualRecordLength(c_uint32):actual sample length
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_ActualRecordLength
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, ctypes.byref(actualRecordLength))
        return error_code

    def ArtScope_SampleRate(taskHandle, actualSampleRate):
        """
        get actual sample rate

        Args:
            taskHandle(c_void_p):task handle
            actualSampleRate(c_double):actual sample rate
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_SampleRate
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, ctypes.byref(actualSampleRate))
        return error_code

    def ArtScope_Calibration(taskHandle):
        """
        calibrate

        Args:
            taskHandle(c_void_p):task handle
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_Calibration
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle)
        return error_code

    def ArtScope_Reset(resourceName):
        """
        reset device

        Args:
            resourceName(c_char_p):task name
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_Reset
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(resourceName)
        return error_code

    def ArtScope_GetInfoFromAutoSaveFile(chFileName, fileHeaderSize, wfmInfo):
        """
        get information from auto saved file,include the file header size,information used for conversion

        Args:
            chFileName(c_char_p):file name
            fileHeaderSize(c_int32):file header size
            wfmInfo(ArtScope_wfmInfo):parameters for conversion
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetInfoFromAutoSaveFile
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str,
                              ctypes.POINTER(ctypes.c_int32),
                              ctypes.POINTER(ArtScope_wfmInfo)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(chFileName, ctypes.byref(fileHeaderSize), ctypes.byref(wfmInfo))
        return error_code

    # error handle
    def ArtScope_GetErrorString(errorCode, errorString, bufferSize):
        """
        get error information

        Args:
            errorCode(c_int32):error code
            errorString(c_char_p):error message
            bufferSize(c_uint32):buffer size
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetErrorString
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_int32,
                              ctypes_byte_str,
                              ctypes.c_uint32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(errorCode, errorString, bufferSize)
        return error_code

    def ArtScope_GetExtendedErrorInfo(errorString, bufferSize):
        """
        get accumulation error

        Args:
            errorString(c_char_p):error message
            bufferSize(c_uint32):buffer size
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetExtendedErrorInfo
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str,
                              ctypes.c_uint32]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(errorString, bufferSize)
        return error_code

    def ArtScope_GetDeviceEEPInfo(devName, pEEPInfo):
        """
        get version information

        Args:
            devName(c_char_p):task name
            pEEPInfo(DEVICE_EEP_INFO):dll version,firmware version and calibrate information
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetDeviceEEPInfo
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str,
                              ctypes.POINTER(DEVICE_EEP_INFO)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(devName, ctypes.byref(pEEPInfo))
        return error_code

    def ArtScope_GetDevInformation(resourceName, channelCount, maxSampleRate):
        """
        get channel count and max sample rate the card supported

        Args:
            resourceName(c_char_p):task name
            channelCount(c_int32):channel count
            maxSampleRate(c_float):maximum sample rate
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetDevInformation
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes_byte_str,
                              ctypes.POINTER(ctypes.c_int32),
                              ctypes.POINTER(ctypes.c_float)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(resourceName, ctypes.byref(channelCount), ctypes.byref(maxSampleRate))
        return error_code

    def ArtScope_GetExtenedType(taskHandle, seriesKey):
        """
        get series number the card belongs to

        Args:
            taskHandle(c_void_p):task handle
            seriesKey(c_uint32):series number
        Returns:
            c_int32:
            0-right,<0-error
        """
        cfunc = lib_importer.windll.ArtScope_GetExtenedType
        if cfunc.argtypes is None:
            cfunc.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_uint32)]
        cfunc.restype = ctypes.c_int32
        error_code = cfunc(taskHandle, ctypes.byref(seriesKey))
        return error_code




