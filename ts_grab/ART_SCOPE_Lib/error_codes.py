from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from enum import Enum

__all__ = ['Errors', 'Warnings']


class Errors(Enum):
    """
    error code
    """
    ArtSCOPError_AllocateMemoryError = -171298
    ArtSCOPError_CardNotSupport = -229774
    ArtSCOPError_OpenFileError = -229773
    ArtSCOPError_FileTypeError = -229772
    ArtSCOPError_FileSaveError = -229771
    ArtSCOPError_ConfigureNotSupport = -229770
    ArtSCOPError_BufferTooSmallForString = -200228
    ArtSCOPError_NULLPtr = -200604
    ArtSCOPError_InvalidTaskName = -201340
    ArtSCOPError_DeviceNotExist = -200220
    ArtSCOPError_ReleaseDeviceError = -200219
    ArtSCOPError_ChanDisable = -200218
    ArtSCOPError_InvalidRecord = -200217
    ArtSCOPError_InitAcquisitionError = -200216
    ArtSCOPError_SoftwareTriggerError = -200215
    ArtSCOPError_GetStatusError = -200214
    ArtSCOPError_ReadDataError = -200213
    ArtSCOPError_ReadDataTimeout = -200299
    ArtSCOPError_StopError = -200212
    ArtSCOPError_GetMainInfoError = -200211
    ArtSCOPError_InvalidSampleRate = -200210
    ArtSCOPError_InvalidRange = -200209
    ArtSCOPError_InvalidPhysChanString = -200208
    ArtSCOPError_InvalidSampleMode = -200207
    ArtSCOPError_InvalidCoupling = -200206
    ArtSCOPError_InvalidTriggerPara = -200205
    ArtSCOPError_InvalidRefposition = -200204
    ArtSCOPError_DevAlreadyInTask = -200481
    ArtSCOPError_InvalidTask = -200088
    ArtSCOPError_InvalidTimeOut = -200087
    ArtSCOPError_ClearTaskError = -200086
    ArtSCOPError_SubmitVIError = -200085
    ArtSCOPError_GetVIError = -200084
    ArtSCOPError_ReadBufferTooSmall = -200083
    ArtSCOPErroe_ADCalibrationError = -200082
    ArtSCOPError_GetActualSampleRateError = -200081
    ArtSCOPError_InvalidFileSavePath = -200080
    ArtSCOPError_StartAcquisitionError = -200079
    ArtSCOPError_ReleaseADError = -200078
    ArtSCOPError_GetVersionError = -200077
    ArtSCOPError_InvalidReadLength = -200076
    ArtSCOPError_GetBusInfoError = -200075
    ArtSCOPError_GetSerialNumberError = -200074
    UNKNOWN = -1
