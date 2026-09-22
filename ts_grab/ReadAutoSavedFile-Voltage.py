# read a double bianary file
import sys
sys.path.append(r"../../../")

from ART_SCOPE_Lib.functions import Functions
from ART_SCOPE_Lib.constants import *
from ART_SCOPE_Lib.lib import *
from ART_SCOPE_Lib.errors import check_for_error, ArtScopeError

import ctypes
import numpy
import pprint
import msvcrt

pp = pprint.PrettyPrinter(indent=4)
fileName = input("Enter the full path of file(D:\\1.dat):")

# get information using api
bFileHeaderSize = ctypes.c_int32(0)
wfmInfo = ArtScope_wfmInfo()
wfmInfo.actualSamples = 0
wfmInfo.pAvailSampsPoints = 0
error_code = Functions.ArtScope_GetInfoFromAutoSaveFile(fileName, bFileHeaderSize, wfmInfo)
if error_code < 0:
    check_for_error(error_code)

# get data count
fp = open(fileName, "rb+")
fp.seek(0, 2)
byteSize = fp.tell()
fileDataLen = round((byteSize - bFileHeaderSize.value) / 8)  # one data has 8 bytes

# data buffer
voltDataBuffer = numpy.zeros(1024, dtype=numpy.double)

fp.seek(bFileHeaderSize.value)
while (fileDataLen > 0):
    if msvcrt.kbhit():
        if ord(msvcrt.getch()) != None:
            break
    if fileDataLen >= 1024:
        voltDataBuffer = struct.unpack('d'*1024,fp.read(1024*8))
        for index in range(1024):
            if index > 0 and (index % wfmInfo.channelCount) == 0:
                print("\n")
            print("%8.4f\t"%(voltDataBuffer[index]))
        print("\n")
    else:
        voltDataBuffer = struct.unpack('d'*fileDataLen,fp.read(fileDataLen*8))
        for index in range(fileDataLen):
            if index > 0 and (index % wfmInfo.channelCount) == 0:
                print("\n")
            print("%8.4f(mV)\t"%(voltDataBuffer[index]))
        print("\n")
    break
fp.close()
text = input("Press enter to exit.")


