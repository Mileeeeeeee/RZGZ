from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
from numpy.ctypeslib import ndpointer
import platform
import struct
import six
import sys

class CtypesByteString(object):
    """
    Custom argtype that automatically converts unicode strings to ASCII strings in Python 3.
    """
    def from_param(self, param):
        if isinstance(param, six.text_type):
            param = param.encode('ascii')
        return ctypes.c_char_p(param)

ctypes_byte_str = CtypesByteString()

def wrapped_ndpointer(*args, **kwargs):
    """
    Specifies an ndpointer type that wraps numpy.ctypeslib.ndpointer and
    allows a value of None to be passed to an argument of that type.
    """
    if sys.version_info < (3,):
        if 'flags' in kwargs:
            kwargs['flags'] = tuple(
                f.encode('ascii') for f in kwargs['flags'])

    base = ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(base.__name__, (base,),
                {'from_param': classmethod(from_param)})


class ARTSCOPELibImporter(object):
    """
    Encapsulates ACTS1200 library importing and handle type parsing logic.
    """

    def __init__(self):
        self._windll = None
        self._cdll = None
        self._cal_handle = None
        self._task_handle = None

    def _import_lib(self):
        """
        Determines the location of and loads the ART_SCOPE DLL
        """
        self._windll = None
        self._cdll = None

        windll = None
        cdll = None

        if sys.platform.startswith('win'):
            lib_name = "ART_SCOPE"

            if sys.version_info < (3,):
                lib_name = lib_name.encode('ascii')

            if 'iron' in platform.python_implementation().lower():
                self._windll = ctypes.windll.ART_SCOPE
            else:
                self._windll = ctypes.windll.LoadLibrary(lib_name)
        else:
            raise SCOPENotFoundError(
                'ART_SCOPE Python is not supported on this platform:{0}.'
                'Please direct any questions or feedback to Art Technology'.format(sys.platform))

    def _parse_typedefs(self):
        self._task_handle = ctypes.c_void_p
        self._cal_handle = ctypes.c_uint

    @property
    def windll(self):
        if self._windll is None:
            self._import_lib()
        return self._windll

    @property
    def cdll(self):
        if self._cdll is None:
            self._import_lib()
        return self._cdll

    @property
    def task_handle(self):
        if self._task_handle is None:
            self._parse_typedefs()
        return self._task_handle

    @property
    def cal_handle(self):
        if self._cal_handle is None:
            self._parse_typedefs()
        return self._cal_handle


lib_importer = ARTSCOPELibImporter()
