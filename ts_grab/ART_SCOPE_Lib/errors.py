from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import warnings

from ART_SCOPE_Lib.error_codes import Errors

__all__ = ['ArtScopeError']


class Error(Exception):
    """
    Base error class for module.
    """
    pass


class ArtScopeError(Error):
    """
    Error raised by any ArtScope method.
    """
    def __init__(self, message, error_code, task_name=''):
        """
        Args:
            message (string): Specifies the error message.
            error_code (int): Specifies the ART-SCOPE error code.
        """
        if task_name:
            message = '{0}\n\nTask Name: {1}'.format(message, task_name)

        super(ArtScopeError, self).__init__(message)

        self._error_code = error_code

        try:
            self._error_type = Errors(self._error_code)
        except ValueError:
            self._error_type = Errors.UNKNOWN

    @property
    def error_code(self):
        """
        int: Specifies the ART-SCOPE error code.
        """
        return self._error_code

    @property
    def error_type(self):
        """
        :class:`ART_SCOPE_Lib.error_codes.Errors`: Specifies the ART-SCOPE
            error type.
        """
        return self._error_type

def check_for_error(error_code):
    from ART_SCOPE_Lib.lib import lib_importer

    if error_code < 0:
        error_buffer = ctypes.create_string_buffer(2048)

        cfunc = lib_importer.windll.ArtScope_GetErrorString
        if cfunc.argtypes is None:
            #with cfunc.arglock:
                if cfunc.argtypes is None:
                    cfunc.argtypes = [ctypes.c_int32,
                                      ctypes.c_char_p,
                                      ctypes.c_uint]
        bRet = cfunc(error_code, error_buffer, 2048)

        message = 'Error.\n{0}Error Code: {1}'.format(error_buffer.value.decode("utf-8"), error_code)
        print(message)
        text = input("Press enter to exit.")
        exit()

class ArtScopeWarning(Warning):
    """
    Warning raised by any ART-SCOPE method.
    """
    def __init__(self, message, error_code):
        """
        Args:
            message (string): Specifies the warning message.
            error_code (int): Specifies the ART-SCOPE error code.
        """
        super(ArtScopeWarning, self).__init__(
            '\nWarning {0} occurred.\n\n{1}'.format(error_code, message))

        self._error_code = error_code

        try:
            self._error_type = Warning(self._error_code)
        except ValueError:
            self._error_type = ArtScopeWarning.UNKNOWN

    @property
    def error_code(self):
        """
        int: Specifies the ART-SCOPE error code.
        """
        return self._error_code

    @property
    def error_type(self):
        """
        :class:`ART_SCOPE_Lib.error_codes.ArtScopeWarning`: Specifies the ART-SCOPE
            error type.
        """
        return self._error_type