import ctypes
import os
from ctypes import wintypes


_MUTEX_NAME = r"Local\ChintuAI.SingleInstance"
_ERROR_ALREADY_EXISTS = 183


class SingleInstanceGuard:
    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    def close(self) -> None:
        if self._handle is None:
            return

        handle = self._handle
        self._handle = None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        if not close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def acquire_single_instance(
    name: str = _MUTEX_NAME,
) -> SingleInstanceGuard | None:
    if os.name != "nt":
        raise OSError("The Chintu single-instance guard requires Windows.")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    create_mutex.restype = wintypes.HANDLE

    ctypes.set_last_error(0)
    handle = create_mutex(None, False, name)
    error_code = ctypes.get_last_error()
    if handle is None:
        raise ctypes.WinError(error_code)
    if error_code == _ERROR_ALREADY_EXISTS:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        if not close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        return None

    return SingleInstanceGuard(handle)
