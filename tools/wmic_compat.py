import ctypes
import re
import sys
from ctypes import wintypes


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_BASIC_INFORMATION_CLASS = 0


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll", use_last_error=True)


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", wintypes.LPWSTR),
    ]


class RTL_USER_PROCESS_PARAMETERS(ctypes.Structure):
    _fields_ = [
        ("Reserved1", ctypes.c_byte * 16),
        ("Reserved2", ctypes.c_void_p * 10),
        ("ImagePathName", UNICODE_STRING),
        ("CommandLine", UNICODE_STRING),
    ]


class PEB(ctypes.Structure):
    _fields_ = [
        ("Reserved1", ctypes.c_byte * 2),
        ("BeingDebugged", ctypes.c_byte),
        ("Reserved2", ctypes.c_byte),
        ("Reserved3", ctypes.c_void_p * 2),
        ("Ldr", ctypes.c_void_p),
        ("ProcessParameters", ctypes.c_void_p),
    ]


class PROCESS_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("Reserved1", ctypes.c_void_p),
        ("PebBaseAddress", ctypes.c_void_p),
        ("Reserved2", ctypes.c_void_p * 2),
        ("UniqueProcessId", ctypes.c_void_p),
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
    ]


kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wintypes.BOOL
ntdll.NtQueryInformationProcess.argtypes = [
    wintypes.HANDLE,
    wintypes.ULONG,
    wintypes.PVOID,
    wintypes.ULONG,
    ctypes.POINTER(wintypes.ULONG),
]
ntdll.NtQueryInformationProcess.restype = wintypes.LONG


def _parse_args(argv):
    joined = " ".join(argv)
    pid_match = re.search(r"processid\s*=\s*'?(?P<pid>\d+)'?", joined, re.IGNORECASE)
    pid = int(pid_match.group(1) if pid_match else 0)

    fields = []
    lowered = [arg.lower() for arg in argv]
    if "get" in lowered:
        index = lowered.index("get") + 1
        while index < len(argv):
            token = argv[index]
            if token.startswith("/"):
                break
            for field in token.split(","):
                cleaned = field.strip()
                if cleaned:
                    fields.append(cleaned.lower())
            index += 1

    if not fields:
        fields = ["commandline"]

    return pid, fields


def _open_process(pid):
    access = (
        PROCESS_QUERY_INFORMATION
        | PROCESS_QUERY_LIMITED_INFORMATION
        | PROCESS_VM_READ
    )
    handle = kernel32.OpenProcess(access, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    return handle


def _query_process_basic_info(handle):
    info = PROCESS_BASIC_INFORMATION()
    return_length = wintypes.ULONG()
    status = ntdll.NtQueryInformationProcess(
        handle,
        PROCESS_BASIC_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
        ctypes.byref(return_length),
    )
    if status != 0:
        raise OSError(status, "NtQueryInformationProcess failed")
    return info


def _read_remote_struct(handle, address, struct_type):
    data = struct_type()
    bytes_read = ctypes.c_size_t()
    success = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(address),
        ctypes.byref(data),
        ctypes.sizeof(data),
        ctypes.byref(bytes_read),
    )
    if not success:
        raise OSError(ctypes.get_last_error(), "ReadProcessMemory failed")
    return data


def _read_remote_unicode(handle, unicode_string):
    if not unicode_string.Buffer or unicode_string.Length == 0:
        return ""

    raw = ctypes.create_string_buffer(unicode_string.Length)
    bytes_read = ctypes.c_size_t()
    success = kernel32.ReadProcessMemory(
        handle,
        ctypes.cast(unicode_string.Buffer, ctypes.c_void_p),
        raw,
        unicode_string.Length,
        ctypes.byref(bytes_read),
    )
    if not success:
        raise OSError(ctypes.get_last_error(), "ReadProcessMemory failed")
    return raw.raw[: unicode_string.Length].decode("utf-16-le", errors="ignore")


def _get_command_line(handle, basic_info):
    peb = _read_remote_struct(handle, basic_info.PebBaseAddress, PEB)
    params = _read_remote_struct(
        handle, peb.ProcessParameters, RTL_USER_PROCESS_PARAMETERS
    )
    return _read_remote_unicode(handle, params.CommandLine)


def _get_executable_path(handle):
    buffer_size = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(buffer_size.value)
    success = kernel32.QueryFullProcessImageNameW(
        handle, 0, buffer, ctypes.byref(buffer_size)
    )
    if not success:
        raise OSError(ctypes.get_last_error(), "QueryFullProcessImageNameW failed")
    return buffer.value


def _get_values(pid, fields):
    handle = _open_process(pid)
    try:
        basic_info = _query_process_basic_info(handle)
        results = {}
        for field in fields:
            if field == "commandline":
                results["CommandLine"] = _get_command_line(handle, basic_info)
            elif field == "executablepath":
                results["ExecutablePath"] = _get_executable_path(handle)
            elif field == "parentprocessid":
                results["ParentProcessId"] = str(
                    int(basic_info.InheritedFromUniqueProcessId or 0)
                )
            elif field == "processid":
                results["ProcessId"] = str(pid)
            else:
                results[field] = ""
        return results
    finally:
        kernel32.CloseHandle(handle)


def main():
    pid, fields = _parse_args(sys.argv[1:])
    if pid <= 0:
        return 1

    try:
        values = _get_values(pid, fields)
    except Exception:
        values = {}
        for field in fields:
            canonical = {
                "commandline": "CommandLine",
                "executablepath": "ExecutablePath",
                "parentprocessid": "ParentProcessId",
                "processid": "ProcessId",
            }.get(field, field)
            values[canonical] = ""

    for key, value in values.items():
        print(f"{key}={value}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
