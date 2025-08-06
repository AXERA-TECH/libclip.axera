import ctypes
import os
import platform

def check_error(code: int) -> None:
    if code != 0:
        raise Exception(f"API错误: {code}")

base_dir = os.path.dirname(__file__)
arch = platform.machine()

if arch == 'x86_64':
    arch_dir = 'x86_64'
elif arch in ('aarch64', 'arm64'):
    arch_dir = 'aarch64'
else:
    raise RuntimeError(f"Unsupported architecture: {arch}")

lib_paths = [
    os.path.join(base_dir, arch_dir, 'libclip.so'),
    os.path.join(base_dir, 'libclip.so')
]

last_error = None
diagnostic_shown = set()

for lib_path in lib_paths:
    try:
        print(f"Trying to load: {lib_path}")
        _lib = ctypes.CDLL(lib_path)
        print(f"✅ Successfully loaded: {lib_path}")
        break
    except OSError as e:
        last_error = e
        err_str = str(e)
        print(f"\n❌ Failed to load: {lib_path}")
        print(f"   {err_str}")

        # Only show GLIBCXX tip once
        if "GLIBCXX" in err_str and "not found" in err_str:
            if "missing_glibcxx" not in diagnostic_shown:
                diagnostic_shown.add("missing_glibcxx")
                print("🔍 Detected missing GLIBCXX version in libstdc++.so.6")
                print("💡 This usually happens when your environment (like Conda) uses an older libstdc++")
                print(f"👉 Try running with system libstdc++ preloaded:")
                print(f"   export LD_PRELOAD=/usr/lib/{arch_dir}-linux-gnu/libstdc++.so.6\n")
        elif "No such file" in err_str:
            if "file_not_found" not in diagnostic_shown:
                diagnostic_shown.add("file_not_found")
                print("🔍 File not found. Please verify that libclip.so exists and the path is correct.\n")
        elif "wrong ELF class" in err_str:
            if "elf_mismatch" not in diagnostic_shown:
                diagnostic_shown.add("elf_mismatch")
                print("🔍 ELF class mismatch — likely due to architecture conflict (e.g., loading x86_64 .so on aarch64).")
                print(f"👉 Run `file {lib_path}` to verify the binary architecture.\n")
        else:
            if "generic_error" not in diagnostic_shown:
                diagnostic_shown.add("generic_error")
                print("📎 Tip: Use `ldd` to inspect missing dependencies:")
                print(f"   ldd {lib_path}\n")
else:
    raise RuntimeError(f"\n❗ Failed to load libclip.so.\nLast error:\n{last_error}")


# 定义枚举类型
class AxDeviceType(ctypes.c_int):
    unknown_device = 0
    host_device = 1
    axcl_device = 2

# 定义结构体
class AxMemInfo(ctypes.Structure):
    _fields_ = [
        ('remain', ctypes.c_int),
        ('total', ctypes.c_int)
    ]

class AxHostInfo(ctypes.Structure):
    _fields_ = [
        ('available', ctypes.c_char),
        ('version', ctypes.c_char * 32),
        ('mem_info', AxMemInfo)
    ]

class AxDeviceInfo(ctypes.Structure):
    _fields_ = [
        ('temp', ctypes.c_int),
        ('cpu_usage', ctypes.c_int),
        ('npu_usage', ctypes.c_int),
        ('mem_info', AxMemInfo)
    ]

class AxDevices(ctypes.Structure):
    _fields_ = [
        ('host', AxHostInfo),
        ('host_version', ctypes.c_char * 32),
        ('dev_version', ctypes.c_char * 32),
        ('count', ctypes.c_ubyte),
        ('devices_info', AxDeviceInfo * 16)
    ]


_lib.ax_dev_enum_devices.argtypes = [ctypes.POINTER(AxDevices)]
_lib.ax_dev_enum_devices.restype = ctypes.c_int

_lib.ax_dev_sys_init.argtypes = [AxDeviceType, ctypes.c_char]
_lib.ax_dev_sys_init.restype = ctypes.c_int

_lib.ax_dev_sys_deinit.argtypes = [AxDeviceType, ctypes.c_char]
_lib.ax_dev_sys_deinit.restype = ctypes.c_int

def enum_devices() -> dict:
    devices = AxDevices()
    check_error(_lib.ax_dev_enum_devices(ctypes.byref(devices)))
    
    return {
        'host': {
            'available': bool(devices.host.available[0]),
            'version': devices.host.version.decode('utf-8'),
            'mem_info': {
                'remain': devices.host.mem_info.remain,
                'total': devices.host.mem_info.total
            }
        },
        'devices': {
            'host_version': devices.host_version.decode('utf-8'),
            'dev_version': devices.dev_version.decode('utf-8'),
            'count': devices.count,
            'devices_info': [{
                'temp': dev.temp,
                'cpu_usage': dev.cpu_usage,
                'npu_usage': dev.npu_usage,
                'mem_info': {
                    'remain': dev.mem_info.remain,
                    'total': dev.mem_info.total
                }
            } for dev in devices.devices_info[:devices.count]]
        }
    }


def sys_init(dev_type: AxDeviceType = AxDeviceType.axcl_device, devid: int = 0) -> None:
    check_error(_lib.ax_dev_sys_init(dev_type, devid))


def sys_deinit(dev_type: AxDeviceType = AxDeviceType.axcl_device, devid: int = 0) -> None:
    check_error(_lib.ax_dev_sys_deinit(dev_type, devid))