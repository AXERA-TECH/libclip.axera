import ctypes
import os
from typing import List, Tuple, Optional
import numpy as np
import platform

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
printed_diagnostics = set()

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

        # Deduplicate diagnostics
        if err_str not in printed_diagnostics:
            printed_diagnostics.add(err_str)

            if "GLIBCXX" in err_str and "not found" in err_str:
                print("🔍 Detected missing GLIBCXX version in libstdc++.so.6")
                print("💡 This usually happens when your environment (like Conda) uses an older libstdc++")
                print(f"👉 Try running with system libstdc++ preloaded:")
                print(f"   export LD_PRELOAD=/usr/lib/{arch_dir}-linux-gnu/libstdc++.so.6\n")
            elif "No such file" in err_str:
                print("🔍 File not found. Please verify that libclip.so exists and the path is correct.\n")
            elif "wrong ELF class" in err_str:
                print("🔍 ELF class mismatch — likely due to architecture conflict (e.g., loading x86_64 .so on aarch64).")
                print(f"👉 Run `file {lib_path}` to verify the binary architecture.\n")
            else:
                print("📎 Tip: Use `ldd` to inspect missing dependencies:")
                print(f"   ldd {lib_path}\n")
else:
    raise RuntimeError(f"\n❗ Failed to load libclip.so.\nLast error:\n{last_error}")


# 定义枚举类型
class ClipDeviceType(ctypes.c_int):
    unknown_device = 0
    host_device = 1
    axcl_device = 2

# 定义结构体
class ClipMemInfo(ctypes.Structure):
    _fields_ = [
        ('remain', ctypes.c_int),
        ('total', ctypes.c_int)
    ]

class ClipHostInfo(ctypes.Structure):
    _fields_ = [
        ('available', ctypes.c_char),
        ('version', ctypes.c_char * 32),
        ('mem_info', ClipMemInfo)
    ]

class ClipDeviceInfo(ctypes.Structure):
    _fields_ = [
        ('temp', ctypes.c_int),
        ('cpu_usage', ctypes.c_int),
        ('npu_usage', ctypes.c_int),
        ('mem_info', ClipMemInfo)
    ]

class ClipDevices(ctypes.Structure):
    _fields_ = [
        ('host', ClipHostInfo),
        ('host_version', ctypes.c_char * 32),
        ('dev_version', ctypes.c_char * 32),
        ('count', ctypes.c_ubyte),
        ('devices_info', ClipDeviceInfo * 16)
    ]

class ClipInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', ClipDeviceType),
        ('devid', ctypes.c_char),
        ('text_encoder_path', ctypes.c_char * 128),
        ('image_encoder_path', ctypes.c_char * 128),
        ('tokenizer_path', ctypes.c_char * 128),
        ('isCN', ctypes.c_char),
        ('db_path', ctypes.c_char * 128)
    ]

class ClipImage(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('channels', ctypes.c_int),
        ('stride', ctypes.c_int)
    ]

class ClipResultItem(ctypes.Structure):
    _fields_ = [
        ('key', ctypes.c_char * 64),
        ('score', ctypes.c_float)
    ]

# 设置函数参数和返回类型
_lib.clip_enum_devices.argtypes = [ctypes.POINTER(ClipDevices)]
_lib.clip_enum_devices.restype = ctypes.c_int

_lib.clip_sys_init.argtypes = [ClipDeviceType, ctypes.c_char]
_lib.clip_sys_init.restype = ctypes.c_int

_lib.clip_sys_deinit.argtypes = [ClipDeviceType, ctypes.c_char]
_lib.clip_sys_deinit.restype = ctypes.c_int

_lib.clip_create.argtypes = [ctypes.POINTER(ClipInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.clip_create.restype = ctypes.c_int

_lib.clip_destroy.argtypes = [ctypes.c_void_p]
_lib.clip_destroy.restype = ctypes.c_int

_lib.clip_add.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ClipImage), ctypes.c_char]
_lib.clip_add.restype = ctypes.c_int

_lib.clip_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.clip_remove.restype = ctypes.c_int

_lib.clip_contain.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.clip_contain.restype = ctypes.c_int

_lib.clip_match_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ClipResultItem), ctypes.c_int]
_lib.clip_match_text.restype = ctypes.c_int

_lib.clip_match_image.argtypes = [ctypes.c_void_p, ctypes.POINTER(ClipImage), ctypes.POINTER(ClipResultItem), ctypes.c_int]
_lib.clip_match_image.restype = ctypes.c_int

class ClipError(Exception):
    pass

def check_error(code: int) -> None:
    if code != 0:
        raise ClipError(f"CLIP API错误: {code}")

class Clip:
    def __init__(self, init_info: dict):
        self.handle = None
        self.init_info = ClipInit()
        
        # 设置初始化参数
        self.init_info.dev_type = init_info.get('dev_type', ClipDeviceType.axcl_device)
        self.init_info.devid = init_info.get('devid', 0)
        self.init_info.isCN = init_info.get('isCN', 1)
        
        # 设置路径
        for path_name in ['text_encoder_path', 'image_encoder_path', 'tokenizer_path', 'db_path']:
            if path_name in init_info:
                setattr(self.init_info, path_name, init_info[path_name].encode('utf-8'))
        
        # 创建CLIP实例
        handle = ctypes.c_void_p()
        check_error(_lib.clip_create(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.clip_destroy(self.handle)

    def add_image(self, key: str, image_data: np.ndarray) -> None:
        if self.contains_image(key):
            return
        image = ClipImage()
        image.data = ctypes.cast(image_data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))
        image.width = image_data.shape[1]
        image.height = image_data.shape[0]
        image.channels = image_data.shape[2]
        image.stride = image_data.shape[1] * image_data.shape[2]
        
        check_error(_lib.clip_add(self.handle, key.encode('utf-8'), ctypes.byref(image), 0))

    def remove_image(self, key: str) -> None:
        check_error(_lib.clip_remove(self.handle, key.encode('utf-8')))

    def contains_image(self, key: str) -> bool:
        return _lib.clip_contain(self.handle, key.encode('utf-8')) == 1

    def match_text(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        results = (ClipResultItem * top_k)()
        check_error(_lib.clip_match_text(self.handle, text.encode('utf-8'), results, top_k))
        
        return [(item.key.decode('utf-8'), item.score) for item in results]

    def match_image(self, image_data: bytes, width: int, height: int, channels: int = 3, top_k: int = 10) -> List[Tuple[str, float]]:
        image = ClipImage()
        image.data = ctypes.cast(ctypes.create_string_buffer(image_data), ctypes.POINTER(ctypes.c_ubyte))
        image.width = width
        image.height = height
        image.channels = channels
        image.stride = width * channels
        
        results = (ClipResultItem * top_k)()
        check_error(_lib.clip_match_image(self.handle, ctypes.byref(image), ctypes.byref(results), top_k))
        
        return [(item.key.decode('utf-8'), item.score) for item in results]

def enum_devices() -> dict:
    devices = ClipDevices()
    check_error(_lib.clip_enum_devices(ctypes.byref(devices)))
    
    return {
        'host': {
            'available': bool(devices.host.available),
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


def sys_init(dev_type: ClipDeviceType = ClipDeviceType.axcl_device, devid: int = 0) -> None:
    check_error(_lib.clip_sys_init(dev_type, devid))


def sys_deinit(dev_type: ClipDeviceType = ClipDeviceType.axcl_device, devid: int = 0) -> None:
    check_error(_lib.clip_sys_deinit(dev_type, devid))