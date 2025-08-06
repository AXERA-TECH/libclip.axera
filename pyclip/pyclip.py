import ctypes
import os
from typing import List, Tuple
import numpy as np
import platform
from pyaxdev import _lib, AxDeviceType, AxDevices, check_error


class ClipInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
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


class Clip:
    def __init__(self, init_info: dict):
        self.handle = None
        self.init_info = ClipInit()
        
        # 设置初始化参数
        self.init_info.dev_type = init_info.get('dev_type', AxDeviceType.axcl_device)
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