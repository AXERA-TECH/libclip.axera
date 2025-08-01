import os
from pyclip import Clip, enum_devices, sys_init, sys_deinit, ClipDeviceType
import cv2
import glob
import numpy as np
import tqdm

# 枚举设备
print("可用设备:", enum_devices())

# 初始化系统
sys_init(ClipDeviceType.axcl_device, 0)

try:
    # 创建CLIP实例
    clip = Clip({
        'text_encoder_path': 'cnclip/cnclip_vit_l14_336px_text_u16.axmodel',
        'image_encoder_path': 'cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel',
        'tokenizer_path': 'cnclip/cn_vocab.txt',
        'db_path': 'clip_feat_db_v2',
        'isCN': 1
    })

    # 添加图像
    image_files = glob.glob('coco_1000/*.jpg')
    for image_file in tqdm.tqdm(image_files):
        img = cv2.imread(image_file)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        filename = os.path.basename(image_file)
        clip.add_image(filename, img)

    # 文本匹配
    results = clip.match_text('dog', top_k=10)
    print("匹配结果:", results)

finally:
    # 反初始化系统
    sys_deinit(ClipDeviceType.axcl_device, 0)