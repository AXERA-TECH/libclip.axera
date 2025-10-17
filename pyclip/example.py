import os
from pyclip import Clip
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ienc', type=str, default='cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel')
    parser.add_argument('--tenc', type=str, default='cnclip/cnclip_vit_l14_336px_text_u16.axmodel')
    parser.add_argument('--vocab', type=str, default='cnclip/cn_vocab.txt')
    parser.add_argument('--db_path', type=str, default='clip_feat_db_coco')
    parser.add_argument('--image_folder', type=str, default='coco_1000')
    args = parser.parse_args()

    image_folder = args.image_folder

    # 枚举设备
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
    else:
        raise Exception("No available device")

    try:
        # 创建CLIP实例
        clip = Clip({
            'text_encoder_path': args.tenc,
            'image_encoder_path': args.ienc,
            'tokenizer_path': args.vocab,
            'db_path': args.db_path
        })


        # 添加图像
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        for image_file in tqdm.tqdm(image_files):
            filename = os.path.basename(image_file)
            if clip.contains_image(filename) == 1:
                continue
            img = cv2.imread(image_file)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            clip.add_image(filename, img)

        # 文本匹配
        results = clip.match_text('dog', top_k=10)
        print("匹配结果:", results)
        
        # 特征匹配
        feat = clip.get_text_feat('dog')
        results = clip.match_feat(feat, top_k=10)
        print("匹配结果:", results)

    finally:
        # 反初始化系统
        if devices_info['host']['available']:
            sys_deinit(AxDeviceType.host_device, -1)
        elif devices_info['devices']['count'] > 0:
            sys_deinit(AxDeviceType.axcl_device, 0)