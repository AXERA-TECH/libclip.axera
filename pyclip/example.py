import os
from pyclip import Clip, enum_devices, sys_init, sys_deinit, ClipDeviceType
import cv2
import glob
import numpy as np
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ienc', type=str, default='cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel')
    parser.add_argument('--tenc', type=str, default='cnclip/cnclip_vit_l14_336px_text_u16.axmodel')
    parser.add_argument('--vocab', type=str, default='cnclip/cn_vocab.txt')
    parser.add_argument('--isCN', type=int, default=1)
    parser.add_argument('--db_path', type=str, default='clip_feat_db_coco')
    parser.add_argument('--image_folder', type=str, default='coco_1000')
    args = parser.parse_args()

    image_folder = args.image_folder

    # 枚举设备
    print("可用设备:", enum_devices())

    # 初始化系统
    sys_init(ClipDeviceType.axcl_device, 0)

    try:
        # 创建CLIP实例
        clip = Clip({
            'text_encoder_path': args.tenc,
            'image_encoder_path': args.ienc,
            'tokenizer_path': args.vocab,
            'db_path': args.db_path,
            'isCN': args.isCN
        })


        # 添加图像
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
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