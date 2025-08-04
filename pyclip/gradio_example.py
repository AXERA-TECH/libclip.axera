import os
import base64
import gradio as gr
from pyclip import Clip, enum_devices, sys_init, sys_deinit, ClipDeviceType
import cv2
import glob
import numpy as np
from io import BytesIO
from PIL import Image
import tqdm

# 初始化
print("可用设备:", enum_devices())
sys_init(ClipDeviceType.axcl_device, 0)

clip = Clip({
    'text_encoder_path': 'cnclip/cnclip_vit_l14_336px_text_u16.axmodel',
    'image_encoder_path': 'cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel',
    'tokenizer_path': 'cnclip/cn_vocab.txt',
    'db_path': 'clip_feat_db_coco',
    'isCN': 1
})

image_folder = "coco_1000"

# 加载图片数据库（只做一次）
image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
for image_file in tqdm.tqdm(image_files):
    filename = os.path.basename(image_file)
    if clip.contains_image(filename) == 1:
        continue
    img = cv2.imread(image_file)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    clip.add_image(filename, img)

# 工具函数：图片转 base64
def img_to_pil(img_path):
    return Image.open(img_path).convert("RGB")

# 主搜索函数
def search_images(query, top_k):
    results = clip.match_text(query, top_k=top_k)
    images = []
    for filename, score in results:
        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):
            img = img_to_pil(img_path)
            images.append((img, f"{filename}  Score: {score:.4f}"))
    return images


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 文搜图 Demo")

    with gr.Row():
        query_input = gr.Textbox(label="请输入文本查询")
        topk_input = gr.Number(value=25, precision=0, label="Top-K")
    search_btn = gr.Button("搜图")

    gallery = gr.Gallery(label="匹配结果", show_label=True, columns=4)

    search_btn.click(fn=search_images, inputs=[query_input, topk_input], outputs=gallery)

# 启动
ip = "0.0.0.0"
demo.launch(server_name=ip, server_port=7860)

# 关闭系统（你可加信号处理来自动关闭）
import atexit
atexit.register(lambda: sys_deinit(ClipDeviceType.axcl_device, 0))
