import os
import gradio as gr
import requests
from pyclip import Clip
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
from PIL import Image
import tqdm
import argparse
import subprocess
import re

class Translator:
    def __init__(self, host, port):
        self.url = f"http://{host}:{port}/translate"
    
    def translate(self, text, target_lang='target_chs'):
        try:
            body = {'input': text, 'target': target_lang}
            response = requests.post(self.url, json=body)
            return response.json()['output']
        except:
            return text



def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ienc', type=str, default='cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel')
    parser.add_argument('--tenc', type=str, default='cnclip/cnclip_vit_l14_336px_text_u16.axmodel')
    parser.add_argument('--vocab', type=str, default='cnclip/cn_vocab.txt')
    parser.add_argument('--db_path', type=str, default='clip_feat_db_coco')
    parser.add_argument('--image_folder', type=str, default='coco_1000')
    parser.add_argument('--dev_type', type=str, default='host', choices=['host', 'axcl'])
    parser.add_argument('--host', help='host ip for translate server', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='port number for translate server', type=int, default=12345)
    args = parser.parse_args()

    image_folder = args.image_folder
    
    translator = Translator(args.host, args.port)

    # 初始化
    dev_type = AxDeviceType.unknown_device
    dev_id = -1
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
        dev_type = AxDeviceType.host_device
        dev_id = -1
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
        dev_type = AxDeviceType.axcl_device
        dev_id = 0
    else:
        raise Exception("No available device")

    clip = Clip({
        'text_encoder_path': args.tenc,
        'image_encoder_path': args.ienc,
        'tokenizer_path': args.vocab,
        'db_path': args.db_path,
        'dev_type': dev_type,
        'devid': dev_id,
    })


    # 加载图片数据库（只做一次）
    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    for image_file in tqdm.tqdm(image_files):
        filename = os.path.basename(image_file)
        if clip.contains_image(filename) == 1:
            continue
        img = cv2.imread(image_file)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        clip.add_image(filename, img)

    def img_to_pil(img_path):
        return Image.open(img_path).convert("RGB")

    # 主搜索函数
    def search_images(query, check_translate, top_k):
        if check_translate:
            query_translated = translator.translate(query)
        else:
            query_translated = query
        results = clip.match_text(query_translated, top_k=top_k)
        images = []
        for filename, score in results:
            img_path = os.path.join(image_folder, filename)
            if os.path.exists(img_path):
                img = img_to_pil(img_path)
                images.append((img, f"{filename}  Score: {score:.4f}"))
        return query_translated, images


    # Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 文搜图 Demo")

        with gr.Row():
            query_input = gr.Textbox(label="请输入文本查询")
            with gr.Column():
                check_translate = gr.Checkbox(label="是否翻译查询", value=False)
                query_translated = gr.Textbox(label="翻译后的查询")
            topk_input = gr.Number(value=25, precision=0, label="Top-K")
        search_btn = gr.Button("搜图")

        gallery = gr.Gallery(label="匹配结果", show_label=True, columns=4)

        search_btn.click(fn=search_images, inputs=[query_input, check_translate, topk_input], outputs=[query_translated, gallery])

    # 启动
    ips = get_all_local_ips()
    for ip in ips:
        print(f"* Running on local URL:  http://{ip}:7860")
    ip = "0.0.0.0"
    demo.launch(server_name=ip, server_port=7860)

    import atexit
    if devices_info['host']['available']:
        atexit.register(lambda: sys_deinit(AxDeviceType.host_device, -1))
    elif devices_info['devices']['count'] > 0:
        atexit.register(lambda: sys_deinit(AxDeviceType.axcl_device, 0))
    
    
