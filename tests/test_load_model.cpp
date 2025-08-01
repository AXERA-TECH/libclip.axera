#include "clip.h"
#include "cmdline.hpp"
#include <fstream>
#include <cstring>

int main(int argc, char *argv[])
{
    clip_devices_t clip_devices;
    memset(&clip_devices, 0, sizeof(clip_devices_t));
    if (clip_enum_devices(&clip_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (clip_devices.host.available)
    {
        clip_sys_init(host_device, -1);
    }
    else if (clip_devices.devices.count > 0)
    {
        clip_sys_init(axcl_device, 0);
    }
    else
    {
        printf("no device available\n");
        return -1;
    }

    clip_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", false, "/home/axera/CLIP-ONNX-AX650-CPP/build/cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", false, "/home/axera/CLIP-ONNX-AX650-CPP/build/cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", false, "/home/axera/CLIP-ONNX-AX650-CPP/cn_vocab.txt");
    parser.add<int>("language", 'l', "language choose, 0:english 1:chinese", false, 1);
    parser.add<std::string>("db_path", 'd', "db path", false, "");
    parser.parse_check(argc, argv);

    sprintf(init_info.image_encoder_path, "%s", parser.get<std::string>("ienc").c_str());
    sprintf(init_info.text_encoder_path, "%s", parser.get<std::string>("tenc").c_str());
    sprintf(init_info.tokenizer_path, "%s", parser.get<std::string>("vocab").c_str());
    init_info.isCN = parser.get<int>("language");
    sprintf(init_info.db_path, "%s", parser.get<std::string>("db_path").c_str());

    printf("image_encoder_path: %s\n", init_info.image_encoder_path);
    printf("text_encoder_path: %s\n", init_info.text_encoder_path);
    printf("tokenizer_path: %s\n", init_info.tokenizer_path);
    printf("isCN: %d\n", init_info.isCN);
    printf("db_path: %s\n", init_info.db_path);

    if (clip_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (clip_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    auto handle = clip_create(&init_info);

    clip_destroy(handle);

    if (clip_devices.host.available)
    {
        clip_sys_deinit(host_device, -1);
    }
    else if (clip_devices.devices.count > 0)
    {

        clip_sys_deinit(axcl_device, 0);
    }

    return 0;
}