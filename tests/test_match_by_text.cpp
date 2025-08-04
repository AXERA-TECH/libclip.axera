#include "clip.h"
#include "cmdline.hpp"
#include "timer.hpp"
#include <fstream>
#include <cstring>
#include <opencv2/opencv.hpp>

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
    parser.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", true, "cnclip/cn_vocab.txt");
    parser.add<int>("language", 'l', "language choose, 0:english 1:chinese", false, 1);
    parser.add<std::string>("db_path", 'd', "db path", false, "clip_feat_db");

    parser.add<std::string>("image", 'i', "image folder(jpg png etc....)", true);
    parser.add<std::string>("text", 't', "text or txt file", true);
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

    clip_handle_t handle;
    int ret = clip_create(&init_info, &handle);
    if (ret != clip_errcode_success)
    {
        printf("clip_create failed\n");
        return -1;
    }

    std::string image_src = parser.get<std::string>("image");
    std::string text = parser.get<std::string>("text");

    std::vector<std::string> image_paths;
    cv::glob(image_src + "/*.*", image_paths);

    for (size_t i = 0; i < image_paths.size(); i++)
    {
        std::string image_path = image_paths[i];
        std::string image_name = image_path.substr(image_path.find_last_of("/") + 1);
        char key[CLIP_KEY_MAX_LEN];
        sprintf(key, "%s", image_name.c_str());
        if (clip_contain(handle, key))
        {
            // printf("%s is exist %04ld/%04ld\n", key, i, image_paths.size());
            continue;
        }

        cv::Mat src = cv::imread(image_path);
        cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
        clip_image_t image;
        image.data = src.data;
        image.width = src.cols;
        image.height = src.rows;
        image.channels = src.channels();
        image.stride = src.step;

        timer t;
        clip_add(handle, key, &image, 0);
        // printf("add image %s  %04ld/%04ld  %6.2fms\n", image_name.c_str(), i, image_paths.size(), t.cost());
    }
    int topk = 10;
    std::vector<clip_result_item_t> results(topk);
    timer t;
    clip_match_text(handle, text.c_str(), results.data(), topk);
    printf("match text \"%s\" %6.2fms\n", text.c_str(), t.cost());
    printf("|%32s | %6s|\n", "key", "score");
    for (size_t i = 0; i < results.size(); i++)
    {
        printf("|%32s | %6.2f|\n", results[i].key, results[i].score);
    }

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