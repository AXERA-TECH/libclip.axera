#include "clip.h"
#include "utils/cmdline.hpp"
#include "utils/timer.hpp"
#include <fstream>
#include <cstring>
#include <SimpleCV.hpp>
#include "utils/cqdm.h"
#include <filesystem>
#include <axcl.h>

int main(int argc, char *argv[])
{
    axclInit(0);

    clip_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", true, "tests/cnclip_tokenizer.txt");
    parser.add<std::string>("db_path", 'd', "db path", false, "clip_feat_db");

    parser.add<std::string>("image", 'i', "image folder(jpg png etc....)", true);
    parser.add<std::string>("text", 't', "text or txt file", true);
    parser.parse_check(argc, argv);

    sprintf(init_info.image_encoder_path, "%s", parser.get<std::string>("ienc").c_str());
    sprintf(init_info.text_encoder_path, "%s", parser.get<std::string>("tenc").c_str());
    sprintf(init_info.tokenizer_path, "%s", parser.get<std::string>("vocab").c_str());
    sprintf(init_info.db_path, "%s", parser.get<std::string>("db_path").c_str());

    printf("image_encoder_path: %s\n", init_info.image_encoder_path);
    printf("text_encoder_path: %s\n", init_info.text_encoder_path);
    printf("tokenizer_path: %s\n", init_info.tokenizer_path);
    printf("db_path: %s\n", init_info.db_path);

    init_info.devid = 0;

    clip_handle_t handle;
    int ret = clip_create(&init_info, &handle);
    if (ret != clip_errcode_success)
    {
        printf("clip_create failed\n");
        axclFinalize();
        return -1;
    }

    std::string image_src = parser.get<std::string>("image");
    std::string text = parser.get<std::string>("text");

    std::vector<std::string> image_paths = SimpleCV::glob(image_src + "/*.*");
    auto cqdm = create_cqdm(image_paths.size(), 32);
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

        SimpleCV::Mat src = SimpleCV::imread(image_path, SimpleCV::ColorSpace::RGB);
        clip_image_t image;
        image.data = src.data;
        image.width = src.width;
        image.height = src.height;
        image.channels = src.channels;
        image.stride = src.step;

        timer t;
        clip_add(handle, key, &image, 0);
        update_cqdm(&cqdm, i, "count", "get image embeding");
        // printf("add image %s  %04ld/%04ld  %6.2fms\n", image_name.c_str(), i, image_paths.size(), t.cost());
    }
    int topk = 10;
    std::vector<clip_result_item_t> results(topk);
    timer t;
    clip_match_text(handle, text.c_str(), results.data(), topk);
    printf("match text \"%s\" %6.2fms\n", text.c_str(), t.cost());
    printf("|%32s | %6s|\n", "key", "score");
    std::filesystem::path image_src_path(image_src);
    for (size_t i = 0; i < results.size(); i++)
    {
        std::filesystem::path key_path = image_src_path / results[i].key;
        printf("|%32s | %6.2f|\n", key_path.string().c_str(), results[i].score);
    }

    clip_destroy(handle);

    axclFinalize();

    return 0;
}