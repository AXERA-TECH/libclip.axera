#include "clip.h"
#include "utils/cmdline.hpp"
#include "utils/timer.hpp"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <SimpleCV.hpp>
#include "utils/cqdm.h"
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

static void overwrite_argv_utf8_on_windows(int &argc, char **&argv)
{
    int wargc = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv || wargc <= 0)
        return;

    // Keep storage alive for the lifetime of main().
    static std::vector<std::string> args_u8;
    static std::vector<char *> argv_u8;
    args_u8.clear();
    argv_u8.clear();
    args_u8.reserve(wargc);
    argv_u8.reserve(wargc + 1);

    for (int i = 0; i < wargc; ++i)
    {
        const wchar_t *ws = wargv[i];
        const int need = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
        std::string s;
        if (need > 0)
        {
            // `need` includes the trailing null, so allocate `need` bytes then drop it.
            s.resize(static_cast<size_t>(need));
            WideCharToMultiByte(CP_UTF8, 0, ws, -1, s.data(), need, nullptr, nullptr);
            if (!s.empty() && s.back() == '\0')
            {
                s.pop_back();
            }
        }
        args_u8.emplace_back(std::move(s));
    }
    LocalFree(wargv);

    for (auto &s : args_u8)
        argv_u8.push_back(const_cast<char *>(s.c_str()));
    argv_u8.push_back(nullptr);

    argc = wargc;
    argv = argv_u8.data();
}
#endif

int main(int argc, char *argv[])
{
#ifdef _WIN32
    // cmd.exe passes non-ASCII args in the system ANSI code page. Convert from the
    // Unicode command line to UTF-8 so the tokenizer sees correct UTF-8 text.
    overwrite_argv_utf8_on_windows(argc, argv);
#endif

    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }

    if (!ax_devices.host.available && ax_devices.devices.count == 0)
    {
        printf("no device available\n");
        return -1;
    }

    clip_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", true, "tests/cnclip_tokenizer.txt");
    parser.add<std::string>("db_path", 'd', "db path", false, "clip_feat_db");
    parser.add<int>("model_type", 'm', "model type (0=unknown, 1=clip, 2=cn_clip, 3=jina_clip_v2, 4=siglip2)", false, 0);

    parser.add<std::string>("image", 'i', "image folder(jpg png etc....)", true);
    parser.add<std::string>("text", 't', "text or txt file", true);
    parser.parse_check(argc, argv);

    sprintf(init_info.image_encoder_path, "%s", parser.get<std::string>("ienc").c_str());
    sprintf(init_info.text_encoder_path, "%s", parser.get<std::string>("tenc").c_str());
    sprintf(init_info.tokenizer_path, "%s", parser.get<std::string>("vocab").c_str());
    sprintf(init_info.db_path, "%s", parser.get<std::string>("db_path").c_str());
    init_info.model_type = (model_type_e)parser.get<int>("model_type");

    printf("image_encoder_path: %s\n", init_info.image_encoder_path);
    printf("text_encoder_path: %s\n", init_info.text_encoder_path);
    printf("tokenizer_path: %s\n", init_info.tokenizer_path);
    printf("db_path: %s\n", init_info.db_path);

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
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

    std::vector<std::string> image_paths = SimpleCV::glob(image_src + "/*.*");
    auto cqdm = create_cqdm(image_paths.size(), 32);
    for (size_t i = 0; i < image_paths.size(); i++)
    {
        std::string image_path = image_paths[i];
        // Keep keys stable across platforms (Windows paths use '\\').
        std::string image_name = std::filesystem::path(image_path).filename().string();
        char key[CLIP_KEY_MAX_LEN];
        std::snprintf(key, sizeof(key), "%s", image_name.c_str());
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

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}
