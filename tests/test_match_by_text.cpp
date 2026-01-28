#include "clip.h"
#include "utils/cmdline.hpp"
#include "utils/timer.hpp"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <SimpleCV.hpp>
#include "utils/cqdm.h"
#include <filesystem>
#include <axcl.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <vector>

static std::string WStringToUtf8(const std::wstring &w)
{
    if (w.empty())
        return {};
    int size_needed = WideCharToMultiByte(
        CP_UTF8, 0,
        w.data(), (int)w.size(),
        nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0)
        throw std::runtime_error("WideCharToMultiByte failed");

    std::string utf8(size_needed, '\0');
    int written = WideCharToMultiByte(
        CP_UTF8, 0,
        w.data(), (int)w.size(),
        utf8.data(), size_needed,
        nullptr, nullptr);
    if (written != size_needed)
        throw std::runtime_error("WideCharToMultiByte write mismatch");
    return utf8;
}
#endif

// 把原 main 的内容挪到这里
static int app_main(int argc, char *argv[])
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

    // 这里拿到的已经是 UTF-8（Windows 下由 wmain 转过）
    std::string image_src = parser.get<std::string>("image");
    std::string text = parser.get<std::string>("text");

#if defined(_WIN32) || defined(_WIN64)
    // 如果你后续 std::filesystem / 你的 glob / imread 能吃 UTF-8，就用 u8path 更稳
    std::filesystem::path image_src_path = std::filesystem::u8path(image_src);
#else
    std::filesystem::path image_src_path(image_src);
#endif

    std::vector<std::string> image_paths = SimpleCV::glob(image_src + "/*.*");
    auto cqdm = create_cqdm(image_paths.size(), 32);
    for (size_t i = 0; i < image_paths.size(); i++)
    {
        std::string image_path = image_paths[i];
        std::filesystem::path img_p =
#if defined(_WIN32) || defined(_WIN64)
            std::filesystem::u8path(image_path);
#else
            std::filesystem::path(image_path);
#endif
        std::string image_name = img_p.filename().string();

        char key[CLIP_KEY_MAX_LEN];
        sprintf(key, "%s", image_name.c_str());
        if (clip_contain(handle, key))
        {
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
    }

    int topk = 10;
    std::vector<clip_result_item_t> results(topk);
    timer t;
    clip_match_text(handle, text.c_str(), results.data(), topk);
    printf("match text \"%s\" %6.2fms\n", text.c_str(), t.cost());
    printf("|%32s | %6s|\n", "key", "score");

    for (size_t i = 0; i < results.size(); i++)
    {
        std::filesystem::path key_path = image_src_path / results[i].key;
#if defined(_WIN32) || defined(_WIN64)
        // 如果你担心控制台显示乱码，可以打印 u8string()；但 printf 仍可能受控制台代码页影响
        auto out = key_path.u8string();
        printf("|%32s | %6.2f|\n", out.c_str(), results[i].score);
#else
        printf("|%32s | %6.2f|\n", key_path.string().c_str(), results[i].score);
#endif
    }

    clip_destroy(handle);
    axclFinalize();
    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
// Windows 入口：用 wmain 拿 UTF-16，再转 UTF-8 交给 app_main
int wmain(int argc, wchar_t *wargv[])
{
    std::vector<std::string> utf8_args;
    utf8_args.reserve(argc);
    for (int i = 0; i < argc; ++i)
    {
        utf8_args.push_back(WStringToUtf8(wargv[i]));
    }

    std::vector<char *> argv_utf8;
    argv_utf8.reserve(argc);
    for (int i = 0; i < argc; ++i)
    {
        argv_utf8.push_back(const_cast<char *>(utf8_args[i].c_str()));
    }

    return app_main(argc, argv_utf8.data());
}
#else
int main(int argc, char *argv[])
{
    return app_main(argc, argv);
}
#endif
