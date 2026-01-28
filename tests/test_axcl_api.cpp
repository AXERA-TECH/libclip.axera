#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include <fstream>
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <vector>

static std::string WStringToUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    int size_needed = WideCharToMultiByte(
        CP_UTF8, 0,
        w.data(), (int)w.size(),
        nullptr, 0, nullptr, nullptr
    );
    if (size_needed <= 0) throw std::runtime_error("WideCharToMultiByte failed");

    std::string utf8(size_needed, '\0');
    int written = WideCharToMultiByte(
        CP_UTF8, 0,
        w.data(), (int)w.size(),
        utf8.data(), size_needed,
        nullptr, nullptr
    );
    if (written != size_needed) throw std::runtime_error("WideCharToMultiByte write mismatch");
    return utf8;
}
#endif

int app_main(int argc, char *argv[])
{
    auto ret = axcl_Init();
    if (ret != 0)
    {
        printf("axclInit failed\n");
        return -1;
    }

    axcl_Dev_Init(0);

    ax_runner_axcl runner;
    std::ifstream file("cnclip/cnclip_vit_l14_336px_text_u16.axmodel", std::ios::binary);
    if (!file.is_open())
    {
        printf("open file failed\n");
        return -1;
    }
    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    runner.init(model_data.data(), model_data.size(), 0);

    runner.deinit();

    axcl_Dev_Exit(0);
    axcl_Finalize();
    return 0;
}


#if defined(_WIN32) || defined(_WIN64)
// Windows 入口：用 wmain 拿 UTF-16，再转 UTF-8 交给 app_main
int wmain(int argc, wchar_t *wargv[])
{
    std::vector<std::string> utf8_args;
    utf8_args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        utf8_args.push_back(WStringToUtf8(wargv[i]));
    }

    std::vector<char*> argv_utf8;
    argv_utf8.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        argv_utf8.push_back(const_cast<char*>(utf8_args[i].c_str()));
    }

    return app_main(argc, argv_utf8.data());
}
#else
int main(int argc, char *argv[])
{
    return app_main(argc, argv);
}
#endif