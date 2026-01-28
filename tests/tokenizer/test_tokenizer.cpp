#include "../src/tokenizer/tokenizer.hpp"
#include "../utils/cmdline.hpp"



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


std::map<std::string, std::pair<std::string, std::string>> bos_eos_map = {
    {"jina_clip_v2", {"<s>", "</s>"}},
    {"cn_clip", {"[CLS]", "[SEP]"}},
    {"clip", {"<|startoftext|>", "<|endoftext|>"}}
};

int app_main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.add<std::string>("text", 0, "text", true);
    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));

    std::string clip_type;
    for (auto& [key, value] : bos_eos_map)
    {
        if (tokenizer->is_special(tokenizer->encode(value.first)[0]) && tokenizer->is_special(tokenizer->encode(value.second)[0]))
        {
            clip_type = key;
            break;
        }
    }
    if (clip_type.empty())
    {
        std::cout << "Unknown clip type" << std::endl;
        return -1;
    }
    printf("clip_type: %s\n", clip_type.c_str());
    std::string bos_str = bos_eos_map[clip_type].first;
    std::string eos_str = bos_eos_map[clip_type].second;
    std::string prompt = bos_str + a.get<std::string>("text") + eos_str;


    printf("prompt: %s\n", prompt.c_str());
    auto ids = tokenizer->encode(prompt);
    printf("ids size: %ld\n", ids.size());
    for (auto id : ids)
    {
        std::cout << id << ", ";
    }
    std::cout << std::endl;

    std::string text;
    for (auto id : ids)
    {
        text += tokenizer->decode(id);
    }
    std::cout << "text: " << text << std::endl;

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