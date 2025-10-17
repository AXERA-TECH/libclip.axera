#include "../src/tokenizer/tokenizer.hpp"
#include "../utils/cmdline.hpp"

std::map<std::string, std::pair<std::string, std::string>> bos_eos_map = {
    {"jina_clip_v2", {"<s>", "</s>"}},
    {"cn_clip", {"[CLS]", "[SEP]"}},
    {"clip", {"<|startoftext|>", "<|endoftext|>"}}
};

int main(int argc, char *argv[])
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