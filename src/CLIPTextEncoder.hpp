#pragma once
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <memory>
#include "clip.h"
#include "sample_log.h"
#include "tokenizer/tokenizer.hpp"
#include "CLIP.hpp" 

enum class CLIPType
{
    unknown = 0,
    jina_clip_v2,
    cn_clip,
    clip
};

class CLIPTextEncoder
{
protected:
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;

    // bool _isCN = false;
    int LEN_TEXT_FEATURE = 512;
    int LEN_TEXT_TOKEN = 77;
    
    int PAD_TOKEN = 0;

    std::map<CLIPType, std::pair<std::string, std::string>> bos_eos_map = {
        {CLIPType::jina_clip_v2, {"<s>", "</s>"}},
        {CLIPType::cn_clip, {"[CLS]", "[SEP]"}},
        {CLIPType::clip, {"<|startoftext|>", "<|endoftext|>"}}};

    CLIPType clip_type = CLIPType::unknown;
    std::string bos_str;
    std::string eos_str;

public:
    virtual bool load_text_encoder(clip_init_t *clip_init) = 0;
    virtual bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features) = 0;
    int get_text_feature_size()
    {
        return LEN_TEXT_FEATURE;
    }

    CLIPType get_clip_type()
    {
        return clip_type;
    }

    bool load_tokenizer(std::string tokenizer_path)
    {
        std::ifstream fs(tokenizer_path);
        if (!fs.good())
        {
            ALOGE("vocab file open failed %s", tokenizer_path.c_str());
            return false;
        }
        fs.close();

        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));

        for (auto &[key, value] : bos_eos_map)
        {
            if (tokenizer->is_special(tokenizer->encode(value.first)[0]) && tokenizer->is_special(tokenizer->encode(value.second)[0]))
            {
                clip_type = key;
                break;
            }
        }
        if (clip_type == CLIPType::unknown)
        {
            std::cout << "Unknown clip type" << std::endl;
            return false;
        }
        bos_str = bos_eos_map[clip_type].first;
        eos_str = bos_eos_map[clip_type].second;
        if (clip_type == CLIPType::jina_clip_v2)
        {
            PAD_TOKEN = 1;
        }
        printf("clip_type: %d, bos_str: %s, eos_str: %s\n", (int)clip_type, bos_str.c_str(), eos_str.c_str());
        return true;
    }
};
