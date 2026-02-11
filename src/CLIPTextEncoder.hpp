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

// Forward declaration
enum class CLIPType;

// Map from C API model_type_e to C++ CLIPType
inline int model_type_to_clip_type(model_type_e type)
{
    switch (type)
    {
        case model_type_clip:
            return 3; // CLIPType::clip
        case model_type_cn_clip:
            return 2; // CLIPType::cn_clip
        case model_type_jina_clip_v2:
            return 1; // CLIPType::jina_clip_v2
        case model_type_siglip2:
            return 4; // CLIPType::siglip2
        default:
            return 0; // CLIPType::unknown
    }
}

enum class CLIPType
{
    unknown = 0,
    jina_clip_v2,
    cn_clip,
    clip,
    siglip2
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
        {CLIPType::clip, {"<|startoftext|>", "<|endoftext|>"}},
        {CLIPType::siglip2, {"", "<eos>"}}};

    CLIPType clip_type = CLIPType::unknown;
    std::string bos_str;
    std::string eos_str;

    void setup_clip_params()
    {
        bos_str = bos_eos_map[clip_type].first;
        eos_str = bos_eos_map[clip_type].second;
        
        if (clip_type == CLIPType::jina_clip_v2)
        {
            PAD_TOKEN = 1;
        }
        else if (clip_type == CLIPType::siglip2)
        {
            PAD_TOKEN = 0; // <pad> token id is 0 for SigLIP2
        }
        else
        {
            PAD_TOKEN = 0; // default
        }
        
        ALOGI("clip_type: %d, bos_str: '%s', eos_str: '%s', PAD_TOKEN: %d", 
              (int)clip_type, bos_str.c_str(), eos_str.c_str(), PAD_TOKEN);
    }

    bool detect_clip_type()
    {
        for (auto &[key, value] : bos_eos_map)
        {
            // Handle empty bos_str (SigLIP2 case)
            bool bos_check = true;
            if (!value.first.empty())
            {
                auto bos_ids = tokenizer->encode(value.first);
                if (bos_ids.empty() || !tokenizer->is_special(bos_ids[0]))
                {
                    bos_check = false;
                }
            }
            
            bool eos_check = false;
            auto eos_ids = tokenizer->encode(value.second);
            if (!eos_ids.empty() && tokenizer->is_special(eos_ids[0]))
            {
                eos_check = true;
            }
            
            if (bos_check && eos_check)
            {
                clip_type = key;
                break;
            }
        }
        
        return clip_type != CLIPType::unknown;
    }

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

    // Load tokenizer with explicit model type
    // If model_type is model_type_unknown, will try auto-detect
    bool load_tokenizer(std::string tokenizer_path, model_type_e model_type = model_type_unknown)
    {
        std::ifstream fs(tokenizer_path);
        if (!fs.good())
        {
            ALOGE("vocab file open failed %s", tokenizer_path.c_str());
            return false;
        }
        fs.close();

        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));

        // Use explicit model type if provided, otherwise auto-detect
        if (model_type != model_type_unknown)
        {
            int type_val = model_type_to_clip_type(model_type);
            if (type_val == 0)
            {
                ALOGE("Invalid model_type: %d", (int)model_type);
                return false;
            }
            clip_type = (CLIPType)type_val;
            ALOGI("Using explicit model_type: %d -> clip_type: %d", (int)model_type, type_val);
        }
        else
        {
            // Auto-detect from tokenizer
            if (!detect_clip_type())
            {
                ALOGE("Failed to auto-detect clip type from tokenizer");
                return false;
            }
            ALOGI("Auto-detected clip_type: %d", (int)clip_type);
        }

        setup_clip_params();
        return true;
    }
};
