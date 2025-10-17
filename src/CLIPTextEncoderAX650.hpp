#pragma once
#include "CLIPTextEncoder.hpp"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "runner/axcl/ax_model_runner_axcl.hpp"
#include "mmap.hpp"

#include <math.h>

class CLIPTextEncoderAX650 : public CLIPTextEncoder
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;

public:
    bool load_text_encoder(clip_init_t *init_info) override
    {
        MMap text_mmap(init_info->text_encoder_path);

        if (init_info->dev_type == ax_devive_e::host_device)
        {

            m_encoder = std::make_shared<ax_runner_ax650>();
            auto ret = m_encoder->init(text_mmap.data(), text_mmap.size(), -1);
            if (ret != 0)
            {
                printf("text encoder init failed\n");

                return false;
            }
        }
        else if (init_info->dev_type == ax_devive_e::axcl_device)
        {
            m_encoder = std::make_shared<ax_runner_axcl>();
            auto ret = m_encoder->init(text_mmap.data(), text_mmap.size(), init_info->devid);
            if (ret != 0)
            {
                printf("text encoder init failed\n");

                return false;
            }
        }
        LEN_TEXT_TOKEN = m_encoder->get_input(0).vShape[m_encoder->get_input(0).vShape.size() - 1];
        LEN_TEXT_FEATURE = m_encoder->get_output(0).vShape[m_encoder->get_output(0).vShape.size() - 1];
        ALOGI("text token len %d, text feature len %d", LEN_TEXT_TOKEN, LEN_TEXT_FEATURE);
        return true;
    }

    template <typename T>
    void fill_ids(T *data, int len, std::vector<int> &text_token, int pad_token = 0)
    {
        memset(data, 0, len * sizeof(T));
        for (int i = 0; i < len; i++)
        {
            if (i < (int)text_token.size())
            {
                data[i] = text_token[i];
            }
            else
            {
                data[i] = pad_token;
            }
        }
    }

    bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features) override
    {
        if (m_encoder == nullptr)
        {
            return false;
        }
        text_features.resize(texts.size());
        for (size_t i = 0; i < texts.size(); i++)
        {
            auto text = bos_str + texts[i] + eos_str;
            std::vector<int> text_token = tokenizer->encode(text);
            if (text_token.size() > LEN_TEXT_TOKEN)
            {
                ALOGW("the text of \"%s\" token bigger than %d\n", texts[i].c_str(), LEN_TEXT_TOKEN);
                return false;
            }

            fill_ids((int32_t *)m_encoder->get_input(0).pVirAddr, LEN_TEXT_TOKEN, text_token, PAD_TOKEN);
            m_encoder->inference();
            text_features[i].resize(LEN_TEXT_FEATURE);
            // m_encoder->mem_sync_output(0);
            memcpy(text_features[i].data(), m_encoder->get_output(0).pVirAddr, LEN_TEXT_FEATURE * sizeof(float));

            auto &text_feat = text_features[i];
            float norm = 0.0f;
            for (float v : text_feat)
                norm += v * v;
            norm = std::sqrt(norm);
            for (float &v : text_feat)
                v /= norm;
        }

        return true;
    }
};
