#pragma once
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include "clip.h"
#include "sample_log.h"

#include "CLIPTextEncoderAX650.hpp"
#include "CLIPImageEncoderAX650.hpp"

class CLIP
{
protected:
    std::shared_ptr<CLIPTextEncoder> m_text_encoder;
    std::shared_ptr<CLIPImageEncoder> m_image_encoder;

    // SigLIP2 parameters from model
    float siglip2_logit_scale = 4.7244534f;
    float siglip2_logit_bias = -16.771725f;

    static inline void fast_softmax_row(const std::vector<float> &row, std::vector<float> &out)
    {
        float maxVal = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        out.resize(row.size());
        for (size_t i = 0; i < row.size(); ++i)
        {
            out[i] = std::exp(row[i] - maxVal);
            sum += out[i];
        }
        float invSum = 1.0f / sum;
        for (auto &val : out)
            val *= invSum;
    }
    static void softmax(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output)
    {
        output.clear();
        output.reserve(input.size());

        for (const auto &row : input)
        {
            std::vector<float> expRow;
            fast_softmax_row(row, expRow);

            output.push_back(std::move(expRow));
        }
    }

    // Sigmoid function for SigLIP2
    static inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Standard CLIP postprocess with softmax
    static void postprocess_clip(
        const std::vector<std::vector<float>> &imageFeatures,
        const std::vector<std::vector<float>> &textFeatures,
        std::vector<std::vector<float>> &logits_per_image,
        std::vector<std::vector<float>> &logits_per_text)
    {
        const float logit_scale = 100.0f;

        // Step 3: Compute logits_per_image = image @ text^T
        std::vector<std::vector<float>> logitsPerImage;
        logitsPerImage.reserve(imageFeatures.size());

        for (const auto &imgVec : imageFeatures)
        {
            std::vector<float> row;
            row.reserve(textFeatures.size());
            for (const auto &txtVec : textFeatures)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < imgVec.size(); ++i)
                    dot += imgVec[i] * txtVec[i];
                row.push_back(logit_scale * dot);
            }
            logitsPerImage.push_back(std::move(row));
        }

        // Step 4: Transpose logitsPerImage to get logitsPerText
        std::vector<std::vector<float>> logitsPerText(textFeatures.size(), std::vector<float>(imageFeatures.size()));
        for (size_t i = 0; i < logitsPerImage.size(); ++i)
            for (size_t j = 0; j < logitsPerImage[i].size(); ++j)
                logitsPerText[j][i] = logitsPerImage[i][j];

        // Step 5: Apply softmax
        softmax(logitsPerImage, logits_per_image); // shape (N_img, N_txt)
        softmax(logitsPerText, logits_per_text);   // shape (N_txt, N_img)
    }

    // SigLIP2 postprocess with sigmoid (no softmax)
    void postprocess_siglip2(
        const std::vector<std::vector<float>> &imageFeatures,
        const std::vector<std::vector<float>> &textFeatures,
        std::vector<std::vector<float>> &logits_per_image,
        std::vector<std::vector<float>> &logits_per_text)
    {
        // Compute logits_per_text = text @ image^T * exp(logit_scale) + logit_bias
        // Then apply sigmoid to get probabilities
        
        std::vector<std::vector<float>> logitsPerText;
        logitsPerText.reserve(textFeatures.size());

        for (const auto &txtVec : textFeatures)
        {
            std::vector<float> row;
            row.reserve(imageFeatures.size());
            for (const auto &imgVec : imageFeatures)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < txtVec.size(); ++i)
                    dot += txtVec[i] * imgVec[i];
                // Apply logit_scale and logit_bias
                float logit = dot * std::exp(siglip2_logit_scale) + siglip2_logit_bias;
                // Apply sigmoid to get probability
                row.push_back(sigmoid(logit));
            }
            logitsPerText.push_back(std::move(row));
        }

        // Transpose logitsPerText to get logitsPerImage
        std::vector<std::vector<float>> logitsPerImage(imageFeatures.size(), std::vector<float>(textFeatures.size()));
        for (size_t i = 0; i < logitsPerText.size(); ++i)
            for (size_t j = 0; j < logitsPerText[i].size(); ++j)
                logitsPerImage[j][i] = logitsPerText[i][j];

        logits_per_image = std::move(logitsPerImage);
        logits_per_text = std::move(logitsPerText);
    }

public:
    CLIP()
    {
    }

    int get_image_feature_size()
    {
        if (m_image_encoder == nullptr)
        {
            ALOGE("image encoder is null");
            return -1;
        }
        return m_image_encoder->get_image_feature_size();
    }

    int get_text_feature_size()
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return -1;
        }
        return m_text_encoder->get_text_feature_size();
    }

    bool load_tokenizer(std::string vocab_path, model_type_e model_type = model_type_unknown)
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return -1;
        }
        bool ret = m_text_encoder->load_tokenizer(vocab_path, model_type);
        if (!ret)
        {
            return false;
        }

        // Set image preprocessing parameters based on model type
        CLIPType clip_type = m_text_encoder->get_clip_type();
        if (clip_type == CLIPType::siglip2)
        {
            if (m_image_encoder != nullptr)
            {
                m_image_encoder->set_preprocess_params(SIGLIP2_PREPROCESS);
                ALOGI("SigLIP2: set image preprocessing parameters (mean=0.5, std=0.5)");
            }
        }

        return true;
    }

    bool load_text_encoder(clip_init_t *init_info)
    {
        if (m_text_encoder == nullptr)
        {
            m_text_encoder.reset(new CLIPTextEncoderAX650);
        }
        return m_text_encoder->load_text_encoder(init_info);
    }

    bool load_image_encoder(clip_init_t *init_info)
    {
        if (m_image_encoder == nullptr)
        {
            m_image_encoder.reset(new CLIPImageEncoderAX650);
        }
        return m_image_encoder->load_image_encoder(init_info);
    }

    bool encode(clip_image_t *image, std::vector<float> &image_features)
    {
        if (m_image_encoder == nullptr)
        {
            ALOGE("image encoder is null");
            return false;
        }
        auto ret = m_image_encoder->encode(image, image_features);
        return ret;
    }

    bool encode(SimpleCV::Mat image, std::vector<float> &image_features)
    {
        if (m_image_encoder == nullptr)
        {
            ALOGE("image encoder is null");
            return false;
        }
        auto ret = m_image_encoder->encode(image, image_features);
        return ret;
    }

    bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features)
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return false;
        }
        auto ret = m_text_encoder->encode(texts, text_features);
        return ret;
    }

    bool encode(std::string text, std::vector<float> &text_feature)
    {
        std::vector<std::vector<float>> text_features;
        std::vector<std::string> texts = {text};
        auto ret = encode(texts, text_features);
        if (ret)
        {
            text_feature = text_features[0];
        }
        return ret;
    }

    void decode(std::vector<std::vector<float>> &image_features, std::vector<std::vector<float>> &text_features,
                std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        CLIPType clip_type = m_text_encoder->get_clip_type();
        if (clip_type == CLIPType::siglip2)
        {
            postprocess_siglip2(image_features, text_features, logits_per_image, logits_per_text);
        }
        else
        {
            postprocess_clip(image_features, text_features, logits_per_image, logits_per_text);
        }
    }
};
