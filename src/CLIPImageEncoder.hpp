#pragma once
// #include <opencv2/opencv.hpp>
#include <SimpleCV.hpp>
#include <memory>
#include "clip.h"
#include "sample_log.h"

// Preprocessing parameters for different models
struct ImagePreprocessParams
{
    float mean[3];
    float std[3];
};

// CLIP preprocessing (default)
static const ImagePreprocessParams CLIP_PREPROCESS = {
    {0.48145466f * 255.f, 0.4578275f * 255.f, 0.40821073f * 255.f},
    {1 / (0.26862954f * 255.f), 1 / (0.26130258f * 255.f), 1 / (0.27577711f * 255.f)}};

// SigLIP2 preprocessing: mean=0.5, std=0.5 for all channels
static const ImagePreprocessParams SIGLIP2_PREPROCESS = {
    {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f},
    {1 / (0.5f * 255.f), 1 / (0.5f * 255.f), 1 / (0.5f * 255.f)}};

class CLIPImageEncoder
{
protected:
    float _mean_val[3] = {0.48145466f * 255.f, 0.4578275f * 255.f, 0.40821073f * 255.f};
    float _std_val[3] = {1 / (0.26862954f * 255.f), 1 / (0.26130258f * 255.f), 1 / (0.27577711f * 255.f)};

    int LEN_IMAGE_FEATURE = 512;
    int input_height, input_width;

public:
    virtual bool load_image_encoder(clip_init_t *clip_init) = 0;
    virtual bool encode(SimpleCV::Mat image, std::vector<float> &image_features) = 0;
    virtual bool encode(clip_image_t *image, std::vector<float> &image_features) = 0;

    int get_image_feature_size()
    {
        return LEN_IMAGE_FEATURE;
    }

    // Set preprocessing parameters
    void set_preprocess_params(const ImagePreprocessParams &params)
    {
        memcpy(_mean_val, params.mean, sizeof(params.mean));
        memcpy(_std_val, params.std, sizeof(params.std));
    }
};
