#pragma once
#include "CLIPImageEncoder.hpp"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "runner/axcl/ax_model_runner_axcl.hpp"
#include "mmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

class CLIPImageEncoderFGCLIP : public CLIPImageEncoder
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;

    int patch_size_ = 16;
    int patch_dim_ = 768;
    int max_num_patches_ = 256;

    static const ax_runner_tensor_t &find_input(ax_runner_base *encoder, const std::string &name, int index)
    {
        try
        {
            return encoder->get_input(name);
        }
        catch (const std::exception &)
        {
            return encoder->get_input(index);
        }
    }

    static int determine_max_num_patches(int width, int height)
    {
        const int max_val = (width / 16) * (height / 16);
        if (max_val > 784)
            return 1024;
        if (max_val > 576)
            return 784;
        if (max_val > 256)
            return 576;
        if (max_val > 128)
            return 256;
        return 128;
    }

    static int scaled_image_size(double scale, int size, int patch_size)
    {
        int scaled = static_cast<int>(std::ceil(size * scale / patch_size)) * patch_size;
        return std::max(patch_size, scaled);
    }

    static std::pair<int, int> resize_size_for_max_patches(int height, int width, int patch_size, int max_num_patches)
    {
        constexpr double eps = 1e-5;
        double scale_min = eps / 10.0;
        double scale_max = 100.0;

        while ((scale_max - scale_min) >= eps)
        {
            const double scale = (scale_min + scale_max) / 2.0;
            const int target_height = scaled_image_size(scale, height, patch_size);
            const int target_width = scaled_image_size(scale, width, patch_size);
            const int num_patches = (target_height / patch_size) * (target_width / patch_size);

            if (num_patches <= max_num_patches)
                scale_min = scale;
            else
                scale_max = scale;
        }

        return {scaled_image_size(scale_min, height, patch_size),
                scaled_image_size(scale_min, width, patch_size)};
    }

    bool encode_rgb(SimpleCV::Mat image, std::vector<float> &image_features)
    {
        if (!m_encoder)
        {
            ALOGE("encoder not init");
            return false;
        }

        auto &pixel_values = find_input(m_encoder.get(), "pixel_values", 0);
        auto &pixel_attention_mask = find_input(m_encoder.get(), "pixel_attention_mask", 1);
        if (pixel_values.pVirAddr == nullptr || pixel_attention_mask.pVirAddr == nullptr)
        {
            ALOGE("FG-CLIP2 image encoder inputs are not ready");
            return false;
        }

        // The AX model has a fixed maximum patch dimension. Resize according
        // to AXERA's reference processor, then zero-pad to the model shape.
        const int effective_num_patches = std::min(max_num_patches_,
                                                   determine_max_num_patches(image.width, image.height));
        auto target_size = resize_size_for_max_patches(image.height, image.width, patch_size_, effective_num_patches);
        const int target_height = target_size.first;
        const int target_width = target_size.second;

        SimpleCV::Mat resized;
        SimpleCV::resize(image, resized, target_width, target_height);
        if (resized.empty())
        {
            ALOGE("failed to resize image for FG-CLIP2");
            return false;
        }

        const int patches_height = target_height / patch_size_;
        const int patches_width = target_width / patch_size_;
        const int valid_patches = std::min(effective_num_patches, patches_height * patches_width);

        const size_t pixel_bytes = static_cast<size_t>(max_num_patches_) * patch_dim_ * sizeof(float);
        std::memset(pixel_values.pVirAddr, 0, std::min(pixel_bytes, static_cast<size_t>(pixel_values.nSize)));

        auto *mask = static_cast<int32_t *>(pixel_attention_mask.pVirAddr);
        const size_t mask_bytes = static_cast<size_t>(max_num_patches_) * sizeof(int32_t);
        std::memset(mask, 0, std::min(mask_bytes, static_cast<size_t>(pixel_attention_mask.nSize)));
        std::fill(mask, mask + valid_patches, 1);

        auto *patch_data = static_cast<float *>(pixel_values.pVirAddr);
        for (int patch_y = 0; patch_y < patches_height; ++patch_y)
        {
            for (int patch_x = 0; patch_x < patches_width; ++patch_x)
            {
                const int patch_index = patch_y * patches_width + patch_x;
                if (patch_index >= max_num_patches_)
                    break;
                float *dst = patch_data + static_cast<size_t>(patch_index) * patch_dim_;

                for (int y = 0; y < patch_size_; ++y)
                {
                    const unsigned char *row = resized.data + (patch_y * patch_size_ + y) * resized.step +
                                               patch_x * patch_size_ * resized.channels;
                    for (int x = 0; x < patch_size_; ++x)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            const int dst_index = (y * patch_size_ + x) * 3 + c;
                            dst[dst_index] = (static_cast<float>(row[x * resized.channels + c]) - _mean_val[c]) * _std_val[c];
                        }
                    }
                }
            }
        }

        if (m_encoder->inference() != 0)
        {
            ALOGE("FG-CLIP2 image encoder inference failed");
            return false;
        }

        image_features.resize(LEN_IMAGE_FEATURE);
        std::memcpy(image_features.data(), m_encoder->get_output(0).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));

        float norm = 0.0f;
        for (float value : image_features)
            norm += value * value;
        norm = std::sqrt(norm);
        if (norm > 0.0f)
        {
            for (float &value : image_features)
                value /= norm;
        }
        return true;
    }

public:
    bool load_image_encoder(clip_init_t *init_info) override
    {
        MMap image_mmap(init_info->image_encoder_path);

        if (init_info->dev_type == ax_devive_e::host_device)
        {
            m_encoder = std::make_shared<ax_runner_ax650>();
            if (m_encoder->init(image_mmap.data(), image_mmap.size(), -1) != 0)
            {
                ALOGE("FG-CLIP2 image encoder init failed");
                return false;
            }
        }
        else if (init_info->dev_type == ax_devive_e::axcl_device)
        {
            m_encoder = std::make_shared<ax_runner_axcl>();
            if (m_encoder->init(image_mmap.data(), image_mmap.size(), init_info->devid) != 0)
            {
                ALOGE("FG-CLIP2 image encoder init failed");
                return false;
            }
        }
        else
        {
            return false;
        }

        if (m_encoder->get_num_inputs() < 2)
        {
            ALOGE("FG-CLIP2 image encoder requires pixel_values and pixel_attention_mask inputs");
            return false;
        }

        const auto &pixel_values = find_input(m_encoder.get(), "pixel_values", 0);
        const auto &pixel_attention_mask = find_input(m_encoder.get(), "pixel_attention_mask", 1);

        if (!pixel_values.vShape.empty() && pixel_values.vShape.back() > 0)
            patch_dim_ = pixel_values.vShape.back();
        if (patch_dim_ != 3 * patch_size_ * patch_size_)
        {
            ALOGE("unsupported FG-CLIP2 patch dimension: %d", patch_dim_);
            return false;
        }

        if (pixel_values.vShape.size() >= 3 && pixel_values.vShape[1] > 0)
            max_num_patches_ = pixel_values.vShape[1];
        else if (pixel_values.nSize > 0)
            max_num_patches_ = pixel_values.nSize / (patch_dim_ * sizeof(float));

        if (max_num_patches_ <= 0)
        {
            ALOGE("invalid FG-CLIP2 max patch count");
            return false;
        }
        if (pixel_values.nSize < static_cast<int>(max_num_patches_ * patch_dim_ * sizeof(float)) ||
            pixel_attention_mask.nSize < static_cast<int>(max_num_patches_ * sizeof(int32_t)))
        {
            ALOGE("FG-CLIP2 image encoder input buffers are too small");
            return false;
        }
        if (pixel_attention_mask.vShape.size() >= 2 && pixel_attention_mask.vShape[1] > 0 &&
            pixel_attention_mask.vShape[1] != max_num_patches_)
        {
            ALOGE("FG-CLIP2 patch mask shape mismatch: %d != %d", pixel_attention_mask.vShape[1], max_num_patches_);
            return false;
        }

        const auto &output = m_encoder->get_output(0);
        LEN_IMAGE_FEATURE = !output.vShape.empty() ? output.vShape.back() : output.nSize / sizeof(float);
        ALOGI("FG-CLIP2 image encoder: max patches=%d, patch dim=%d, feature len=%d",
              max_num_patches_, patch_dim_, LEN_IMAGE_FEATURE);
        return LEN_IMAGE_FEATURE > 0;
    }

    bool encode(clip_image_t *image, std::vector<float> &image_features) override
    {
        SimpleCV::Mat cv_image(image->height, image->width, image->channels, image->data, image->stride);
        SimpleCV::Mat cv_image_input;
        switch (image->channels)
        {
        case 4:
            cv_image_input = SimpleCV::cvtColor(cv_image, SimpleCV::ColorSpace::BGR, SimpleCV::ColorSpace::BGRA);
            break;
        case 1:
            cv_image_input = SimpleCV::cvtColor(cv_image, SimpleCV::ColorSpace::BGR, SimpleCV::ColorSpace::GRAY);
            break;
        case 3:
            cv_image_input = cv_image;
            break;
        default:
            ALOGE("only support channel 1,3,4 uint8 image");
            return false;
        }
        return encode_rgb(cv_image_input, image_features);
    }

    bool encode(SimpleCV::Mat image, std::vector<float> &image_features) override
    {
        SimpleCV::Mat cv_image_input;
        switch (image.channels)
        {
        case 4:
            cv_image_input = SimpleCV::cvtColor(image, SimpleCV::ColorSpace::BGR, SimpleCV::ColorSpace::BGRA);
            break;
        case 1:
            cv_image_input = SimpleCV::cvtColor(image, SimpleCV::ColorSpace::BGR, SimpleCV::ColorSpace::GRAY);
            break;
        case 3:
            cv_image_input = image;
            break;
        default:
            ALOGE("only support channel 1,3,4 uint8 image");
            return false;
        }
        return encode_rgb(cv_image_input, image_features);
    }
};
