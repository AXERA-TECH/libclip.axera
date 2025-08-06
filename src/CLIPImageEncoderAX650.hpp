#pragma once
#include "CLIPImageEncoder.hpp"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "runner/axcl/ax_model_runner_axcl.hpp"
#include "mmap.hpp"

class CLIPImageEncoderAX650 : public CLIPImageEncoder
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;
    cv::Mat input;

    bool nchw;

public:
    bool load_image_encoder(clip_init_t *init_info) override
    {
        // m_encoder.reset(new ax_runner_ax650);
        // m_encoder->init(encoder_path.c_str());

        MMap image_mmap(init_info->image_encoder_path);

        if (init_info->dev_type == ax_devive_e::host_device)
        {

            m_encoder = std::make_shared<ax_runner_ax650>();
            auto ret = m_encoder->init(image_mmap.data(), image_mmap.size(), -1);
            if (ret != 0)
            {
                printf("text encoder init failed\n");

                return false;
            }
        }
        else if (init_info->dev_type == ax_devive_e::axcl_device)
        {
            m_encoder = std::make_shared<ax_runner_axcl>();
            auto ret = m_encoder->init(image_mmap.data(), image_mmap.size(), init_info->devid);
            if (ret != 0)
            {
                printf("text encoder init failed\n");
                return false;
            }
        }
        nchw = m_encoder->get_input(0).vShape[1] == 3;
        if (nchw)
        {
            input_height = m_encoder->get_input(0).vShape[2];
            input_width = m_encoder->get_input(0).vShape[3];
            ALOGI("nchw %d %d", input_height, input_width);
        }
        else // nhwc
        {
            input_height = m_encoder->get_input(0).vShape[1];
            input_width = m_encoder->get_input(0).vShape[2];
            ALOGI("nhwc %d %d", input_height, input_width);
        }

        LEN_IMAGE_FEATURE = m_encoder->get_output(0).vShape[1];
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);
        image_features_input = std::vector<float>(1024 * LEN_IMAGE_FEATURE);
        return true;
    }

    bool encode(clip_image_t *image, std::vector<float> &image_features) override
    {
        cv::Mat cv_image(image->height, image->width, CV_8UC(image->channels), image->data, image->stride);
        // cv::imwrite("debug.jpg", cv_image);
        cv::Mat cv_image_input;
        switch (image->channels)
        {
        case 4:
            cv::cvtColor(cv_image, cv_image_input, cv::COLOR_BGRA2BGR);
            break;
        case 1:
            cv::cvtColor(cv_image, cv_image_input, cv::COLOR_GRAY2BGR);
            break;
        case 3:
            cv_image_input = cv_image;
            break;
        default:
            ALOGE("only support channel 1,3,4 uint8 image");
            return false;
        }
        return encode(cv_image, image_features);
    }

    bool encode(cv::Mat image, std::vector<float> &image_features) override
    {
        if (!m_encoder.get())
        {
            ALOGE("encoder not init");
            return false;
        }
        cv::resize(image, input, cv::Size(input_width, input_height));
        // cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        if (nchw)
        {
            float *inputPtr = (float *)m_encoder->get_input(0).pVirAddr;

            uchar *img_data = input.data;

            int letterbox_cols = input_width;
            int letterbox_rows = input_height;
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < letterbox_rows; h++)
                {
                    for (int w = 0; w < letterbox_cols; w++)
                    {
                        int in_index = h * letterbox_cols * 3 + w * 3 + c;
                        int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                        inputPtr[out_index] = (float(img_data[in_index]) - _mean_val[c]) * _std_val[c];
                    }
                }
            }
        }
        else
        {
            unsigned char *inputPtr = (unsigned char *)m_encoder->get_input(0).pVirAddr;
            memcpy(inputPtr, input.data, input.cols * input.rows * 3);
        }

        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        // m_encoder->mem_sync_output(0);
        memcpy(image_features.data(), m_encoder->get_output(0).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));

        float norm = 0.0f;
        for (float v : image_features)
            norm += v * v;
        norm = std::sqrt(norm);
        for (float &v : image_features)
            v /= norm;

        return true;
    }
};
