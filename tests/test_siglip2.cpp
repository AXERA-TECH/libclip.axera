#include "clip.h"
#include <cstring>
#include <SimpleCV.hpp>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }

    if (!ax_devices.host.available && ax_devices.devices.count == 0)
    {
        printf("no device available\n");
        return -1;
    }

    clip_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    // SigLIP2 model paths
    sprintf(init_info.image_encoder_path, "siglip2-base-patch16-224/ax650/siglip2-base-patch16-224_vision.axmodel");
    sprintf(init_info.text_encoder_path, "siglip2-base-patch16-224/ax650/siglip2-base-patch16-224_text.axmodel");
    sprintf(init_info.tokenizer_path, "../tests/tokenizer/siglip2_tokenizer.txt");
    sprintf(init_info.db_path, "siglip2_feat_db");
    init_info.model_type = model_type_siglip2;  // Explicitly set model type

    printf("image_encoder_path: %s\n", init_info.image_encoder_path);
    printf("text_encoder_path: %s\n", init_info.text_encoder_path);
    printf("tokenizer_path: %s\n", init_info.tokenizer_path);
    printf("db_path: %s\n", init_info.db_path);

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    clip_handle_t handle;
    int ret = clip_create(&init_info, &handle);
    if (ret != clip_errcode_success)
    {
        printf("clip_create failed, ret=%d\n", ret);
        return -1;
    }
    printf("clip_create success!\n");

    // Test image
    std::string image_path = "siglip2-base-patch16-224/000000039769.jpg";
    SimpleCV::Mat src = SimpleCV::imread(image_path, SimpleCV::ColorSpace::RGB);
    if (src.data == nullptr)
    {
        printf("Failed to load image: %s\n", image_path.c_str());
        clip_destroy(handle);
        return -1;
    }
    printf("Loaded image: %s (%dx%d)\n", image_path.c_str(), src.width, src.height);

    clip_image_t image;
    image.data = src.data;
    image.width = src.width;
    image.height = src.height;
    image.channels = src.channels;
    image.stride = src.step;

    // Add image to database
    char key[CLIP_KEY_MAX_LEN] = "test_cat.jpg";
    ret = clip_add(handle, key, &image, 1);
    if (ret != clip_errcode_success)
    {
        printf("clip_add failed, ret=%d\n", ret);
        clip_destroy(handle);
        return -1;
    }
    printf("clip_add success!\n");

    // Test text matching
    const char *texts[] = {
        "a photo of 2 cats",
        "a photo of 2 dogs",
        "a photo of a cat",
        "a photo of a dog"
    };
    int num_texts = sizeof(texts) / sizeof(texts[0]);

    printf("\nTesting text matching:\n");
    printf("| %-30s | %6s |\n", "text", "score");
    printf("|--------------------------------|--------|\n");

    for (int i = 0; i < num_texts; i++)
    {
        clip_result_item_t results[1];
        ret = clip_match_text(handle, texts[i], results, 1);
        if (ret != clip_errcode_success)
        {
            printf("clip_match_text failed for \"%s\", ret=%d\n", texts[i], ret);
            continue;
        }
        printf("| %-30s | %6.4f |\n", texts[i], results[0].score);
    }

    // Test feature extraction
    printf("\nTesting feature extraction:\n");
    clip_feature_item_t text_feat;
    ret = clip_get_text_feat(handle, "a photo of 2 cats", &text_feat);
    if (ret == clip_errcode_success)
    {
        printf("Text feature length: %d\n", text_feat.len);
        printf("First 5 values: ");
        for (int i = 0; i < 5 && i < text_feat.len; i++)
        {
            printf("%.4f ", text_feat.feat[i]);
        }
        printf("...\n");
    }

    clip_destroy(handle);
    
    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    printf("\nSigLIP2 test completed!\n");
    return 0;
}
