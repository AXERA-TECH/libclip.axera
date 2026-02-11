#include "CLIP.hpp"
#include "ax_devices.h"
#include <cstring>
#include <SimpleCV.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

// 打印相似度矩阵表格
void print_similarity_table(
    const std::vector<std::string> &texts,
    const std::vector<std::string> &image_names,
    const std::vector<std::vector<float>> &similarities)
{
    size_t num_texts = texts.size();
    size_t num_images = image_names.size();
    
    if (num_texts == 0 || num_images == 0)
    {
        std::cout << "No texts or images to compare" << std::endl;
        return;
    }

    // 计算列宽
    size_t name_col_width = 20;
    for (const auto &name : image_names)
    {
        name_col_width = std::max(name_col_width, name.length() + 2);
    }

    // 打印表头
    std::cout << "\n" << std::string(name_col_width + num_texts * 12 + 1, '=') << std::endl;
    std::cout << std::setw(name_col_width) << std::left << "Image \\ Text";
    for (const auto &text : texts)
    {
        // 截断过长的文本
        std::string short_text = text.length() > 10 ? text.substr(0, 10) + "..." : text;
        std::cout << " | " << std::setw(9) << short_text;
    }
    std::cout << std::endl;
    std::cout << std::string(name_col_width + num_texts * 12 + 1, '-') << std::endl;

    // 打印每一行
    for (size_t i = 0; i < num_images; i++)
    {
        std::string short_name = image_names[i].length() > name_col_width - 2 
            ? image_names[i].substr(0, name_col_width - 5) + "..." 
            : image_names[i];
        std::cout << std::setw(name_col_width) << std::left << short_name;
        
        for (size_t j = 0; j < num_texts; j++)
        {
            float score = similarities[j][i];  // similarities[text_idx][image_idx]
            // 根据得分设置颜色（高得分用绿色，低得分用红色）
            const char *color = "\033[0m";  // reset
            if (score > 0.3) color = "\033[32m";      // green
            else if (score > 0.1) color = "\033[33m"; // yellow
            else color = "\033[31m";                  // red
            
            std::cout << " | " << color << std::fixed << std::setprecision(4) 
                      << std::setw(9) << score << "\033[0m";
        }
        std::cout << std::endl;
    }
    std::cout << std::string(name_col_width + num_texts * 12 + 1, '=') << std::endl;
}

// 打印纯文本相似度（不带颜色）
void print_similarity_table_plain(
    const std::vector<std::string> &texts,
    const std::vector<std::string> &image_names,
    const std::vector<std::vector<float>> &similarities)
{
    size_t num_texts = texts.size();
    size_t num_images = image_names.size();
    
    if (num_texts == 0 || num_images == 0)
    {
        std::cout << "No texts or images to compare" << std::endl;
        return;
    }

    size_t name_col_width = 20;
    for (const auto &name : image_names)
    {
        name_col_width = std::max(name_col_width, name.length() + 2);
    }

    std::cout << "\n" << std::string(name_col_width + num_texts * 12 + 1, '=') << std::endl;
    std::cout << std::setw(name_col_width) << std::left << "Image \\ Text";
    for (const auto &text : texts)
    {
        std::string short_text = text.length() > 10 ? text.substr(0, 10) + "..." : text;
        std::cout << " | " << std::setw(9) << short_text;
    }
    std::cout << std::endl;
    std::cout << std::string(name_col_width + num_texts * 12 + 1, '-') << std::endl;

    for (size_t i = 0; i < num_images; i++)
    {
        std::string short_name = image_names[i].length() > name_col_width - 2 
            ? image_names[i].substr(0, name_col_width - 5) + "..." 
            : image_names[i];
        std::cout << std::setw(name_col_width) << std::left << short_name;
        
        for (size_t j = 0; j < num_texts; j++)
        {
            float score = similarities[j][i];
            std::cout << " | " << std::fixed << std::setprecision(4) 
                      << std::setw(9) << score;
        }
        std::cout << std::endl;
    }
    std::cout << std::string(name_col_width + num_texts * 12 + 1, '=') << std::endl;
}

// 查找最佳匹配
void print_best_matches(
    const std::vector<std::string> &texts,
    const std::vector<std::string> &image_names,
    const std::vector<std::vector<float>> &similarities)
{
    std::cout << "\n=== Best Matches ===" << std::endl;
    
    // 每个文本的最佳匹配图片
    std::cout << "\nFor each text, best matching image:" << std::endl;
    for (size_t j = 0; j < texts.size(); j++)
    {
        float best_score = -1;
        size_t best_idx = 0;
        for (size_t i = 0; i < image_names.size(); i++)
        {
            if (similarities[j][i] > best_score)
            {
                best_score = similarities[j][i];
                best_idx = i;
            }
        }
        std::cout << "\"" << texts[j] << "\" -> " << image_names[best_idx] 
                  << " (score: " << std::fixed << std::setprecision(4) << best_score << ")" << std::endl;
    }

    // 每张图片的最佳匹配文本
    std::cout << "\nFor each image, best matching text:" << std::endl;
    for (size_t i = 0; i < image_names.size(); i++)
    {
        float best_score = -1;
        size_t best_idx = 0;
        for (size_t j = 0; j < texts.size(); j++)
        {
            if (similarities[j][i] > best_score)
            {
                best_score = similarities[j][i];
                best_idx = j;
            }
        }
        std::cout << image_names[i] << " -> \"" << texts[best_idx] 
                  << "\" (score: " << std::fixed << std::setprecision(4) << best_score << ")" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // 检查参数
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model_type> <image1,image2,...> <text1,text2,...> [ienc] [tenc] [vocab]" << std::endl;
        std::cout << "  model_type: 1=clip, 2=cn_clip, 3=jina_clip_v2, 4=siglip2" << std::endl;
        std::cout << "  images: comma-separated image paths" << std::endl;
        std::cout << "  texts: comma-separated text queries (use \"\" for texts with spaces)" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  # SigLIP2 with multiple images and texts" << std::endl;
        std::cout << "  " << argv[0] << " 4 \"img1.jpg,img2.jpg\" \"a cat,a dog\"" << std::endl;
        std::cout << std::endl;
        std::cout << "  # Use custom model paths" << std::endl;
        std::cout << "  " << argv[0] << " 4 \"img1.jpg\" \"hello\" vision.axmodel text.axmodel tokenizer.txt" << std::endl;
        return 0;
    }

    // 解析参数
    model_type_e model_type = (model_type_e)std::stoi(argv[1]);
    
    // 解析图片路径
    std::vector<std::string> image_paths;
    std::string images_arg = argv[2];
    size_t pos = 0;
    while ((pos = images_arg.find(',')) != std::string::npos)
    {
        image_paths.push_back(images_arg.substr(0, pos));
        images_arg.erase(0, pos + 1);
    }
    image_paths.push_back(images_arg);

    // 解析文本
    std::vector<std::string> texts;
    std::string texts_arg = argv[3];
    while ((pos = texts_arg.find(',')) != std::string::npos)
    {
        texts.push_back(texts_arg.substr(0, pos));
        texts_arg.erase(0, pos + 1);
    }
    texts.push_back(texts_arg);

    // 模型路径
    std::string ienc_path = (argc > 4) ? argv[4] : "siglip2-base-patch16-224/ax650/siglip2-base-patch16-224_vision.axmodel";
    std::string tenc_path = (argc > 5) ? argv[5] : "siglip2-base-patch16-224/ax650/siglip2-base-patch16-224_text.axmodel";
    std::string vocab_path = (argc > 6) ? argv[6] : "../tests/tokenizer/siglip2_tokenizer.txt";

    std::cout << "=== CLIP Direct Inference Demo ===" << std::endl;
    std::cout << "Model type: " << model_type << std::endl;
    std::cout << "Images: " << image_paths.size() << std::endl;
    for (const auto &p : image_paths) std::cout << "  - " << p << std::endl;
    std::cout << "Texts: " << texts.size() << std::endl;
    for (const auto &t : texts) std::cout << "  - \"" << t << "\"" << std::endl;

    // 初始化设备
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        std::cerr << "enum devices failed" << std::endl;
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
    else
    {
        std::cerr << "no device available" << std::endl;
        return -1;
    }

    // 创建 CLIP 实例
    CLIP clip;
    
    // 加载模型
    clip_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));
    sprintf(init_info.image_encoder_path, "%s", ienc_path.c_str());
    sprintf(init_info.text_encoder_path, "%s", tenc_path.c_str());
    sprintf(init_info.tokenizer_path, "%s", vocab_path.c_str());
    init_info.model_type = model_type;
    
    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    std::cout << "\nLoading models..." << std::endl;
    std::cout << "  Image encoder: " << ienc_path << std::endl;
    std::cout << "  Text encoder: " << tenc_path << std::endl;
    
    if (!clip.load_image_encoder(&init_info))
    {
        std::cerr << "Failed to load image encoder" << std::endl;
        return -1;
    }
    
    if (!clip.load_text_encoder(&init_info))
    {
        std::cerr << "Failed to load text encoder" << std::endl;
        return -1;
    }
    
    if (!clip.load_tokenizer(vocab_path, model_type))
    {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return -1;
    }

    std::cout << "Models loaded successfully!" << std::endl;
    std::cout << "  Image feature size: " << clip.get_image_feature_size() << std::endl;
    std::cout << "  Text feature size: " << clip.get_text_feature_size() << std::endl;

    // 编码图片
    std::cout << "\nEncoding images..." << std::endl;
    std::vector<std::vector<float>> image_features;
    std::vector<std::string> image_names;
    
    for (const auto &path : image_paths)
    {
        SimpleCV::Mat src = SimpleCV::imread(path, SimpleCV::ColorSpace::RGB);
        if (src.data == nullptr)
        {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }

        std::vector<float> feat;
        if (clip.encode(src, feat))
        {
            image_features.push_back(feat);
            // 提取文件名
            size_t last_slash = path.find_last_of("/\\");
            image_names.push_back((last_slash != std::string::npos) ? path.substr(last_slash + 1) : path);
            std::cout << "  Encoded: " << path << " -> feature size " << feat.size() << std::endl;
        }
        else
        {
            std::cerr << "Failed to encode image: " << path << std::endl;
        }
    }

    if (image_features.empty())
    {
        std::cerr << "No images successfully encoded" << std::endl;
        return -1;
    }

    // 编码文本
    std::cout << "\nEncoding texts..." << std::endl;
    std::vector<std::vector<float>> text_features;
    
    for (const auto &text : texts)
    {
        std::vector<float> feat;
        if (clip.encode(text, feat))
        {
            text_features.push_back(feat);
            std::cout << "  Encoded: \"" << text << "\" -> feature size " << feat.size() << std::endl;
        }
        else
        {
            std::cerr << "Failed to encode text: " << text << std::endl;
        }
    }

    if (text_features.empty())
    {
        std::cerr << "No texts successfully encoded" << std::endl;
        return -1;
    }

    // 计算相似度
    std::cout << "\nComputing similarities..." << std::endl;
    std::vector<std::vector<float>> logits_per_image;
    std::vector<std::vector<float>> logits_per_text;
    
    clip.decode(image_features, text_features, logits_per_image, logits_per_text);

    // logits_per_text[text_idx][image_idx] = similarity
    print_similarity_table(texts, image_names, logits_per_text);
    print_best_matches(texts, image_names, logits_per_text);

    // 清理
    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
