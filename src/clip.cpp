#include "clip.h"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "CLIP.hpp"

#include "leveldb/db.h"
#include "leveldb/options.h"

#include <queue>
#include <cstring>
#include <fstream>
#include <memory>

struct clip_internal_handle_t
{
    CLIP m_clip;
    std::vector<std::string> m_keys;
    std::vector<std::vector<float>> m_image_features;

    leveldb::DB *m_db;
    leveldb::Options m_options;
    leveldb::WriteOptions m_write_options;
    leveldb::ReadOptions m_read_options;
};

int clip_create(clip_init_t *init_info, clip_handle_t *_handle)
{
    if (!axcl_Dev_IsInit(init_info->devid))
    {
        int ret = axcl_Dev_Init(init_info->devid);
        if (ret != clip_errcode_success)
        {
            printf("axcl_Dev_Init failed\n");
            return clip_errcode_failed;
        }
    }

    clip_internal_handle_t *handle = new clip_internal_handle_t;
    auto ret = handle->m_clip.load_image_encoder(init_info);
    if (!ret)
    {
        printf("load image encoder failed\n");
        delete handle;
        return clip_errcode_create_failed_ienc;
    }
    ret = handle->m_clip.load_text_encoder(init_info);
    if (!ret)
    {
        printf("load text encoder failed\n");
        delete handle;
        return clip_errcode_create_failed_tenc;
    }
    ret = handle->m_clip.load_tokenizer(init_info->tokenizer_path);
    if (!ret)
    {
        printf("load tokenizer failed\n");
        delete handle;
        return clip_errcode_create_failed_vocab;
    }

    handle->m_options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(handle->m_options, init_info->db_path, &handle->m_db);
    if (!status.ok())
    {
        printf("open db failed, status: %s\n", status.ToString().c_str());
        delete handle;
        return clip_errcode_create_failed_db;
    }

    auto it = handle->m_db->NewIterator(handle->m_read_options);
    for (it->SeekToFirst(); it->Valid(); it->Next())
    {
        handle->m_keys.push_back(it->key().ToString());
        std::vector<float> image_features;
        image_features.resize(it->value().size() / sizeof(float));
        memcpy(image_features.data(), it->value().data(), it->value().size());
        handle->m_image_features.push_back(image_features);
        // printf("key: %s, value size: %ld\n", it->key().ToString().c_str(), it->value().size());
    }
    *_handle = handle;
    return clip_errcode_success;
}

int clip_destroy(clip_handle_t handle)
{
    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle)
    {
        delete internal_handle;
    }
    return clip_errcode_success;
}

int clip_add(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN], clip_image_t *image, char overwrite)
{
    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }

    if (!overwrite)
    {
        for (int i = 0; i < internal_handle->m_keys.size(); i++)
        {
            if (strcmp(internal_handle->m_keys[i].c_str(), key) == 0)
            {
                printf("key already exists\n");
                return clip_errcode_add_failed_key_exist;
            }
        }
    }

    std::vector<float> image_features;
    auto ret = internal_handle->m_clip.encode(image, image_features);
    if (!ret)
    {
        printf("encode image failed\n");
        return clip_errcode_add_failed_encode_image;
    }

    internal_handle->m_keys.push_back(key);
    internal_handle->m_image_features.push_back(image_features);
    leveldb::Slice key_slice(key);
    leveldb::Slice value_slice((char *)image_features.data(), image_features.size() * sizeof(float));
    leveldb::Status status = internal_handle->m_db->Put(internal_handle->m_write_options, key_slice, value_slice);
    if (!status.ok())
    {
        printf("put db failed, status: %s\n", status.ToString().c_str());
        return clip_errcode_add_failed_push_db;
    }
    return clip_errcode_success;
}

int clip_remove(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN])
{
    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }
    int index = -1;
    for (int i = 0; i < internal_handle->m_keys.size(); i++)
    {
        if (strcmp(internal_handle->m_keys[i].c_str(), key) == 0)
        {
            index = i;
            break;
        }
    }
    if (index == -1)
    {
        printf("key not found\n");
        return clip_errcode_remove_failed_key_not_exist;
    }
    internal_handle->m_keys.erase(internal_handle->m_keys.begin() + index);
    internal_handle->m_image_features.erase(internal_handle->m_image_features.begin() + index);
    leveldb::Slice key_slice(key);
    leveldb::Status status = internal_handle->m_db->Delete(internal_handle->m_write_options, key_slice);
    if (!status.ok())
    {
        printf("delete db failed, status: %s\n", status.ToString().c_str());
        return clip_errcode_remove_failed_del_db;
    }
    return clip_errcode_success;
}

int clip_contain(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN])
{
    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }
    for (int i = 0; i < internal_handle->m_keys.size(); i++)
    {
        if (strcmp(internal_handle->m_keys[i].c_str(), key) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int clip_get_text_feat(clip_handle_t handle, const char *text, clip_feature_item_t *feature)
{
    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }
    std::vector<std::vector<float>> text_features;
    std::vector<std::string> texts = {text};
    auto ret = internal_handle->m_clip.encode(texts, text_features);
    if (!ret)
    {
        printf("encode text failed\n");
        return clip_errcode_match_failed_encode_text;
    }
    if (text_features.size() != 1)
    {
        printf("encode text failed, text_features size: %ld\n", text_features.size());
        return clip_errcode_match_failed_encode_text;
    }
    if (text_features[0].size() > CLIP_TEXT_FEAT_MAX_LEN)
    {
        printf("encode text failed, text_features size: %ld > %d\n", text_features[0].size(), CLIP_TEXT_FEAT_MAX_LEN);
        return clip_errcode_match_failed_encode_text;
    }

    memcpy(feature->feat, text_features[0].data(), text_features[0].size() * sizeof(float));
    feature->len = text_features[0].size();
    return clip_errcode_success;
}

// 结构体保存 index 和 score 用于比较
struct ScoreIndex
{
    int index;
    float score;
    bool operator<(const ScoreIndex &other) const
    {
        return score > other.score; // 小顶堆，保留 top_k 最大的
    }
};

void get_top_k_results(const std::vector<float> &scores,
                       const std::vector<std::string> &m_keys,
                       clip_result_item_t *results,
                       int top_k)
{
    if (scores.size() != m_keys.size() || top_k <= 0 || results == nullptr)
        return;

    int n = scores.size();
    top_k = std::min(top_k, n);

    // 用小顶堆维护 top_k
    std::priority_queue<ScoreIndex> min_heap;
    for (int i = 0; i < n; ++i)
    {
        if ((int)min_heap.size() < top_k)
        {
            min_heap.push({i, scores[i]});
        }
        else if (scores[i] > min_heap.top().score)
        {
            min_heap.pop();
            min_heap.push({i, scores[i]});
        }
    }

    // 将结果从堆中取出并存入数组（顺序按分数从高到低）
    std::vector<ScoreIndex> top_results;
    while (!min_heap.empty())
    {
        top_results.push_back(min_heap.top());
        min_heap.pop();
    }
    std::sort(top_results.begin(), top_results.end(), [](const ScoreIndex &a, const ScoreIndex &b)
              { return a.score > b.score; });

    for (int i = 0; i < top_k; ++i)
    {
        int idx = top_results[i].index;
        std::strncpy(results[i].key, m_keys[idx].c_str(), CLIP_KEY_MAX_LEN - 1);
        results[i].key[CLIP_KEY_MAX_LEN - 1] = '\0'; // 确保 null 结尾
        results[i].score = scores[idx];
    }
}

int clip_match_feat(clip_handle_t handle, clip_feature_item_t *feature, clip_result_item_t *results, int top_k)
{
    std::vector<std::vector<float>> text_features(1);
    text_features[0].resize(feature->len);
    memcpy(text_features[0].data(), feature->feat, feature->len * sizeof(float));

    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }

    std::vector<std::vector<float>> logits_per_image;
    std::vector<std::vector<float>> logits_per_text;
    internal_handle->m_clip.decode(internal_handle->m_image_features, text_features, logits_per_image, logits_per_text);

    std::vector<float> &scores = logits_per_text[0];

    get_top_k_results(scores, internal_handle->m_keys, results, top_k);

    return clip_errcode_success;
}

int clip_match_text(clip_handle_t handle, const char *text, clip_result_item_t *results, int top_k)
{
    clip_feature_item_t feature = {0};
    if (clip_get_text_feat(handle, text, &feature) != clip_errcode_success)
    {
        printf("get text feature failed\n");
        return clip_errcode_match_failed_encode_text;
    }
    return clip_match_feat(handle, &feature, results, top_k);
}

int clip_match_image(clip_handle_t handle, clip_image_t *image, clip_result_item_t *results, int top_k)
{

    clip_internal_handle_t *internal_handle = (clip_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        printf("handle is null\n");
        return clip_errcode_invalid_ptr;
    }
    std::vector<float> image_features;
    auto ret = internal_handle->m_clip.encode(image, image_features);
    if (!ret)
    {
        printf("encode image failed\n");
        return clip_errcode_match_failed_encode_image;
    }

    std::vector<float> scores;
    for (auto &feat : internal_handle->m_image_features)
    {
        float similarity = 0.0;
        for (int i = 0; i < image_features.size(); i++)
        {
            similarity += image_features[i] * feat[i];
        }
        similarity = similarity < 0 ? 0 : similarity > 1 ? 1
                                                         : similarity;
        scores.push_back(similarity);
    }

    get_top_k_results(scores, internal_handle->m_keys, results, top_k);

    return clip_errcode_success;
}
