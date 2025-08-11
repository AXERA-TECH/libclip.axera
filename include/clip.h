#ifndef __CLIP_H__
#define __CLIP_H__

#if defined(__cplusplus)
extern "C"
{
#endif
#include "ax_devices.h"
#define CLIP_DEVICES_COUNT 16
#define CLIP_VERSION_LEN 32
#define CLIP_KEY_MAX_LEN 64
#define CLIP_PATH_LEN 128
#define CLIP_TEXT_FEAT_MAX_LEN 768

    typedef enum
    {
        clip_errcode_failed = -1,
        clip_errcode_success = 0,

        clip_errcode_invalid_ptr,
        // clip_errcode_sysinit_failed,
        // clip_errcode_sysdeinit_failed,
        // clip_errcode_axcl_sysinit_failed,
        // clip_errcode_axcl_sysdeinit_failed,

        clip_errcode_create_failed = 0x10000,
        clip_errcode_create_failed_sys,
        clip_errcode_create_failed_ienc,
        clip_errcode_create_failed_tenc,
        clip_errcode_create_failed_vocab,
        clip_errcode_create_failed_db,

        clip_errcode_destroy_failed = 0x20000,

        clip_errcode_add_failed = 0x30000,
        clip_errcode_add_failed_key_exist,
        clip_errcode_add_failed_encode_image,
        clip_errcode_add_failed_push_db,

        clip_errcode_remove_failed = 0x40000,
        clip_errcode_remove_failed_key_not_exist,
        clip_errcode_remove_failed_del_db,

        clip_errcode_match_failed = 0x50000,
        clip_errcode_match_failed_encode_text,
        clip_errcode_match_failed_encode_image,
    } clip_errcode_e;

    // typedef enum
    // {
    //     unknown_device = 0,
    //     host_device = 1,
    //     axcl_device = 2
    // } clip_devive_e;

    typedef void *clip_handle_t;

    // typedef struct
    // {
    //     struct
    //     {
    //         char available;
    //         char version[CLIP_VERSION_LEN];
    //         struct
    //         {
    //             int remain;
    //             int total;
    //         } mem_info;
    //     } host;

    //     struct
    //     {
    //         char host_version[CLIP_VERSION_LEN];
    //         char dev_version[CLIP_VERSION_LEN];
    //         unsigned char count;
    //         struct
    //         {
    //             int temp;
    //             int cpu_usage;
    //             int npu_usage;
    //             struct
    //             {
    //                 int remain;
    //                 int total;
    //             } mem_info;
    //         } devices_info[CLIP_DEVICES_COUNT];

    //     } devices;
    // } clip_devices_t;

    typedef struct
    {
        ax_devive_e dev_type;                   // Device type
        char devid;                             // axcl device ID
        char text_encoder_path[CLIP_PATH_LEN];  // Text encoder model path
        char image_encoder_path[CLIP_PATH_LEN]; // Image encoder model path
        char tokenizer_path[CLIP_PATH_LEN];     // Tokenizer model path
        char isCN;                              // Whether it's a Chinese model (0: English, 1: Chinese)
        char db_path[CLIP_PATH_LEN];            // Database path (if empty path is specified, a folder will be created)
    } clip_init_t;

    typedef struct
    {
        unsigned char *data;
        int width;
        int height;
        int channels;
        int stride;
    } clip_image_t;

    typedef struct
    {
        float feat[CLIP_TEXT_FEAT_MAX_LEN];
        int len;
    } clip_feature_item_t;

    typedef struct
    {
        char key[CLIP_KEY_MAX_LEN];
        float score;
    } clip_result_item_t;

    // /**
    //  * @brief Enumerate available devices in the current system
    //  * @param devices Pointer to device information structure
    //  * @return int Returns 0 on success, -1 on failure
    //  */
    // int clip_enum_devices(clip_devices_t *devices);

    // /**
    //  * @brief Initialize CLIP system resources
    //  * @param dev_type Device type
    //  * @param devid Device ID
    //  * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
    //  */
    // int clip_sys_init(clip_devive_e dev_type, char devid);

    // /**
    //  * @brief Deinitialize CLIP system resources
    //  * @param dev_type Device type
    //  * @param devid Device ID
    //  * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
    //  */
    // int clip_sys_deinit(clip_devive_e dev_type, char devid);

    /**
     * @brief Create CLIP handle
     * @param init_info Pointer to initialization information structure
     * @param handle Handle pointer
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_create(clip_init_t *init_info, clip_handle_t *handle);

    /**
     * @brief Destroy CLIP handle
     * @param handle Handle
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_destroy(clip_handle_t handle);

    /**
     * @brief Add image to CLIP database
     * @param handle Handle
     * @param key Image key
     * @param image Pointer to image structure
     * @param overwrite Whether to overwrite
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_add(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN], clip_image_t *image, char overwrite);

    /**
     * @brief Remove image from CLIP database
     * @param handle Handle
     * @param key Image key
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_remove(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN]);

    /**
     * @brief Check if image exists in CLIP database
     * @param handle Handle
     * @param key Image key
     * @return int Returns 1 if exists, 0 if not exists
     */
    int clip_contain(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN]);

    /**
     * @brief Get text feature
     * @param handle Handle
     * @param text Text
     * @param feat Pointer to feature structure
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_get_text_feat(clip_handle_t handle, const char *text, clip_feature_item_t *feat);

    /**
     * @brief Feature match CLIP database images (cosine similarity)
     * @param handle Handle
     * @param feat Pointer to feature structure
     * @param results Pointer to result structure
     * @param top_k Top k results
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_match_feat(clip_handle_t handle, clip_feature_item_t *feat, clip_result_item_t *results, int top_k);

    /**
     * @brief Text match CLIP database images (softmax)
     * @param handle Handle
     * @param text Text
     * @param results Pointer to result structure
     * @param top_k Top k results
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_match_text(clip_handle_t handle, const char *text, clip_result_item_t *results, int top_k);

    /**
     * @brief Image match CLIP database images (cosine similarity)
     * @param handle Handle
     * @param image Pointer to image structure
     * @param results Pointer to result structure
     * @param top_k Top k results
     * @return clip_errcode_e Returns 0 on success, error codes see clip_errcode_e
     */
    int clip_match_image(clip_handle_t handle, clip_image_t *image, clip_result_item_t *results, int top_k);

#if defined(__cplusplus)
}
#endif

#endif // __CLIP_H__