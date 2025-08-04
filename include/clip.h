#ifndef __CLIP_H__
#define __CLIP_H__

#if defined(__cplusplus)
extern "C"
{
#endif
#define CLIP_DEVICES_COUNT 16
#define CLIP_VERSION_LEN 32
#define CLIP_KEY_MAX_LEN 64
#define CLIP_PATH_LEN 128

    typedef enum
    {
        clip_errcode_failed = -1,
        clip_errcode_success = 0,

        clip_errcode_invalid_ptr,
        clip_errcode_sysinit_failed,
        clip_errcode_sysdeinit_failed,
        clip_errcode_axcl_sysinit_failed,
        clip_errcode_axcl_sysdeinit_failed,

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

    typedef enum
    {
        unknown_device = 0,
        host_device = 1,
        axcl_device = 2
    } clip_devive_e;

    typedef void *clip_handle_t;

    typedef struct
    {
        struct
        {
            char available;
            char version[CLIP_VERSION_LEN];
            struct
            {
                int remain;
                int total;
            } mem_info;
        } host;

        struct
        {
            char host_version[CLIP_VERSION_LEN];
            char dev_version[CLIP_VERSION_LEN];
            unsigned char count;
            struct
            {
                int temp;
                int cpu_usage;
                int npu_usage;
                struct
                {
                    int remain;
                    int total;
                } mem_info;
            } devices_info[CLIP_DEVICES_COUNT];

        } devices;
    } clip_devices_t;

    typedef struct
    {
        clip_devive_e dev_type;
        char devid;
        char text_encoder_path[CLIP_PATH_LEN];
        char image_encoder_path[CLIP_PATH_LEN];
        char tokenizer_path[CLIP_PATH_LEN];
        char isCN;

        char db_path[CLIP_PATH_LEN];
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
        char key[CLIP_KEY_MAX_LEN];
        float score;
    } clip_result_item_t;

    int clip_enum_devices(clip_devices_t *devices);

    int clip_sys_init(clip_devive_e dev_type, char devid);
    int clip_sys_deinit(clip_devive_e dev_type, char devid);

    int clip_create(clip_init_t *init_info, clip_handle_t *handle);
    int clip_destroy(clip_handle_t handle);

    int clip_add(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN], clip_image_t *image, char overwrite);
    int clip_remove(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN]);
    int clip_contain(clip_handle_t handle, char key[CLIP_KEY_MAX_LEN]);

    int clip_match_text(clip_handle_t handle, const char *text, clip_result_item_t *results, int top_k);
    int clip_match_image(clip_handle_t handle, clip_image_t *image, clip_result_item_t *results, int top_k);

#if defined(__cplusplus)
}
#endif

#endif // __CLIP_H__