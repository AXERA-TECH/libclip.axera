#ifndef __AX_DEVICES_H__
#define __AX_DEVICES_H__

#if defined(__cplusplus)
extern "C"
{
#endif
#define AX_DEVICES_COUNT 16
#define AX_VERSION_LEN 32

    typedef enum
    {
        ax_dev_errcode_failed = -1,
        ax_dev_errcode_success = 0,

        ax_dev_errcode_sysinit_failed,
        ax_dev_errcode_sysdeinit_failed,
        ax_dev_errcode_axcl_sysinit_failed,
        ax_dev_errcode_axcl_sysdeinit_failed,
    } ax_dev_errcode_e;

    typedef enum
    {
        unknown_device = 0,
        host_device = 1,
        axcl_device = 2
    } ax_devive_e;

    typedef struct
    {
        struct
        {
            char available;
            char version[AX_VERSION_LEN];
            struct
            {
                int remain;
                int total;
            } mem_info;
        } host;

        struct
        {
            char host_version[AX_VERSION_LEN];
            char dev_version[AX_VERSION_LEN];
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
            } devices_info[AX_DEVICES_COUNT];

        } devices;
    } ax_devices_t;

    int ax_dev_enum_devices(ax_devices_t *devices);
    int ax_dev_sys_init(ax_devive_e dev_type, char devid);
    int ax_dev_sys_deinit(ax_devive_e dev_type, char devid);

#if defined(__cplusplus)
}
#endif

#endif // __AX_DEVICES_H__