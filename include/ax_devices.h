#ifndef __AX_DEVICES_H__
#define __AX_DEVICES_H__

/*
 * Windows support:
 *  - Build DLL:   define AX_BUILD_DLL
 *  - Use DLL:     do NOT define AX_BUILD_DLL (default -> dllimport)
 *  - Static lib:  define AX_STATIC
 */

#if defined(_WIN32) || defined(_WIN64)
#define AX_PLATFORM_WINDOWS 1
#else
#define AX_PLATFORM_WINDOWS 0
#endif

/* Export / Import */
#if AX_PLATFORM_WINDOWS
#if defined(AX_STATIC)
#define AX_API
#else
#if defined(AX_BUILD_DLL)
#define AX_API __declspec(dllexport)
#else
#define AX_API __declspec(dllimport)
#endif
#endif

#ifndef AX_CALL
#define AX_CALL __cdecl
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define AX_API __attribute__((visibility("default")))
#else
#define AX_API
#endif

#ifndef AX_CALL
#define AX_CALL
#endif
#endif

#ifdef __cplusplus
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

    AX_API int AX_CALL ax_dev_enum_devices(ax_devices_t *devices);
    AX_API int AX_CALL ax_dev_sys_init(ax_devive_e dev_type, char devid);
    AX_API int AX_CALL ax_dev_sys_deinit(ax_devive_e dev_type, char devid);

#ifdef __cplusplus
}
#endif

#endif // __AX_DEVICES_H__
