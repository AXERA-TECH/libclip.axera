#include "ax_devices.h"

#include "enum_devices.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"

#include <cstring>
#include <mutex>
#include <vector>

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

namespace
{
std::once_flag g_backend_log_once;
std::once_flag g_axcl_init_once;
int g_axcl_init_status = ax_dev_errcode_success;

void log_supported_backends_once()
{
    std::call_once(g_backend_log_once, []()
    {
        std::vector<std::string> supported_backends;
        if (getLoader().is_init())
        {
            supported_backends.push_back("axcl");
        }

        if (get_ax_sys_loader().is_init() && get_ax_engine_loader().is_init())
        {
            supported_backends.push_back("ax650");
        }

        printf("supported backends: [");
        for (size_t i = 0; i < supported_backends.size(); ++i)
        {
            printf("%s", supported_backends[i].c_str());
            if (i + 1 < supported_backends.size())
            {
                printf(", ");
            }
        }
        printf("]\n");
    });
}

int ensure_axcl_runtime_init()
{
    log_supported_backends_once();

    if (!getLoader().is_init())
    {
        printf("unsupport axcl\n");
        return ax_dev_errcode_axcl_sysinit_failed;
    }

    std::call_once(g_axcl_init_once, []()
    {
        auto ret = axcl_Init();
        if (ret != 0)
        {
            printf("axclInit failed\n");
            g_axcl_init_status = ax_dev_errcode_axcl_sysinit_failed;
        }
    });

    return g_axcl_init_status;
}
} // namespace

int ax_dev_enum_devices(ax_devices_t *devices)
{
    log_supported_backends_once();
    get_host_info(devices);
    get_axcl_devices(devices);
    return 0;
}

int ax_dev_sys_init(ax_devive_e dev_type, char devid)
{
    if (dev_type == ax_devive_e::host_device)
    {
        if (get_ax_sys_loader().is_init() && get_ax_engine_loader().is_init())
        {
            AxSysApiLoader &ax_sys_loader = get_ax_sys_loader();
            AxEngineApiLoader &ax_engine_loader = get_ax_engine_loader();
            auto ret = ax_sys_loader.AX_SYS_Init();
            if (ret != 0)
            {
                printf("AX_SYS_Init failed\n");
                return ax_dev_errcode_sysinit_failed;
            }

            AX_ENGINE_NPU_ATTR_T npu_attr;
            memset(&npu_attr, 0, sizeof(AX_ENGINE_NPU_ATTR_T));
            npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
            ret = ax_engine_loader.AX_ENGINE_Init(&npu_attr);
            if (ret != 0)
            {
                printf("AX_ENGINE_Init failed\n");
                return ax_dev_errcode_sysinit_failed;
            }
            return ax_dev_errcode_success;
        }

        printf("axsys or axengine not init\n");
        return ax_dev_errcode_sysinit_failed;
    }

    if (dev_type == ax_devive_e::axcl_device)
    {
        auto init_ret = ensure_axcl_runtime_init();
        if (init_ret != ax_dev_errcode_success)
        {
            return init_ret;
        }

        auto ret = axcl_Dev_Init(devid);
        if (ret != 0)
        {
            printf("axcl_Dev_Init failed\n");
            return ax_dev_errcode_axcl_sysinit_failed;
        }
        return ax_dev_errcode_success;
    }

    return ax_dev_errcode_sysinit_failed;
}

int ax_dev_sys_deinit(ax_devive_e dev_type, char devid)
{
    if (dev_type == ax_devive_e::host_device)
    {
        if (get_ax_sys_loader().is_init() && get_ax_engine_loader().is_init())
        {
            AxSysApiLoader &ax_sys_loader = get_ax_sys_loader();
            AxEngineApiLoader &ax_engine_loader = get_ax_engine_loader();
            auto ret = ax_engine_loader.AX_ENGINE_Deinit();
            if (ret != 0)
            {
                printf("AX_ENGINE_Deinit failed\n");
                return ax_dev_errcode_sysdeinit_failed;
            }
            ret = ax_sys_loader.AX_SYS_Deinit();
            if (ret != 0)
            {
                printf("AX_SYS_Deinit failed\n");
                return ax_dev_errcode_sysdeinit_failed;
            }
            return ax_dev_errcode_success;
        }

        printf("axsys or axengine not init\n");
        return ax_dev_errcode_sysdeinit_failed;
    }

    if (dev_type == ax_devive_e::axcl_device)
    {
        if (!getLoader().is_init())
        {
            printf("unsupport axcl\n");
            return ax_dev_errcode_axcl_sysdeinit_failed;
        }

        auto ret = axcl_Dev_Exit(devid);
        if (ret != 0)
        {
            printf("axcl_Dev_Exit failed\n");
            return ax_dev_errcode_axcl_sysdeinit_failed;
        }
        return ax_dev_errcode_success;
    }

    return ax_dev_errcode_sysdeinit_failed;
}
