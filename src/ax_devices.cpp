#include "ax_devices.h"

#include "enum_devices.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"

#include <queue>
#include <cstring>
#include <fstream>
#include <memory>

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

struct gInit
{
    gInit()
    {
        if (getLoader().is_init())
        {
            auto ret = axclInit();
            if (ret != 0)
            {
                printf("axclInit failed\n");
            }
        }
        else
        {
            printf("unsupport axcl\n");
        }
    }

    ~gInit()
    {
        if (getLoader().is_init())
        {
            auto ret = axclFinalize();
            if (ret != 0)
            {
                printf("axclFinalize failed\n");
            }
        }
    }
};
std::shared_ptr<gInit> gIniter = std::make_shared<gInit>();

int ax_dev_enum_devices(ax_devices_t *devices)
{
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
        else
        {
            printf("axsys or axengine not init\n");
            return ax_dev_errcode_sysinit_failed;
        }
    }
    else if (dev_type == ax_devive_e::axcl_device)
    {
        if (!getLoader().is_init())
        {
            printf("unsupport axcl\n");
            return ax_dev_errcode_axcl_sysinit_failed;
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
        else
        {
            printf("axsys or axengine not init\n");
            return ax_dev_errcode_sysdeinit_failed;
        }
    }
    else if (dev_type == ax_devive_e::axcl_device)
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
