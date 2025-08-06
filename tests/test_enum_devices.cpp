#include "ax_devices.h"
#include <iostream>
#include <cstring>

int main()
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    std::cout << "host npu avaiable:" << static_cast<int>(ax_devices.host.available) << " version:" << ax_devices.host.version << std::endl;
    std::cout << "host mem total:" << ax_devices.host.mem_info.total << " MiB remain:" << ax_devices.host.mem_info.remain << " MiB" << std::endl;

    std::cout << "Host Version: " << ax_devices.devices.host_version << std::endl;
    std::cout << "Dev Version: " << ax_devices.devices.dev_version << std::endl;
    std::cout << "Detected Devices Count: " << static_cast<int>(ax_devices.devices.count) << std::endl;

    for (unsigned char i = 0; i < ax_devices.devices.count; ++i)
    {
        std::cout << "  Device " << static_cast<int>(i) << ":" << std::endl;
        std::cout << "    Temperature: " << ax_devices.devices.devices_info[i].temp << "C" << std::endl;
        std::cout << "    CPU Usage: " << ax_devices.devices.devices_info[i].cpu_usage << "%" << std::endl;
        std::cout << "    NPU Usage: " << ax_devices.devices.devices_info[i].npu_usage << "%" << std::endl;
        std::cout << "    Memory Remaining: " << ax_devices.devices.devices_info[i].mem_info.remain << " MiB" << std::endl;
        std::cout << "    Memory Total: " << ax_devices.devices.devices_info[i].mem_info.total << " MiB" << std::endl;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        for (unsigned char i = 0; i < ax_devices.devices.count; ++i)
        {
            ax_dev_sys_init(axcl_device, i);
        }
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        for (unsigned char i = 0; i < ax_devices.devices.count; ++i)
        {
            ax_dev_sys_deinit(axcl_device, i);
        }
    }

    return 0;
}