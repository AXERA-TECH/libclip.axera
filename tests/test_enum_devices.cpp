#include "clip.h"
#include <iostream>
#include <cstring>

int main()
{
    clip_devices_t clip_devices;
    memset(&clip_devices, 0, sizeof(clip_devices_t));
    if (clip_enum_devices(&clip_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    std::cout << "host npu avaiable:" << static_cast<int>(clip_devices.host.available) << " version:" << clip_devices.host.version << std::endl;
    std::cout << "host mem total:" << clip_devices.host.mem_info.total << " MiB remain:" << clip_devices.host.mem_info.remain << " MiB" << std::endl;

    std::cout << "Host Version: " << clip_devices.devices.host_version << std::endl;
    std::cout << "Dev Version: " << clip_devices.devices.dev_version << std::endl;
    std::cout << "Detected Devices Count: " << static_cast<int>(clip_devices.devices.count) << std::endl;

    for (unsigned char i = 0; i < clip_devices.devices.count; ++i)
    {
        std::cout << "  Device " << static_cast<int>(i) << ":" << std::endl;
        std::cout << "    Temperature: " << clip_devices.devices.devices_info[i].temp << "C" << std::endl;
        std::cout << "    CPU Usage: " << clip_devices.devices.devices_info[i].cpu_usage << "%" << std::endl;
        std::cout << "    NPU Usage: " << clip_devices.devices.devices_info[i].npu_usage << "%" << std::endl;
        std::cout << "    Memory Remaining: " << clip_devices.devices.devices_info[i].mem_info.remain << " MiB" << std::endl;
        std::cout << "    Memory Total: " << clip_devices.devices.devices_info[i].mem_info.total << " MiB" << std::endl;
    }

    if (clip_devices.host.available)
    {
        clip_sys_init(host_device, -1);
    }

    if (clip_devices.devices.count > 0)
    {
        for (unsigned char i = 0; i < clip_devices.devices.count; ++i)
        {
            clip_sys_init(axcl_device, i);
        }
    }

    if (clip_devices.host.available)
    {
        clip_sys_deinit(host_device, -1);
    }

    if (clip_devices.devices.count > 0)
    {
        for (unsigned char i = 0; i < clip_devices.devices.count; ++i)
        {
            clip_sys_deinit(axcl_device, i);
        }
    }

    return 0;
}