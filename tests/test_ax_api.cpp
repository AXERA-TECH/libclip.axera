#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"

#include <fstream>
#include <vector>
#include <cstring>

AxSysApiLoader &get_ax_sys_loader();

AxEngineApiLoader &get_ax_engine_loader();

int main()
{
    AxSysApiLoader &ax_sys_loader = get_ax_sys_loader();

    AxEngineApiLoader &ax_engine_loader = get_ax_engine_loader();

    ax_sys_loader.AX_SYS_Init();
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(AX_ENGINE_NPU_ATTR_T));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ax_engine_loader.AX_ENGINE_Init(&npu_attr);

    ax_runner_ax650 runner;
    std::ifstream file("/home/axera/CLIP-ONNX-AX650-CPP/build/cnclip/cnclip_vit_l14_336px_text_u16.axmodel", std::ios::binary);
    if (!file.is_open())
    {
        printf("open file failed\n");
        return -1;
    }
    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    runner.init(model_data.data(), model_data.size(), 0);

    ax_engine_loader.AX_ENGINE_Deinit();
    ax_sys_loader.AX_SYS_Deinit();
    return 0;
}