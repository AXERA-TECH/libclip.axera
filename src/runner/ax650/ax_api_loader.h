#pragma once
#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>

// 假设这些类型是已包含的头文件中定义的
#include <ax_global_type.h>
#include <ax_base_type.h>
#include <ax_pool_type.h>
#include <ax_engine_type.h>

class AxSysApiLoader
{
public:
    AxSysApiLoader()
    {
        static std::vector<std::string> v_libax_sys_so_path = {
            "/soc/lib/libax_sys.so",
            "/opt/lib/libax_sys.so",
            "/usr/lib/libax_sys.so"};
        for (auto &lib_path : v_libax_sys_so_path)
        {
            if (open(lib_path))
            {
                break;
            }
        }
        if (!handle_)
        {
            printf("open libax_sys.so failed\n");
        }
    }

    ~AxSysApiLoader()
    {
        if (handle_)
        {
            dlclose(handle_);
        }
    }

    bool is_init()
    {
        return handle_ != nullptr;
    }

    // 所有 API 函数指针

    AX_S32 (*AX_SYS_Init)(AX_VOID);
    AX_S32 (*AX_SYS_Deinit)(AX_VOID);

    /* CMM API */
    AX_S32 (*AX_SYS_MemAlloc)(AX_U64 *phyaddr, AX_VOID **pviraddr, AX_U32 size, AX_U32 align, const AX_S8 *token);
    AX_S32 (*AX_SYS_MemAllocCached)(AX_U64 *phyaddr, AX_VOID **pviraddr, AX_U32 size, AX_U32 align, const AX_S8 *token);
    AX_S32 (*AX_SYS_MemFree)(AX_U64 phyaddr, AX_VOID *pviraddr);
    AX_S32 (*AX_SYS_MflushCache)(AX_U64 phyaddr, AX_VOID *pviraddr, AX_U32 size);
    AX_S32 (*AX_SYS_MinvalidateCache)(AX_U64 phyaddr, AX_VOID *pviraddr, AX_U32 size);

private:
    void *handle_ = nullptr;

    bool open(const std::string &lib_path)
    {
        handle_ = dlopen(lib_path.c_str(), RTLD_NOW);
        if (!handle_)
        {
            // printf("open %s failed\n", lib_path.c_str());
            return false;
        }
        load_all_symbols();
        return true;
    }

    template <typename T>
    void load_symbol(T &func, const std::string &symbol_name)
    {
        dlerror(); // 清除错误信息
        func = reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
        const char *dlsym_error = dlerror();
        if (dlsym_error)
        {
            // throw std::runtime_error("dlsym failed for " + symbol_name + ": " + std::string(dlsym_error));
            printf("dlsym failed for %s: %s\n", symbol_name.c_str(), dlsym_error);
            func = nullptr;
        }
    }

    void load_all_symbols()
    {
        load_symbol(AX_SYS_Init, "AX_SYS_Init");
        load_symbol(AX_SYS_Deinit, "AX_SYS_Deinit");

        load_symbol(AX_SYS_MemAlloc, "AX_SYS_MemAlloc");
        load_symbol(AX_SYS_MemAllocCached, "AX_SYS_MemAllocCached");
        load_symbol(AX_SYS_MemFree, "AX_SYS_MemFree");
        load_symbol(AX_SYS_MflushCache, "AX_SYS_MflushCache");
        load_symbol(AX_SYS_MinvalidateCache, "AX_SYS_MinvalidateCache");
    }
};

class AxEngineApiLoader
{
public:
    AxEngineApiLoader()
    {
        static std::vector<std::string> v_libax_engine_so_path = {
            "/soc/lib/libax_engine.so",
            "/opt/lib/libax_engine.so",
            "/usr/lib/libax_engine.so"};
        for (auto &lib_path : v_libax_engine_so_path)
        {
            if (open(lib_path))
            {
                break;
            }
        }
        if (!handle_)
        {
            printf("open libax_engine.so failed\n");
        }
    }

    ~AxEngineApiLoader()
    {
        if (handle_)
        {
            dlclose(handle_);
        }
    }

    bool is_init()
    {
        return handle_ != nullptr;
    }

    // 所有 API 函数指针
    const AX_CHAR *(*AX_ENGINE_GetVersion)(AX_VOID);
    AX_VOID (*AX_ENGINE_NPUReset)(AX_VOID);
    AX_S32 (*AX_ENGINE_Init)(AX_ENGINE_NPU_ATTR_T *pNpuAttr);
    AX_S32 (*AX_ENGINE_GetVNPUAttr)(AX_ENGINE_NPU_ATTR_T *pNpuAttr);
    AX_S32 (*AX_ENGINE_Deinit)(AX_VOID);
    AX_S32 (*AX_ENGINE_GetModelType)(const AX_VOID *pData, AX_U32 nDataSize, AX_ENGINE_MODEL_TYPE_T *pModelType);
    AX_S32 (*AX_ENGINE_CreateHandle)(AX_ENGINE_HANDLE *pHandle, const AX_VOID *pData, AX_U32 nDataSize);
    AX_S32 (*AX_ENGINE_CreateHandleV2)(AX_ENGINE_HANDLE *pHandle, const AX_VOID *pData, AX_U32 nDataSize, AX_ENGINE_HANDLE_EXTRA_T *pExtraParam);
    AX_S32 (*AX_ENGINE_DestroyHandle)(AX_ENGINE_HANDLE nHandle);
    AX_S32 (*AX_ENGINE_GetIOInfo)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_IO_INFO_T **pIO);
    AX_S32 (*AX_ENGINE_GetGroupIOInfoCount)(AX_ENGINE_HANDLE nHandle, AX_U32 *pCount);
    AX_S32 (*AX_ENGINE_GetGroupIOInfo)(AX_ENGINE_HANDLE nHandle, AX_U32 nIndex, AX_ENGINE_IO_INFO_T **pIO);
    AX_S32 (*AX_ENGINE_GetHandleModelType)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_MODEL_TYPE_T *pModelType);
    AX_S32 (*AX_ENGINE_CreateContext)(AX_ENGINE_HANDLE handle);
    AX_S32 (*AX_ENGINE_CreateContextV2)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_CONTEXT_T *pContext);
    AX_S32 (*AX_ENGINE_RunSync)(AX_ENGINE_HANDLE handle, AX_ENGINE_IO_T *pIO);
    AX_S32 (*AX_ENGINE_RunSyncV2)(AX_ENGINE_HANDLE handle, AX_ENGINE_CONTEXT_T context, AX_ENGINE_IO_T *pIO);
    AX_S32 (*AX_ENGINE_RunGroupIOSync)(AX_ENGINE_HANDLE handle, AX_ENGINE_CONTEXT_T context, AX_U32 nIndex, AX_ENGINE_IO_T *pIO);
    AX_S32 (*AX_ENGINE_SetAffinity)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_NPU_SET_T nNpuSet);
    AX_S32 (*AX_ENGINE_GetAffinity)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_NPU_SET_T *pNpuSet);
    AX_S32 (*AX_ENGINE_GetCMMUsage)(AX_ENGINE_HANDLE nHandle, AX_ENGINE_CMM_INFO *pCMMInfo);
    const AX_CHAR *(*AX_ENGINE_GetModelToolsVersion)(AX_ENGINE_HANDLE nHandle);

private:
    void *handle_ = nullptr;

    bool open(const std::string &lib_path = "/usr/lib/axcl/libaxcl_rt.so")
    {
        handle_ = dlopen(lib_path.c_str(), RTLD_NOW);
        if (!handle_)
        {
            // printf("open %s failed\n", lib_path.c_str());
            return false;
        }
        load_all_symbols();
        return true;
    }

    template <typename T>
    void load_symbol(T &func, const std::string &symbol_name)
    {
        dlerror(); // 清除错误信息
        func = reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
        const char *dlsym_error = dlerror();
        if (dlsym_error)
        {
            // throw std::runtime_error("dlsym failed for " + symbol_name + ": " + std::string(dlsym_error));
            printf("dlsym failed for %s: %s\n", symbol_name.c_str(), dlsym_error);
            func = nullptr;
        }
    }

    void load_all_symbols()
    {
        load_symbol(AX_ENGINE_GetVersion, "AX_ENGINE_GetVersion");
        load_symbol(AX_ENGINE_NPUReset, "AX_ENGINE_NPUReset");
        load_symbol(AX_ENGINE_Init, "AX_ENGINE_Init");
        load_symbol(AX_ENGINE_GetVNPUAttr, "AX_ENGINE_GetVNPUAttr");
        load_symbol(AX_ENGINE_Deinit, "AX_ENGINE_Deinit");
        load_symbol(AX_ENGINE_GetModelType, "AX_ENGINE_GetModelType");
        load_symbol(AX_ENGINE_CreateHandle, "AX_ENGINE_CreateHandle");
        load_symbol(AX_ENGINE_CreateHandleV2, "AX_ENGINE_CreateHandleV2");
        load_symbol(AX_ENGINE_DestroyHandle, "AX_ENGINE_DestroyHandle");
        load_symbol(AX_ENGINE_GetIOInfo, "AX_ENGINE_GetIOInfo");
        load_symbol(AX_ENGINE_GetGroupIOInfoCount, "AX_ENGINE_GetGroupIOInfoCount");
        load_symbol(AX_ENGINE_GetGroupIOInfo, "AX_ENGINE_GetGroupIOInfo");
        load_symbol(AX_ENGINE_GetHandleModelType, "AX_ENGINE_GetHandleModelType");
        load_symbol(AX_ENGINE_CreateContext, "AX_ENGINE_CreateContext");
        load_symbol(AX_ENGINE_CreateContextV2, "AX_ENGINE_CreateContextV2");
        load_symbol(AX_ENGINE_RunSync, "AX_ENGINE_RunSync");
        load_symbol(AX_ENGINE_RunSyncV2, "AX_ENGINE_RunSyncV2");
        load_symbol(AX_ENGINE_RunGroupIOSync, "AX_ENGINE_RunGroupIOSync");
        load_symbol(AX_ENGINE_SetAffinity, "AX_ENGINE_SetAffinity");
        load_symbol(AX_ENGINE_GetAffinity, "AX_ENGINE_GetAffinity");
        load_symbol(AX_ENGINE_GetCMMUsage, "AX_ENGINE_GetCMMUsage");
        load_symbol(AX_ENGINE_GetModelToolsVersion, "AX_ENGINE_GetModelToolsVersion");
    }
};
