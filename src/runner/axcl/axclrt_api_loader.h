#pragma once
#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>

// 假设这些类型是已包含的头文件中定义的
#include <axcl_rt_type.h>
#include <axcl_rt_engine_type.h>

class AxclApiLoader
{
public:
    explicit AxclApiLoader()
    {
        static std::vector<std::string> v_libaxcl_rt_so_path = {
            "/usr/lib/axcl/libaxcl_rt.so"};

        for (const auto &lib_path : v_libaxcl_rt_so_path)
        {
            handle_ = dlopen(lib_path.c_str(), RTLD_NOW);
            if (handle_)
            {
                break;
            }
        }

        if (!handle_)
        {
            printf("open libaxcl_rt.so failed\n");
        }
        else
        {
            load_all_symbols();
        }
    }

    ~AxclApiLoader()
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
    axclError (*axclInit)(const char *config);
    axclError (*axclFinalize)();

    axclError (*axclrtSetDevice)(int32_t deviceId);
    axclError (*axclrtResetDevice)(int32_t deviceId);
    axclError (*axclrtGetDevice)(int32_t *deviceId);
    axclError (*axclrtGetDeviceCount)(uint32_t *count);
    axclError (*axclrtGetDeviceList)(axclrtDeviceList *deviceList);
    axclError (*axclrtSynchronizeDevice)();
    // axclError (*axclrtGetDeviceProperties)(int32_t deviceId, axclrtDeviceProperties *properties);
    axclError (*axclrtRebootDevice)(int32_t deviceId);

    axclError (*axclrtMalloc)(void **devPtr, size_t size, axclrtMemMallocPolicy policy);
    axclError (*axclrtMallocCached)(void **devPtr, size_t size, axclrtMemMallocPolicy policy);
    axclError (*axclrtFree)(void *devPtr);
    axclError (*axclrtMemFlush)(void *devPtr, size_t size);
    axclError (*axclrtMemInvalidate)(void *devPtr, size_t size);
    axclError (*axclrtMallocHost)(void **hostPtr, size_t size);
    axclError (*axclrtFreeHost)(void *hostPtr);
    axclError (*axclrtMemset)(void *devPtr, uint8_t value, size_t count);
    axclError (*axclrtMemcpy)(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind);
    axclError (*axclrtMemcmp)(const void *devPtr1, const void *devPtr2, size_t count);

    axclError (*axclrtEngineInit)(axclrtEngineVNpuKind npuKind);
    axclError (*axclrtEngineGetVNpuKind)(axclrtEngineVNpuKind *npuKind);
    axclError (*axclrtEngineFinalize)();
    axclError (*axclrtEngineLoadFromFile)(const char *modelPath, uint64_t *modelId);
    axclError (*axclrtEngineLoadFromMem)(const void *model, uint64_t modelSize, uint64_t *modelId);
    axclError (*axclrtEngineUnload)(uint64_t modelId);
    const char *(*axclrtEngineGetModelCompilerVersion)(uint64_t modelId);
    axclError (*axclrtEngineSetAffinity)(uint64_t modelId, axclrtEngineSet set);
    axclError (*axclrtEngineGetAffinity)(uint64_t modelId, axclrtEngineSet *set);
    axclError (*axclrtEngineSetContextAffinity)(uint64_t modelId, uint64_t contextId, axclrtEngineSet set);
    axclError (*axclrtEngineGetContextAffinity)(uint64_t modelId, uint64_t contextId, axclrtEngineSet *set);
    axclError (*axclrtEngineGetUsage)(const char *modelPath, int64_t *sysSize, int64_t *cmmSize);
    axclError (*axclrtEngineGetUsageFromMem)(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize);
    axclError (*axclrtEngineGetUsageFromModelId)(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize);
    axclError (*axclrtEngineGetModelType)(const char *modelPath, axclrtEngineModelKind *modelType);
    axclError (*axclrtEngineGetModelTypeFromMem)(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType);
    axclError (*axclrtEngineGetModelTypeFromModelId)(uint64_t modelId, axclrtEngineModelKind *modelType);
    axclError (*axclrtEngineGetIOInfo)(uint64_t modelId, axclrtEngineIOInfo *ioInfo);
    axclError (*axclrtEngineDestroyIOInfo)(axclrtEngineIOInfo ioInfo);
    axclError (*axclrtEngineGetShapeGroupsCount)(axclrtEngineIOInfo ioInfo, int32_t *count);
    uint32_t (*axclrtEngineGetNumInputs)(axclrtEngineIOInfo ioInfo);
    uint32_t (*axclrtEngineGetNumOutputs)(axclrtEngineIOInfo ioInfo);
    uint64_t (*axclrtEngineGetInputSizeByIndex)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);
    uint64_t (*axclrtEngineGetOutputSizeByIndex)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);
    const char *(*axclrtEngineGetInputNameByIndex)(axclrtEngineIOInfo ioInfo, uint32_t index);
    const char *(*axclrtEngineGetOutputNameByIndex)(axclrtEngineIOInfo ioInfo, uint32_t index);
    int32_t (*axclrtEngineGetInputIndexByName)(axclrtEngineIOInfo ioInfo, const char *name);
    int32_t (*axclrtEngineGetOutputIndexByName)(axclrtEngineIOInfo ioInfo, const char *name);
    axclError (*axclrtEngineGetInputDims)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);
    axclError (*axclrtEngineGetInputDataType)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type);
    axclError (*axclrtEngineGetOutputDataType)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type);
    axclError (*axclrtEngineGetInputDataLayout)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout);
    axclError (*axclrtEngineGetOutputDataLayout)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout);
    axclError (*axclrtEngineGetOutputDims)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);
    axclError (*axclrtEngineCreateIO)(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io);
    axclError (*axclrtEngineDestroyIO)(axclrtEngineIO io);
    axclError (*axclrtEngineSetInputBufferByIndex)(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    axclError (*axclrtEngineSetOutputBufferByIndex)(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    axclError (*axclrtEngineSetInputBufferByName)(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size);
    axclError (*axclrtEngineSetOutputBufferByName)(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size);
    axclError (*axclrtEngineGetInputBufferByIndex)(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size);
    axclError (*axclrtEngineGetOutputBufferByIndex)(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size);
    axclError (*axclrtEngineGetInputBufferByName)(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size);
    axclError (*axclrtEngineGetOutputBufferByName)(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size);
    axclError (*axclrtEngineSetDynamicBatchSize)(axclrtEngineIO io, uint32_t batchSize);
    axclError (*axclrtEngineCreateContext)(uint64_t modelId, uint64_t *contextId);
    axclError (*axclrtEngineExecute)(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io);
    axclError (*axclrtEngineExecuteAsync)(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream);

private:
    void *handle_ = nullptr;

    template <typename T>
    void load_symbol(T &func, const std::string &symbol_name)
    {
        dlerror(); // 清除错误信息
        func = reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
        const char *dlsym_error = dlerror();
        if (dlsym_error)
        {
            // throw std::runtime_error("dlsym failed for " + symbol_name + ": " + std::string(dlsym_error));
            func = nullptr;
            printf("dlsym failed for %s: %s\n", symbol_name.c_str(), dlsym_error);
        }
    }

    void load_all_symbols()
    {
        load_symbol(axclInit, "axclInit");
        load_symbol(axclFinalize, "axclFinalize");

        load_symbol(axclrtSetDevice, "axclrtSetDevice");
        load_symbol(axclrtResetDevice, "axclrtResetDevice");
        load_symbol(axclrtGetDevice, "axclrtGetDevice");
        load_symbol(axclrtGetDeviceCount, "axclrtGetDeviceCount");
        load_symbol(axclrtGetDeviceList, "axclrtGetDeviceList");
        load_symbol(axclrtSynchronizeDevice, "axclrtSynchronizeDevice");
        // load_symbol(axclrtGetDeviceProperties, "axclrtGetDeviceProperties");
        load_symbol(axclrtRebootDevice, "axclrtRebootDevice");

        load_symbol(axclrtMalloc, "axclrtMalloc");
        load_symbol(axclrtMallocCached, "axclrtMallocCached");
        load_symbol(axclrtFree, "axclrtFree");
        load_symbol(axclrtMemFlush, "axclrtMemFlush");
        load_symbol(axclrtMemInvalidate, "axclrtMemInvalidate");
        load_symbol(axclrtMallocHost, "axclrtMallocHost");
        load_symbol(axclrtFreeHost, "axclrtFreeHost");
        load_symbol(axclrtMemset, "axclrtMemset");
        load_symbol(axclrtMemcpy, "axclrtMemcpy");
        load_symbol(axclrtMemcmp, "axclrtMemcmp");

        load_symbol(axclrtEngineInit, "axclrtEngineInit");
        load_symbol(axclrtEngineGetVNpuKind, "axclrtEngineGetVNpuKind");
        load_symbol(axclrtEngineFinalize, "axclrtEngineFinalize");
        load_symbol(axclrtEngineLoadFromFile, "axclrtEngineLoadFromFile");
        load_symbol(axclrtEngineLoadFromMem, "axclrtEngineLoadFromMem");
        load_symbol(axclrtEngineUnload, "axclrtEngineUnload");
        load_symbol(axclrtEngineGetModelCompilerVersion, "axclrtEngineGetModelCompilerVersion");
        load_symbol(axclrtEngineSetAffinity, "axclrtEngineSetAffinity");
        load_symbol(axclrtEngineGetAffinity, "axclrtEngineGetAffinity");
        load_symbol(axclrtEngineSetContextAffinity, "axclrtEngineSetContextAffinity");
        load_symbol(axclrtEngineGetContextAffinity, "axclrtEngineGetContextAffinity");
        load_symbol(axclrtEngineGetUsage, "axclrtEngineGetUsage");
        load_symbol(axclrtEngineGetUsageFromMem, "axclrtEngineGetUsageFromMem");
        load_symbol(axclrtEngineGetUsageFromModelId, "axclrtEngineGetUsageFromModelId");
        load_symbol(axclrtEngineGetModelType, "axclrtEngineGetModelType");
        load_symbol(axclrtEngineGetModelTypeFromMem, "axclrtEngineGetModelTypeFromMem");
        load_symbol(axclrtEngineGetModelTypeFromModelId, "axclrtEngineGetModelTypeFromModelId");
        load_symbol(axclrtEngineGetIOInfo, "axclrtEngineGetIOInfo");
        load_symbol(axclrtEngineDestroyIOInfo, "axclrtEngineDestroyIOInfo");
        load_symbol(axclrtEngineGetShapeGroupsCount, "axclrtEngineGetShapeGroupsCount");
        load_symbol(axclrtEngineGetNumInputs, "axclrtEngineGetNumInputs");
        load_symbol(axclrtEngineGetNumOutputs, "axclrtEngineGetNumOutputs");
        load_symbol(axclrtEngineGetInputSizeByIndex, "axclrtEngineGetInputSizeByIndex");
        load_symbol(axclrtEngineGetOutputSizeByIndex, "axclrtEngineGetOutputSizeByIndex");
        load_symbol(axclrtEngineGetInputNameByIndex, "axclrtEngineGetInputNameByIndex");
        load_symbol(axclrtEngineGetOutputNameByIndex, "axclrtEngineGetOutputNameByIndex");
        load_symbol(axclrtEngineGetInputIndexByName, "axclrtEngineGetInputIndexByName");
        load_symbol(axclrtEngineGetOutputIndexByName, "axclrtEngineGetOutputIndexByName");
        load_symbol(axclrtEngineGetInputDims, "axclrtEngineGetInputDims");
        load_symbol(axclrtEngineGetInputDataType, "axclrtEngineGetInputDataType");
        load_symbol(axclrtEngineGetOutputDataType, "axclrtEngineGetOutputDataType");
        load_symbol(axclrtEngineGetInputDataLayout, "axclrtEngineGetInputDataLayout");
        load_symbol(axclrtEngineGetOutputDataLayout, "axclrtEngineGetOutputDataLayout");
        load_symbol(axclrtEngineGetOutputDims, "axclrtEngineGetOutputDims");
        load_symbol(axclrtEngineCreateIO, "axclrtEngineCreateIO");
        load_symbol(axclrtEngineDestroyIO, "axclrtEngineDestroyIO");
        load_symbol(axclrtEngineSetInputBufferByIndex, "axclrtEngineSetInputBufferByIndex");
        load_symbol(axclrtEngineSetOutputBufferByIndex, "axclrtEngineSetOutputBufferByIndex");
        load_symbol(axclrtEngineSetInputBufferByName, "axclrtEngineSetInputBufferByName");
        load_symbol(axclrtEngineSetOutputBufferByName, "axclrtEngineSetOutputBufferByName");
        load_symbol(axclrtEngineGetInputBufferByIndex, "axclrtEngineGetInputBufferByIndex");
        load_symbol(axclrtEngineGetOutputBufferByIndex, "axclrtEngineGetOutputBufferByIndex");
        load_symbol(axclrtEngineGetInputBufferByName, "axclrtEngineGetInputBufferByName");
        load_symbol(axclrtEngineGetOutputBufferByName, "axclrtEngineGetOutputBufferByName");
        load_symbol(axclrtEngineSetDynamicBatchSize, "axclrtEngineSetDynamicBatchSize");
        load_symbol(axclrtEngineCreateContext, "axclrtEngineCreateContext");
        load_symbol(axclrtEngineExecute, "axclrtEngineExecute");
        load_symbol(axclrtEngineExecuteAsync, "axclrtEngineExecuteAsync");
    }
};
