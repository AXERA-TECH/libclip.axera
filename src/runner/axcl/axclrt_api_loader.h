#pragma once
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>

#if defined(_WIN32) || defined(_WIN64)
#define AXCL_PLATFORM_WINDOWS 1
#include <windows.h>
#else
#define AXCL_PLATFORM_WINDOWS 0
#include <dlfcn.h>
#endif

// 假设这些类型是已包含的头文件中定义的
#include <axcl_rt_type.h>
#include <axcl_rt_engine_type.h>

class AxclApiLoader
{
public:
    explicit AxclApiLoader()
    {
#if AXCL_PLATFORM_WINDOWS
        static std::vector<std::string> v_libaxcl_rt_path = {
            "C:/Program Files/AXCL/axcl/out/axcl_win_x64/bin/libaxcl_rt.dll"};
#else
        static std::vector<std::string> v_libaxcl_rt_path = {
            "/usr/lib/axcl/libaxcl_rt.so"};
#endif

        for (const auto &lib_path : v_libaxcl_rt_path)
        {
            handle_ = open_library(lib_path);
            if (handle_)
            {
                loaded_path_ = lib_path;
                break;
            }
        }

        if (!handle_)
        {
            std::string err = last_error_string();
            std::fprintf(stderr, "open axcl runtime library failed. last_error=%s\n", err.c_str());
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
            close_library(handle_);
            handle_ = nullptr;
        }
    }

    bool is_init() const
    {
        return handle_ != nullptr;
    }

    const std::string &loaded_path() const
    {
        return loaded_path_;
    }

    // 所有 API 函数指针
    axclError (*axclInit)(const char *config) = nullptr;
    axclError (*axclFinalize)() = nullptr;

    axclError (*axclrtSetDevice)(int32_t deviceId) = nullptr;
    axclError (*axclrtResetDevice)(int32_t deviceId) = nullptr;
    axclError (*axclrtGetDevice)(int32_t *deviceId) = nullptr;
    axclError (*axclrtGetDeviceCount)(uint32_t *count) = nullptr;
    axclError (*axclrtGetDeviceList)(axclrtDeviceList *deviceList) = nullptr;
    axclError (*axclrtSynchronizeDevice)() = nullptr;
    // axclError (*axclrtGetDeviceProperties)(int32_t deviceId, axclrtDeviceProperties *properties) = nullptr;
    axclError (*axclrtRebootDevice)(int32_t deviceId) = nullptr;

    axclError (*axclrtMalloc)(void **devPtr, size_t size, axclrtMemMallocPolicy policy) = nullptr;
    axclError (*axclrtMallocCached)(void **devPtr, size_t size, axclrtMemMallocPolicy policy) = nullptr;
    axclError (*axclrtFree)(void *devPtr) = nullptr;
    axclError (*axclrtMemFlush)(void *devPtr, size_t size) = nullptr;
    axclError (*axclrtMemInvalidate)(void *devPtr, size_t size) = nullptr;
    axclError (*axclrtMallocHost)(void **hostPtr, size_t size) = nullptr;
    axclError (*axclrtFreeHost)(void *hostPtr) = nullptr;
    axclError (*axclrtMemset)(void *devPtr, uint8_t value, size_t count) = nullptr;
    axclError (*axclrtMemcpy)(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind) = nullptr;
    axclError (*axclrtMemcmp)(const void *devPtr1, const void *devPtr2, size_t count) = nullptr;

    axclError (*axclrtEngineInit)(axclrtEngineVNpuKind npuKind) = nullptr;
    axclError (*axclrtEngineGetVNpuKind)(axclrtEngineVNpuKind *npuKind) = nullptr;
    axclError (*axclrtEngineFinalize)() = nullptr;
    axclError (*axclrtEngineLoadFromFile)(const char *modelPath, uint64_t *modelId) = nullptr;
    axclError (*axclrtEngineLoadFromMem)(const void *model, uint64_t modelSize, uint64_t *modelId) = nullptr;
    axclError (*axclrtEngineUnload)(uint64_t modelId) = nullptr;
    const char *(*axclrtEngineGetModelCompilerVersion)(uint64_t modelId) = nullptr;
    axclError (*axclrtEngineSetAffinity)(uint64_t modelId, axclrtEngineSet set) = nullptr;
    axclError (*axclrtEngineGetAffinity)(uint64_t modelId, axclrtEngineSet *set) = nullptr;
    axclError (*axclrtEngineSetContextAffinity)(uint64_t modelId, uint64_t contextId, axclrtEngineSet set) = nullptr;
    axclError (*axclrtEngineGetContextAffinity)(uint64_t modelId, uint64_t contextId, axclrtEngineSet *set) = nullptr;
    axclError (*axclrtEngineGetUsage)(const char *modelPath, int64_t *sysSize, int64_t *cmmSize) = nullptr;
    axclError (*axclrtEngineGetUsageFromMem)(const void *model, uint64_t modelSize, int64_t *sysSize, int64_t *cmmSize) = nullptr;
    axclError (*axclrtEngineGetUsageFromModelId)(uint64_t modelId, int64_t *sysSize, int64_t *cmmSize) = nullptr;
    axclError (*axclrtEngineGetModelType)(const char *modelPath, axclrtEngineModelKind *modelType) = nullptr;
    axclError (*axclrtEngineGetModelTypeFromMem)(const void *model, uint64_t modelSize, axclrtEngineModelKind *modelType) = nullptr;
    axclError (*axclrtEngineGetModelTypeFromModelId)(uint64_t modelId, axclrtEngineModelKind *modelType) = nullptr;
    axclError (*axclrtEngineGetIOInfo)(uint64_t modelId, axclrtEngineIOInfo *ioInfo) = nullptr;
    axclError (*axclrtEngineDestroyIOInfo)(axclrtEngineIOInfo ioInfo) = nullptr;
    axclError (*axclrtEngineGetShapeGroupsCount)(axclrtEngineIOInfo ioInfo, int32_t *count) = nullptr;
    uint32_t (*axclrtEngineGetNumInputs)(axclrtEngineIOInfo ioInfo) = nullptr;
    uint32_t (*axclrtEngineGetNumOutputs)(axclrtEngineIOInfo ioInfo) = nullptr;
    uint64_t (*axclrtEngineGetInputSizeByIndex)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index) = nullptr;
    uint64_t (*axclrtEngineGetOutputSizeByIndex)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index) = nullptr;
    const char *(*axclrtEngineGetInputNameByIndex)(axclrtEngineIOInfo ioInfo, uint32_t index) = nullptr;
    const char *(*axclrtEngineGetOutputNameByIndex)(axclrtEngineIOInfo ioInfo, uint32_t index) = nullptr;
    int32_t (*axclrtEngineGetInputIndexByName)(axclrtEngineIOInfo ioInfo, const char *name) = nullptr;
    int32_t (*axclrtEngineGetOutputIndexByName)(axclrtEngineIOInfo ioInfo, const char *name) = nullptr;
    axclError (*axclrtEngineGetInputDims)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims) = nullptr;
    axclError (*axclrtEngineGetInputDataType)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type) = nullptr;
    axclError (*axclrtEngineGetOutputDataType)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type) = nullptr;
    axclError (*axclrtEngineGetInputDataLayout)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout) = nullptr;
    axclError (*axclrtEngineGetOutputDataLayout)(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout) = nullptr;
    axclError (*axclrtEngineGetOutputDims)(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims) = nullptr;
    axclError (*axclrtEngineCreateIO)(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io) = nullptr;
    axclError (*axclrtEngineDestroyIO)(axclrtEngineIO io) = nullptr;
    axclError (*axclrtEngineSetInputBufferByIndex)(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size) = nullptr;
    axclError (*axclrtEngineSetOutputBufferByIndex)(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size) = nullptr;
    axclError (*axclrtEngineSetInputBufferByName)(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size) = nullptr;
    axclError (*axclrtEngineSetOutputBufferByName)(axclrtEngineIO io, const char *name, const void *dataBuffer, uint64_t size) = nullptr;
    axclError (*axclrtEngineGetInputBufferByIndex)(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size) = nullptr;
    axclError (*axclrtEngineGetOutputBufferByIndex)(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size) = nullptr;
    axclError (*axclrtEngineGetInputBufferByName)(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size) = nullptr;
    axclError (*axclrtEngineGetOutputBufferByName)(axclrtEngineIO io, const char *name, void **dataBuffer, uint64_t *size) = nullptr;
    axclError (*axclrtEngineSetDynamicBatchSize)(axclrtEngineIO io, uint32_t batchSize) = nullptr;
    axclError (*axclrtEngineCreateContext)(uint64_t modelId, uint64_t *contextId) = nullptr;
    axclError (*axclrtEngineExecute)(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io) = nullptr;
    axclError (*axclrtEngineExecuteAsync)(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io, axclrtStream stream) = nullptr;

private:
#if AXCL_PLATFORM_WINDOWS
    using LibHandle = HMODULE;
#else
    using LibHandle = void *;
#endif

    LibHandle handle_ = nullptr;
    std::string loaded_path_;

private:
    static LibHandle open_library(const std::string &path)
    {
#if AXCL_PLATFORM_WINDOWS
        // 使用 ANSI 版本，传入 std::string
        return ::LoadLibraryA(path.c_str());
#else
        return ::dlopen(path.c_str(), RTLD_NOW);
#endif
    }

    static void close_library(LibHandle h)
    {
#if AXCL_PLATFORM_WINDOWS
        ::FreeLibrary(h);
#else
        ::dlclose(h);
#endif
    }

    static std::string last_error_string()
    {
#if AXCL_PLATFORM_WINDOWS
        DWORD err = ::GetLastError();
        if (err == 0)
            return "no error";

        LPSTR buf = nullptr;
        DWORD size = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            err,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR)&buf,
            0,
            nullptr);

        std::string msg = (size && buf) ? std::string(buf, size) : std::string("FormatMessage failed");
        if (buf)
            ::LocalFree(buf);

        // 去掉尾部换行
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n'))
            msg.pop_back();
        return msg;
#else
        const char *e = ::dlerror();
        return e ? std::string(e) : std::string("no error");
#endif
    }

    template <typename T>
    void load_symbol(T &func, const std::string &symbol_name)
    {
#if AXCL_PLATFORM_WINDOWS
        // 清除上一次错误（GetProcAddress 失败不一定会设置 last error，但通常会）
        ::SetLastError(0);

        FARPROC p = ::GetProcAddress(handle_, symbol_name.c_str());
        if (!p)
        {
            func = nullptr;
            std::string err = last_error_string();
            std::fprintf(stderr, "GetProcAddress failed for %s: %s\n", symbol_name.c_str(), err.c_str());
            return;
        }
        func = reinterpret_cast<T>(p);
#else
        ::dlerror(); // 清除错误信息
        func = reinterpret_cast<T>(::dlsym(handle_, symbol_name.c_str()));
        const char *dlsym_error = ::dlerror();
        if (dlsym_error)
        {
            func = nullptr;
            std::fprintf(stderr, "dlsym failed for %s: %s\n", symbol_name.c_str(), dlsym_error);
        }
#endif
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
