#pragma once
#include <string>
#include <cstdio>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#define MMAP_PLATFORM_WINDOWS 1
#include <windows.h>
#else
#define MMAP_PLATFORM_WINDOWS 0
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#endif

class MMap
{
private:
    void *_add = nullptr;
    size_t _size = 0;

#if MMAP_PLATFORM_WINDOWS
    HANDLE _hFile = NULL;
    HANDLE _hMapping = NULL;
#endif

public:
    MMap() = default;

    explicit MMap(const char *file)
    {
        open_file(file);
    }

    ~MMap()
    {
        close_file();
    }

    bool open_file(const char *file)
    {
        close_file(); // 防止重复 open 泄露
        return _mmap(file);
    }

    void close_file()
    {
#if MMAP_PLATFORM_WINDOWS
        if (_add)
        {
            ::UnmapViewOfFile(_add);
            _add = nullptr;
        }
        if (_hMapping)
        {
            ::CloseHandle(_hMapping);
            _hMapping = NULL;
        }
        if (_hFile && _hFile != INVALID_HANDLE_VALUE)
        {
            ::CloseHandle(_hFile);
            _hFile = NULL;
        }
        _size = 0;
#else
        if (_add)
        {
            ::munmap(_add, _size);
            _add = nullptr;
            _size = 0;
        }
#endif
    }

    size_t size() const
    {
        return _size;
    }

    void *data() const
    {
        return _add;
    }

private:
#if MMAP_PLATFORM_WINDOWS
    static std::string win_last_error()
    {
        DWORD err = ::GetLastError();
        if (err == 0)
            return "no error";
        LPSTR buf = nullptr;
        DWORD size = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR)&buf, 0, nullptr);
        std::string msg = (size && buf) ? std::string(buf, size) : std::string("FormatMessage failed");
        if (buf)
            ::LocalFree(buf);
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n'))
            msg.pop_back();
        return msg;
    }
#endif

    bool _mmap(const char *model_file)
    {
#if MMAP_PLATFORM_WINDOWS
        _hFile = ::CreateFileA(
            model_file,
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (_hFile == INVALID_HANDLE_VALUE)
        {
            std::fprintf(stderr, "[MMap] CreateFileA failed for %s: %s\n", model_file, win_last_error().c_str());
            _hFile = NULL;
            return false;
        }

        LARGE_INTEGER fileSize;
        if (!::GetFileSizeEx(_hFile, &fileSize))
        {
            std::fprintf(stderr, "[MMap] GetFileSizeEx failed for %s: %s\n", model_file, win_last_error().c_str());
            ::CloseHandle(_hFile);
            _hFile = NULL;
            return false;
        }

        if (fileSize.QuadPart <= 0)
        {
            std::fprintf(stderr, "[MMap] file size is zero for %s\n", model_file);
            ::CloseHandle(_hFile);
            _hFile = NULL;
            return false;
        }

        _size = static_cast<size_t>(fileSize.QuadPart);

        _hMapping = ::CreateFileMappingA(_hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!_hMapping)
        {
            std::fprintf(stderr, "[MMap] CreateFileMappingA failed for %s: %s\n", model_file, win_last_error().c_str());
            ::CloseHandle(_hFile);
            _hFile = NULL;
            return false;
        }

        _add = ::MapViewOfFile(_hMapping, FILE_MAP_READ, 0, 0, 0);
        if (!_add)
        {
            std::fprintf(stderr, "[MMap] MapViewOfFile failed for %s: %s\n", model_file, win_last_error().c_str());
            ::CloseHandle(_hMapping);
            _hMapping = NULL;
            ::CloseHandle(_hFile);
            _hFile = NULL;
            _size = 0;
            return false;
        }

        return true;

#else
        int fd = ::open(model_file, O_RDONLY);
        if (fd < 0)
        {
            std::fprintf(stderr, "[MMap] open failed for file %s: %s\n", model_file, std::strerror(errno));
            return false;
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0)
        {
            std::fprintf(stderr, "[MMap] fstat failed for file %s: %s\n", model_file, std::strerror(errno));
            ::close(fd);
            return false;
        }

        if (st.st_size <= 0)
        {
            std::fprintf(stderr, "[MMap] file size is zero for %s\n", model_file);
            ::close(fd);
            return false;
        }

        _size = static_cast<size_t>(st.st_size);

        void *addr = ::mmap(nullptr, _size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);

        if (addr == MAP_FAILED)
        {
            std::fprintf(stderr, "[MMap] mmap failed for file %s: %s\n", model_file, std::strerror(errno));
            _size = 0;
            return false;
        }

        _add = addr;
        return true;
#endif
    }

public:
    // 如果你还想保留原来的 static _mmap 接口（返回指针 + size）
    // 这里也给一个跨平台版本（Windows 下不会暴露句柄，所以不建议外部单独用它来长期持有）
    static void *_mmap_oneshot_readonly(const char *model_file, size_t *model_size)
    {
        if (!model_size)
            return nullptr;
        MMap mm;
        if (!mm.open_file(model_file))
            return nullptr;
        *model_size = mm.size();
        // ⚠️ 这个函数返回的指针依赖 mm 生命周期，所以不安全。
        // 如果你一定要 static 返回指针，请改接口：同时返回/输出句柄，并提供对应 unmap。
        return mm.data();
    }
};