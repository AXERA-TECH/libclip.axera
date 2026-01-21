#include "enum_devices.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <regex>
#include <cstring> // For strncpy

#if defined(_WIN32) || defined(_WIN64)
#define AX_POPEN _popen
#define AX_PCLOSE _pclose
#else
#define AX_POPEN popen
#define AX_PCLOSE pclose
#endif

static inline void trim_inplace(std::string &s)
{
    // 去掉尾部 \r\n 和空白
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || std::isspace((unsigned char)s.back())))
        s.pop_back();

    // 去掉头部空白
    size_t i = 0;
    while (i < s.size() && std::isspace((unsigned char)s[i]))
        ++i;
    if (i > 0)
        s.erase(0, i);
}

static std::vector<std::string> exec_cmd_lines(const std::string &cmd)
{
    using PipeCloser = int (*)(FILE *);
    std::unique_ptr<FILE, PipeCloser> pipe(AX_POPEN(cmd.c_str(), "r"), AX_PCLOSE);
    if (!pipe)
        return {};

    std::array<char, 4096> buf{};
    std::vector<std::string> lines;

    while (std::fgets(buf.data(), (int)buf.size(), pipe.get()))
    {
        std::string line(buf.data());
        trim_inplace(line);
        if (!line.empty())
            lines.push_back(std::move(line));
    }
    return lines;
}

static std::string exec_cmd(std::string cmd)
{
    auto lines = exec_cmd_lines(cmd);
    std::stringstream ss;
    for (const auto &line : lines)
    {
        ss << line << '\n';
    }
    return ss.str();
}

bool parse_axcl_smi_output(std::vector<std::string> &lines, ax_devices_t &out)
{
    if (lines.empty())
        return false;

    if (lines.size() < 5)
        return false;

    // 提取 host + driver 版本号（第2行）
    std::regex version_regex(R"(AXCL-SMI\s+(V[^\s]+)\s+Driver\s+(V[^\s]+))");
    std::smatch match;
    if (std::regex_search(lines[1], match, version_regex))
    {
        strncpy(out.devices.host_version, match[1].str().c_str(), sizeof(out.devices.host_version) - 1);
        strncpy(out.devices.dev_version, match[2].str().c_str(), sizeof(out.devices.dev_version) - 1);
    }
    else
    {
        return false;
    }

    int device_index = 0;
    for (size_t i = 0; i + 1 < lines.size(); ++i)
    {
        if (lines[i].find("|    ") == 0 && lines[i + 1].find("|   --") == 0)
        {
            const std::string &status_line = lines[i + 1];

            // 示例行：
            // |   --   61C                      -- / -- | 1%        0% | 18 MiB /     7040 MiB |

            std::regex stat_regex(R"(\|\s+--\s+(\d+)C\s+-- / --\s+\|\s+(\d+)%\s+(\d+)%\s+\|\s+(\d+)\s+MiB\s+/\s+(\d+)\s+MiB\s+\|)");
            std::smatch sm;
            if (std::regex_search(status_line, sm, stat_regex))
            {
                int temp = std::stoi(sm[1]);
                int cpu_usage = std::stoi(sm[2]);
                int npu_usage = std::stoi(sm[3]);
                int used_mem = std::stoi(sm[4]);
                int total_mem = std::stoi(sm[5]);

                if (device_index < 16)
                {
                    auto &dev = out.devices.devices_info[device_index];
                    dev.temp = temp;
                    dev.cpu_usage = cpu_usage;
                    dev.npu_usage = npu_usage;
                    dev.mem_info.total = total_mem;
                    dev.mem_info.remain = total_mem - used_mem;
                    ++device_index;
                }
            }
        }
    }

    out.devices.count = device_index;
    return true;
}

bool get_axcl_devices(ax_devices_t *info)
{
    std::vector<std::string> cmds = {"axcl-smi",
                                     "C:\\Program Files\\AXCL\\axcl\\out\\axcl_win_x64\\bin\\axcl-smi.exe"};
    for (const auto &cmd : cmds)
    {
        std::vector<std::string> lines = exec_cmd_lines(cmd);
        bool success = parse_axcl_smi_output(lines, *info);
        if (success)
        {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> v_libax_sys_so_path = {
    "/soc/lib/libax_sys.so",
    "/opt/lib/libax_sys.so",
    "/usr/lib/libax_sys.so"};

static bool file_exists(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

static bool get_board_info(ax_devices_t *info)
{
    // std::string cmd = "strings ${BSP_MSP_DIR}/lib/libax_sys.so | grep 'Axera version' | awk '{print $4}'";
    char cmd[128] = {0};
    for (size_t i = 0; i < v_libax_sys_so_path.size(); i++)
    {
        if (!file_exists(v_libax_sys_so_path[i]))
        {
            continue;
        }
        sprintf(cmd, "strings %s | grep 'Axera version' | awk '{print $4}'", v_libax_sys_so_path[i].c_str());
        std::string version = exec_cmd(cmd);
        if (!version.empty())
        {
            version = version.substr(0, version.size() - 1);
            info->host.available = 1;
            strncpy(info->host.version, version.c_str(), sizeof(info->host.version) - 1);
            std::string mem_info = exec_cmd("cat /proc/ax_proc/mem_cmm_info |grep \"total size\"");

            std::regex pattern(R"(total size=\d+KB\((\d+)MB\).*?remain=\d+KB\((\d+)MB)");
            std::smatch match;

            if (std::regex_search(mem_info, match, pattern))
            {
                info->host.mem_info.total = std::stoi(match[1].str());
                info->host.mem_info.remain = std::stoi(match[2].str());
            }
            return true;
        }
    }
    return false;
}

bool get_host_info(ax_devices_t *info)
{
    std::string version;
    if (get_board_info(info))
    {
        return true;
    }
    return false;
}