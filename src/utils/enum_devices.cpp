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

bool parse_axcl_smi_output(FILE *fp, ax_devices_t &out)
{
    if (!fp)
        return false;

    char line[512];
    std::vector<std::string> lines;

    while (fgets(line, sizeof(line), fp))
    {
        lines.emplace_back(line);
    }

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
    FILE *fp = popen("axcl-smi", "r");
    if (!fp)
    {
        std::cerr << "Failed to run axcl-smi." << std::endl;
        return false;
    }

    bool success = parse_axcl_smi_output(fp, *info);
    pclose(fp);
    return success;
}

static std::vector<std::string> v_libax_sys_so_path = {
    "/soc/lib/libax_sys.so",
    "/opt/lib/libax_sys.so",
    "/usr/lib/libax_sys.so"};

static std::string exec_cmd(std::string cmd)
{
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        return "";
    }
    char buffer[128];
    std::string result = "";
    while (!feof(pipe))
    {
        if (fgets(buffer, 128, pipe) != NULL)
        {
            result += buffer;
        }
    }
    pclose(pipe);
    return result;
}

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