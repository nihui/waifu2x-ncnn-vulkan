#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#if _WIN32
#include <windows.h>
#include "win32dirent.h"
#else // _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif // _WIN32

#if _WIN32
typedef std::wstring path_t;
#define PATHSTR(X) L##X
#else
typedef std::string path_t;
#define PATHSTR(X) X
#endif

#if _WIN32
static bool path_is_directory(const path_t& path)
{
    DWORD attr = GetFileAttributesW(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

static int list_directory(const path_t& dirpath, std::vector<path_t>& imagepaths)
{
    imagepaths.clear();

    _WDIR* dir = _wopendir(dirpath.c_str());
    if (!dir)
    {
        fwprintf(stderr, L"opendir failed %ls\n", dirpath.c_str());
        return -1;
    }

    struct _wdirent* ent = 0;
    while ((ent = _wreaddir(dir)))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(path_t(ent->d_name));
    }

    _wclosedir(dir);
    std::sort(imagepaths.begin(), imagepaths.end());

    return 0;
}
#else // _WIN32
static bool path_is_directory(const path_t& path)
{
    struct stat s;
    if (stat(path.c_str(), &s) != 0)
        return false;
    return S_ISDIR(s.st_mode);
}

static int list_directory(const path_t& dirpath, std::vector<path_t>& imagepaths)
{
    imagepaths.clear();

    DIR* dir = opendir(dirpath.c_str());
    if (!dir)
    {
        fprintf(stderr, "opendir failed %s\n", dirpath.c_str());
        return -1;
    }

    struct dirent* ent = 0;
    while ((ent = readdir(dir)))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(path_t(ent->d_name));
    }

    closedir(dir);
    std::sort(imagepaths.begin(), imagepaths.end());

    return 0;
}
#endif // _WIN32

static path_t get_file_extension(const path_t& path)
{
    size_t dot = path.rfind(PATHSTR('.'));
    if (dot == path_t::npos)
        return path_t();

    return path.substr(dot + 1);
}

#endif // FILESYSTEM_UTILS_H
