#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <stdio.h>
#include <vector>
#include <string>

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
static bool path_is_directory(const wchar_t* path)
{
    return GetFileAttributesW(path) & FILE_ATTRIBUTE_DIRECTORY;
}

static int list_directory(const wchar_t* dirpath, std::vector<std::wstring>& imagepaths)
{
    imagepaths.clear();

    _WDIR* dir = _wopendir(dirpath);
    if (!dir)
    {
        fwprintf(stderr, L"opendir failed %s\n", dirpath);
        return -1;
    }

    struct _wdirent* ent = 0;
    while (ent = _wreaddir(dir))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(std::wstring(ent->d_name));
    }

    _wclosedir(dir);

    return 0;
}
#else // _WIN32
static bool path_is_directory(const char* path)
{
    struct stat s;
    stat(path, &s);
    return S_ISDIR(s.st_mode);
}

static int list_directory(const char* dirpath, std::vector<std::string>& imagepaths)
{
    imagepaths.clear();

    DIR* dir = opendir(dirpath);
    if (!dir)
    {
        fprintf(stderr, "opendir failed %s\n", dirpath);
        return -1;
    }

    struct dirent* ent = 0;
    while (ent = readdir(dir))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(std::string(ent->d_name));
    }

    closedir(dir);

    return 0;
}
#endif // _WIN32

#endif // FILESYSTEM_UTILS_H
