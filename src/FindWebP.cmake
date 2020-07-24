# Copyright (C) 2020 Sony Interactive Entertainment Inc.
# Copyright (C) 2012 Raphael Kubo da Costa <rakuco@webkit.org>
# Copyright (C) 2013 Igalia S.L.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1.  Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
# 2.  Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND ITS CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR ITS
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[=======================================================================[.rst:
FindWebP
--------------

Find WebP headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``webp``
  The WebP library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``WebP_FOUND``
  true if (the requested version of) WebP is available.
``WebP_VERSION``
  the version of WebP.
``WebP_LIBRARIES``
  the libraries to link against to use WebP.
``WebP_INCLUDE_DIRS``
  where to find the WebP headers.
``WebP_COMPILE_OPTIONS``
  this should be passed to target_compile_options(), if the
  target is not used for linking

#]=======================================================================]

find_package(PkgConfig QUIET)
pkg_check_modules(PC_WEBP QUIET libwebp)
set(WebP_COMPILE_OPTIONS ${PC_WEBP_CFLAGS_OTHER})
set(WebP_VERSION ${PC_WEBP_CFLAGS_VERSION})

find_path(WebP_INCLUDE_DIR
    NAMES webp/decode.h
    HINTS ${PC_WEBP_INCLUDEDIR} ${PC_WEBP_INCLUDE_DIRS}
)

find_library(WebP_LIBRARY
    NAMES ${WebP_NAMES} webp
    HINTS ${PC_WEBP_LIBDIR} ${PC_WEBP_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WebP
    FOUND_VAR WebP_FOUND
    REQUIRED_VARS WebP_INCLUDE_DIR WebP_LIBRARY
    VERSION_VAR WebP_VERSION
)

if (WebP_LIBRARY AND NOT TARGET webp)
    add_library(webp UNKNOWN IMPORTED GLOBAL)
    set_target_properties(webp PROPERTIES
        IMPORTED_LOCATION "${WebP_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${WebP_COMPILE_OPTIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${WebP_INCLUDE_DIR}"
    )
endif ()

mark_as_advanced(
    WebP_INCLUDE_DIR
    WebP_LIBRARY
)

if (WebP_FOUND)
    set(WebP_LIBRARIES ${WebP_LIBRARY})
    set(WebP_INCLUDE_DIRS ${WebP_INCLUDE_DIR})
endif ()
