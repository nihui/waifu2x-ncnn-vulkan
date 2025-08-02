
option(USE_SYSTEM_JPEG "build with system libjpeg" OFF)
option(USE_SYSTEM_ZLIB "build with system zlib" OFF)
option(USE_SYSTEM_PNG "build with system libpng" OFF)
option(USE_SYSTEM_WEBP "build with system libwebp" OFF)

if(USE_SYSTEM_JPEG)
    find_package(JPEG)
    if(NOT TARGET JPEG::JPEG)
        message(WARNING "jpeg target not found! USE_SYSTEM_JPEG will be turned off.")
        set(USE_SYSTEM_JPEG OFF)
    endif()
endif()

if(USE_SYSTEM_ZLIB)
    find_package(ZLIB)
    if(NOT TARGET ZLIB::ZLIB)
        message(WARNING "zlib target not found! USE_SYSTEM_ZLIB will be turned off.")
        set(USE_SYSTEM_ZLIB OFF)
    endif()
endif()

if(USE_SYSTEM_PNG)
    find_package(PNG)
    if(NOT TARGET PNG::PNG)
        message(WARNING "png target not found! USE_SYSTEM_PNG will be turned off.")
        set(USE_SYSTEM_PNG OFF)
    endif()
endif()

if(USE_SYSTEM_WEBP)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
    find_package(WebP)
    if(NOT TARGET webp)
        message(WARNING "webp target not found! USE_SYSTEM_WEBP will be turned off.")
        set(USE_SYSTEM_WEBP OFF)
    endif()
endif()

if(NOT USE_SYSTEM_JPEG)
    # build libjpeg-turbo library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    include(ExternalProject)

    ExternalProject_Add(deps-libjpeg-turbo
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps-install"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DENABLE_SHARED=OFF
            -DENABLE_STATIC=ON
            -DWITH_TURBOJPEG=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
    )

    add_dependencies(waifu2x-ncnn-vulkan deps-libjpeg-turbo)

    target_include_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_INCLUDEDIR}")
    target_link_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_LIBDIR}")
    if(WIN32)
        set(JPEG_LIBRARIES jpeg-static)
    else()
        set(JPEG_LIBRARIES jpeg)
    endif()
endif()

if(NOT USE_SYSTEM_ZLIB)
    # build zlib library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zlib-ng/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    include(ExternalProject)

    ExternalProject_Add(deps-zlib-ng
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zlib-ng"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/zlib-ng"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps-install"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DBUILD_SHARED_LIBS=OFF
            -DWITH_GZFILEOP=OFF
            -DZLIB_COMPAT=ON
            -DZLIB_ENABLE_TESTS=OFF
            -DZLIBNG_ENABLE_TESTS=OFF
            -DWITH_GTEST=OFF
            -DWITH_FUZZERS=OFF
            -DWITH_BENCHMARKS=OFF
            -DWITH_OPTIM=ON
            -DWITH_REDUCED_MEM=OFF
            -DWITH_NEW_STRATEGIES=ON
            -DWITH_RUNTIME_CPU_DETECTION=ON
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
    )

    add_dependencies(waifu2x-ncnn-vulkan deps-zlib-ng)

    target_include_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_INCLUDEDIR}")
    target_link_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_LIBDIR}")
    if(WIN32)
        set(ZLIB_LIBRARIES zlibstatic)
    else()
        set(ZLIB_LIBRARIES z)
    endif()
endif()

if(NOT USE_SYSTEM_PNG)
    # build libpng library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libpng/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    include(ExternalProject)

    if(USE_SYSTEM_ZLIB)
        ExternalProject_Add(deps-libpng
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libpng"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libpng"
            INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps-install"
            CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                -DPNG_SHARED=OFF
                -DPNG_STATIC=ON
                -DPNG_TESTS=OFF
                -DPNG_TOOLS=OFF
                -DPNG_HARDWARE_OPTIMIZATIONS=ON
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
        )
    else()
        ExternalProject_Add(deps-libpng
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libpng"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libpng"
            INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps-install"
            CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                -DPNG_SHARED=OFF
                -DPNG_STATIC=ON
                -DPNG_TESTS=OFF
                -DPNG_TOOLS=OFF
                -DPNG_HARDWARE_OPTIMIZATIONS=ON
                -DZLIB_ROOT=<INSTALL_DIR>
                -DZLIB_USE_STATIC_LIBS=TRUE
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
        )

        add_dependencies(deps-libpng deps-zlib-ng)
    endif()

    add_dependencies(waifu2x-ncnn-vulkan deps-libpng)

    target_include_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_INCLUDEDIR}")
    target_link_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_LIBDIR}")
    if(WIN32)
        set(PNG_LIBRARIES libpng16_static)
    else()
        set(PNG_LIBRARIES png)
    endif()
endif()

if(NOT USE_SYSTEM_WEBP)
    # build libwebp library
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libwebp/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    include(ExternalProject)

    ExternalProject_Add(deps-libwebp
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libwebp"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libwebp"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps-install"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DBUILD_SHARED_LIBS=OFF
            -DWEBP_ENABLE_SIMD=ON
            -DWEBP_BUILD_ANIM_UTILS=OFF
            -DWEBP_BUILD_CWEBP=OFF
            -DWEBP_BUILD_DWEBP=OFF
            -DWEBP_BUILD_GIF2WEBP=OFF
            -DWEBP_BUILD_IMG2WEBP=OFF
            -DWEBP_BUILD_VWEBP=OFF
            -DWEBP_BUILD_WEBPINFO=OFF
            -DWEBP_BUILD_WEBPMUX=OFF
            -DWEBP_BUILD_EXTRAS=OFF
            -DWEBP_BUILD_WEBP_JS=OFF
            -DWEBP_NEAR_LOSSLESS=OFF
            -DWEBP_ENABLE_SWAP_16BIT_CSP=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
    )

    add_dependencies(waifu2x-ncnn-vulkan deps-libwebp)

    target_include_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_INCLUDEDIR}")
    target_link_directories(waifu2x-ncnn-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/deps-install/${CMAKE_INSTALL_LIBDIR}")
    if(WIN32)
        set(WebP_LIBRARIES libwebp libwebpdecoder libwebpdemux libwebpmux libsharpyuv)
    else()
        set(WebP_LIBRARIES webp webpdecoder webpdemux webpmux sharpyuv)
    endif()
endif()

set(DEPS_CODEC_LIBRARIES ${WebP_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
