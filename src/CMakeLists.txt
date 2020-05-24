project(waifu2x-ncnn-vulkan)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)

option(USE_STATIC_MOLTENVK "link moltenvk static library" OFF)

find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)

find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    # int8 storage
    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# build ncnn library
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
endif()

option(NCNN_INSTALL_SDK "" OFF)
option(NCNN_PIXEL_ROTATE "" OFF)
option(NCNN_VULKAN "" ON)
option(NCNN_VULKAN_ONLINE_SPIRV "" ON)
option(NCNN_BUILD_BENCHMARK "" OFF)
option(NCNN_BUILD_TESTS "" OFF)
option(NCNN_BUILD_TOOLS "" OFF)
option(NCNN_BUILD_EXAMPLES "" OFF)
option(NCNN_DISABLE_RTTI "" ON)
option(NCNN_DISABLE_EXCEPTION "" ON)

option(WITH_LAYER_absval "" OFF)
option(WITH_LAYER_argmax "" OFF)
option(WITH_LAYER_batchnorm "" OFF)
option(WITH_LAYER_bias "" OFF)
option(WITH_LAYER_bnll "" OFF)
option(WITH_LAYER_concat "" OFF)
option(WITH_LAYER_convolution "" ON)
option(WITH_LAYER_crop "" ON)
option(WITH_LAYER_deconvolution "" ON)
option(WITH_LAYER_dropout "" OFF)
option(WITH_LAYER_eltwise "" ON)
option(WITH_LAYER_elu "" OFF)
option(WITH_LAYER_embed "" OFF)
option(WITH_LAYER_exp "" OFF)
option(WITH_LAYER_flatten "" ON)
option(WITH_LAYER_innerproduct "" ON)
option(WITH_LAYER_input "" ON)
option(WITH_LAYER_log "" OFF)
option(WITH_LAYER_lrn "" OFF)
option(WITH_LAYER_memorydata "" OFF)
option(WITH_LAYER_mvn "" OFF)
option(WITH_LAYER_pooling "" ON)
option(WITH_LAYER_power "" OFF)
option(WITH_LAYER_prelu "" OFF)
option(WITH_LAYER_proposal "" OFF)
option(WITH_LAYER_reduction "" OFF)
option(WITH_LAYER_relu "" ON)
option(WITH_LAYER_reshape "" OFF)
option(WITH_LAYER_roipooling "" OFF)
option(WITH_LAYER_scale "" ON)
option(WITH_LAYER_sigmoid "" OFF)
option(WITH_LAYER_slice "" OFF)
option(WITH_LAYER_softmax "" OFF)
option(WITH_LAYER_split "" ON)
option(WITH_LAYER_spp "" OFF)
option(WITH_LAYER_tanh "" OFF)
option(WITH_LAYER_threshold "" OFF)
option(WITH_LAYER_tile "" OFF)
option(WITH_LAYER_rnn "" OFF)
option(WITH_LAYER_lstm "" OFF)
option(WITH_LAYER_binaryop "" OFF)
option(WITH_LAYER_unaryop "" OFF)
option(WITH_LAYER_convolutiondepthwise "" OFF)
option(WITH_LAYER_padding "" ON)
option(WITH_LAYER_squeeze "" OFF)
option(WITH_LAYER_expanddims "" OFF)
option(WITH_LAYER_normalize "" OFF)
option(WITH_LAYER_permute "" OFF)
option(WITH_LAYER_priorbox "" OFF)
option(WITH_LAYER_detectionoutput "" OFF)
option(WITH_LAYER_interp "" ON)
option(WITH_LAYER_deconvolutiondepthwise "" OFF)
option(WITH_LAYER_shufflechannel "" OFF)
option(WITH_LAYER_instancenorm "" OFF)
option(WITH_LAYER_clip "" OFF)
option(WITH_LAYER_reorg "" OFF)
option(WITH_LAYER_yolodetectionoutput "" OFF)
option(WITH_LAYER_quantize "" OFF)
option(WITH_LAYER_dequantize "" OFF)
option(WITH_LAYER_yolov3detectionoutput "" OFF)
option(WITH_LAYER_psroipooling "" OFF)
option(WITH_LAYER_roialign "" OFF)
option(WITH_LAYER_packing "" ON)
option(WITH_LAYER_requantize "" OFF)
option(WITH_LAYER_cast "" ON)
option(WITH_LAYER_hardsigmoid "" OFF)
option(WITH_LAYER_selu "" OFF)
option(WITH_LAYER_hardswish "" OFF)
option(WITH_LAYER_noop "" OFF)
option(WITH_LAYER_pixelshuffle "" OFF)
option(WITH_LAYER_deepcopy "" OFF)
option(WITH_LAYER_mish "" OFF)
option(WITH_LAYER_statisticspooling "" OFF)

add_subdirectory(ncnn)

# look for vulkan compute shader and compile
set(SHADER_SPV_HEX_FILES)

compile_shader(waifu2x_preproc.comp)
compile_shader(waifu2x_postproc.comp)
compile_shader(waifu2x_preproc_tta.comp)
compile_shader(waifu2x_postproc_tta.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_executable(waifu2x-ncnn-vulkan main.cpp waifu2x.cpp)

add_dependencies(waifu2x-ncnn-vulkan generate-spirv)

set(WAIFU2X_LINK_LIBRARIES ncnn ${Vulkan_LIBRARY})

if(USE_STATIC_MOLTENVK)
    find_library(CoreFoundation NAMES CoreFoundation)
    find_library(Foundation NAMES Foundation)
    find_library(Metal NAMES Metal)
    find_library(QuartzCore NAMES QuartzCore)
    find_library(CoreGraphics NAMES CoreGraphics)
    find_library(Cocoa NAMES Cocoa)
    find_library(IOKit NAMES IOKit)
    find_library(IOSurface NAMES IOSurface)

    list(APPEND WAIFU2X_LINK_LIBRARIES
        ${Metal}
        ${QuartzCore}
        ${CoreGraphics}
        ${Cocoa}
        ${IOKit}
        ${IOSurface}
        ${Foundation}
        ${CoreFoundation}
    )
endif()

if(OPENMP_FOUND)
    list(APPEND WAIFU2X_LINK_LIBRARIES ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(waifu2x-ncnn-vulkan ${WAIFU2X_LINK_LIBRARIES})
