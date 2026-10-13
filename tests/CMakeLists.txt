cmake_minimum_required(VERSION 3.22)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(vinylizer-facial-tests LANGUAGES CXX)
    enable_testing()
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

get_filename_component(VINYLIZER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_executable(pixel_loss_test pixel_loss_test.cpp)
target_compile_features(pixel_loss_test PRIVATE cxx_std_17)
target_include_directories(pixel_loss_test PRIVATE "${VINYLIZER_ROOT}/src")
add_test(NAME pixel_loss COMMAND pixel_loss_test)

add_executable(fitting_math_test fitting_math_test.cpp)
target_compile_features(fitting_math_test PRIVATE cxx_std_17)
target_include_directories(fitting_math_test PRIVATE "${VINYLIZER_ROOT}/src")
add_test(NAME fitting_math COMMAND fitting_math_test)

add_executable(gui_config_test gui_config_test.cpp)
target_compile_features(gui_config_test PRIVATE cxx_std_17)
target_include_directories(gui_config_test PRIVATE "${VINYLIZER_ROOT}/src")
add_test(NAME gui_config COMMAND gui_config_test)

option(VINYLIZER_TEST_CUDA "Test real renderer gradients on a CUDA device" OFF)
if(VINYLIZER_TEST_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(renderer_gradient_test renderer_gradient_test.cu
        "${VINYLIZER_ROOT}/src/core/diff_renderer.cu"
        "${VINYLIZER_ROOT}/src/cuda/render_kernel.cu")
    set_target_properties(renderer_gradient_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(renderer_gradient_test PRIVATE "${VINYLIZER_ROOT}/src")
    target_link_libraries(renderer_gradient_test PRIVATE CUDA::cudart)
    add_test(NAME renderer_gradient COMMAND renderer_gradient_test)
    set_tests_properties(renderer_gradient PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(WIN32)
    add_executable(safety_test safety_test.cpp
        "${VINYLIZER_ROOT}/src/inject/inject.cpp"
        "${VINYLIZER_ROOT}/src/output/json_writer.cpp")
    target_compile_features(safety_test PRIVATE cxx_std_17)
    target_include_directories(safety_test PRIVATE "${VINYLIZER_ROOT}/src" "${VINYLIZER_ROOT}/vendor")
    add_test(NAME safety COMMAND safety_test)
endif()

if(VINYLIZER_ENABLE_GUI)
    add_executable(gui_preview_test gui_preview_test.cpp
        "${VINYLIZER_ROOT}/src/gui/json_preview.cpp"
        "${VINYLIZER_ROOT}/src/common/canvas.cpp"
        "${VINYLIZER_ROOT}/src/cuda/canvas_render.cu"
        "${VINYLIZER_ROOT}/src/inject/inject.cpp")
    target_compile_features(gui_preview_test PRIVATE cxx_std_17)
    set_target_properties(gui_preview_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(gui_preview_test PRIVATE
        "${VINYLIZER_ROOT}/src" "${VINYLIZER_ROOT}/vendor" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(gui_preview_test PRIVATE Qt6::Gui opencv_core CUDA::cudart)
    add_test(NAME gui_preview COMMAND gui_preview_test)
    set_tests_properties(gui_preview PROPERTIES SKIP_RETURN_CODE 77)
endif()

option(VINYLIZER_TEST_OPENCV "Test facial maps, weights, and cascade assets" ON)
if(VINYLIZER_TEST_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc objdetect)
    add_executable(facial_saliency_test
        facial_saliency_test.cpp
        "${VINYLIZER_ROOT}/src/preprocess/facial_saliency.cpp"
    )
    target_compile_features(facial_saliency_test PRIVATE cxx_std_17)
    target_include_directories(facial_saliency_test PRIVATE "${VINYLIZER_ROOT}/src" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(facial_saliency_test PRIVATE ${OpenCV_LIBS})
    add_executable(image_input_test image_input_test.cpp)
    target_compile_features(image_input_test PRIVATE cxx_std_17)
    target_include_directories(image_input_test PRIVATE "${VINYLIZER_ROOT}/src" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(image_input_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME image_input COMMAND image_input_test)
    add_custom_command(TARGET facial_saliency_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VINYLIZER_ROOT}/assets/haarcascades"
            "$<TARGET_FILE_DIR:facial_saliency_test>/assets/haarcascades"
        VERBATIM
    )
    add_test(NAME facial_saliency COMMAND facial_saliency_test
        "$<TARGET_FILE_DIR:facial_saliency_test>/assets/haarcascades")
endif()
