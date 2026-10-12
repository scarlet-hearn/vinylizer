cmake_minimum_required(VERSION 3.22)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(vinylizer-facial-tests LANGUAGES CXX)
    enable_testing()
endif()

get_filename_component(VINYLIZER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_executable(pixel_loss_test pixel_loss_test.cpp)
target_compile_features(pixel_loss_test PRIVATE cxx_std_17)
target_include_directories(pixel_loss_test PRIVATE "${VINYLIZER_ROOT}/src")
add_test(NAME pixel_loss COMMAND pixel_loss_test)

if(WIN32)
    add_executable(safety_test safety_test.cpp
        "${VINYLIZER_ROOT}/src/inject/inject.cpp"
        "${VINYLIZER_ROOT}/src/output/json_writer.cpp")
    target_compile_features(safety_test PRIVATE cxx_std_17)
    target_include_directories(safety_test PRIVATE "${VINYLIZER_ROOT}/src" "${VINYLIZER_ROOT}/vendor")
    add_test(NAME safety COMMAND safety_test)
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
