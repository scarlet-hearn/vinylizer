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
    add_custom_command(TARGET facial_saliency_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VINYLIZER_ROOT}/assets/haarcascades"
            "$<TARGET_FILE_DIR:facial_saliency_test>/assets/haarcascades"
        VERBATIM
    )
    add_test(NAME facial_saliency COMMAND facial_saliency_test
        "$<TARGET_FILE_DIR:facial_saliency_test>/assets/haarcascades")
endif()
