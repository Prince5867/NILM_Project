cmake_minimum_required(VERSION 3.10)
project(NILM_Inference)

set(CMAKE_CXX_STANDARD 11)

# Set these paths to where the TFLite libraries/headers are located on your system.
# For example, if you've built TensorFlow Lite for ARM, you might have them installed in /usr/local.
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Add the source file.
add_executable(nilm_inference main.cpp)

# Link against TensorFlow Lite library.
# Ensure that libtensorflow-lite.a or the shared library is present in your link_directories path.
target_link_libraries(nilm_inference tensorflow-lite)

# If you installed nlohmann/json via a package manager, it might be header-only.
# Otherwise, add the include directory accordingly.
