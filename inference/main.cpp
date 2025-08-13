#include <iostream>
#include <memory>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

int main() {
    const char* model_path = "model.tflite";

    // Load TFLite model from file
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return -1;
    }

    // Build the interpreter with the built-in ops resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return -1;
    }

    // (Optional) Print interpreter capacity constants to ensure compatibility
    std::cout << "Reserved capacity: "
              << tflite::Interpreter::kTensorsReservedCapacity << std::endl;
    std::cout << "Capacity headroom: "
              << tflite::Interpreter::kTensorsCapacityHeadroom << std::endl;

    // Allocate memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }

    // Get input tensor pointer and size
    float* input = interpreter->typed_input_tensor<float>(0);
    int input_size = interpreter->input_tensor(0)->bytes / sizeof(float);

    // Fill input tensor with dummy data (zeros)
    std::fill(input, input + input_size, 0.0f);

    // (Optional) Print interpreter state before running
    // tflite::PrintInterpreterState(interpreter.get());

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return -1;
    }

    // Get output tensor pointer and size
    float* output = interpreter->typed_output_tensor<float>(0);
    int output_size = interpreter->output_tensor(0)->bytes / sizeof(float);

    // Print output values
    std::cout << "Output tensor values:" << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
