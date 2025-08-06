#include <iostream>
#include <memory>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

int main() {
    const char* model_path = "model.tflite";

    // Load TFLite model from file
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return -1;
    }

    // Build the interpreter with the built-in ops resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return -1;
    }

    // Allocate memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }

    // Get input tensor pointer and size
    float* input = interpreter->typed_input_tensor<float>(0);
    int input_size = interpreter->input_tensor(0)->bytes / sizeof(float);

    // Fill input tensor with dummy data (zeroes)
    for (int i = 0; i < input_size; i++) {
        input[i] = 0.0f;
    }

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
    for (int i = 0; i < output_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
