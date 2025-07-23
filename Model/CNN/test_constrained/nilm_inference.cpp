#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>

// TensorFlow Lite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter_builder.h"

// For JSON parsing - make sure to add nlohmann/json to your include path.
// #include "nlohmann/json.hpp"

// Use the nlohmann namespace for JSON.
using json = nlohmann::json;

// -------------------------------
// CSV reading utility
// Reads a CSV file where each line contains comma-separated floats.
// Assumes no header. Each row is returned as a vector of floats.
std::vector<std::vector<float>> read_csv(const std::string &filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open CSV file: " << filename << "\n";
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        data.push_back(row);
    }
    return data;
}

// -------------------------------
// Scaling function using parameters loaded from JSON.
// Here, 'scalerMin' is the offset (corresponding to "min_" in the JSON)
// and 'scaleFactor' corresponds to "scale_".
// The transformation applied is: output = (input - data_min) * scale + scalerMin.
void scale_data(std::vector<float>& sample, const std::vector<float>& data_min, const std::vector<float>& scaleFactor, const std::vector<float>& scalerMin) {
    assert(sample.size() == data_min.size() && sample.size() == scaleFactor.size() && sample.size() == scalerMin.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        sample[i] = (sample[i] - data_min[i]) * scaleFactor[i] + scalerMin[i];
    }
}

// -------------------------------
// Compute Explained Variance Score
float explained_variance_score(const std::vector<float>& y_true, const std::vector<float>& y_pred) {
    assert(y_true.size() == y_pred.size());
    // Compute mean of y_true.
    float mean_y = std::accumulate(y_true.begin(), y_true.end(), 0.0f) / y_true.size();
    float ss_total = 0.0f;
    float ss_res = 0.0f;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_total += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    // Explained variance score: 1 - (variance of residuals) / (variance of target).
    return 1.0f - (ss_res / ss_total);
}

// -------------------------------
// Main function
int main() {

    // ***** STEP 1: Load JSON scaler parameters *****
    // Expecting a file "scaler_params.json" with keys "data_min", "scale_" and "min_".
    // Example format:
    // {
    //    "data_min": [0.0, 0.0, ...],
    //    "scale_": [0.5, 0.3, ...],
    //    "min_": [-1.0, -1.0, ...]
    // }
   std::vector<float> parse_array_from_json_line(const std::string& line) {
    std::vector<float> result;
    size_t start = line.find('[');
    size_t end = line.find(']');
    if (start == std::string::npos || end == std::string::npos || end <= start)
        return result;

    std::string array_str = line.substr(start + 1, end - start - 1);
    std::stringstream ss(array_str);
    std::string value;
    while (std::getline(ss, value, ',')) {
        result.push_back(std::stof(value));
    }
    return result;
    }

    std::vector<float> data_min, scaleFactor, scalerMin;
    std::ifstream json_file("scaler_params.json");
    std::string line;
    while (std::getline(json_file, line)) {
        if (line.find("\"data_min\"") != std::string::npos) {
            data_min = parse_array_from_json_line(line);
        } else if (line.find("\"scale_\"") != std::string::npos) {
            scaleFactor = parse_array_from_json_line(line);
        } else if (line.find("\"min_\"") != std::string::npos) {
            scalerMin = parse_array_from_json_line(line);
        }
    }


    // ***** STEP 2: Load CSV Data *****
    // CSV file ("test_data.csv") with rows: feature1, feature2, ..., featureN, label
    auto csvData = read_csv("test_data.csv");
    if (csvData.empty()) {
        std::cerr << "CSV data empty or cannot be read.\n";
        return -1;
    }
    
    // Separate features and ground truth labels.
    // Assume each row: first N values are features and the last value is the ground truth.
    size_t featureCount = csvData[0].size() - 1;
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    for (const auto& row : csvData) {
        if (row.size() < 2) continue; // skip bad rows
        std::vector<float> feat(row.begin(), row.begin() + featureCount);
        features.push_back(feat);
        labels.push_back(row.back());
    }
    
    // ***** STEP 3: Set up TFLite Interpreter *****
    // Load the TFLite model from file.
    auto model = tflite::FlatBufferModel::BuildFromFile("NILM_model.tflite");
    if (!model) {
        std::cerr << "Failed to load TFLite model\n";
        return -1;
    }
    
    // Build the interpreter.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return -1;
    }
    
    // Allocate tensor buffers.
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!\n";
        return -1;
    }
    
    // Get input tensor info.
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int inputBatch = dims->data[0];
    int inputSize  = dims->data[1]; // number of features expected by the model

    if (inputSize != static_cast<int>(featureCount)) {
        std::cerr << "Mismatch in feature count: CSV (" << featureCount << ") vs Model (" << inputSize << ")\n";
        return -1;
    }
    
    // ***** STEP 4: Inference Loop *****
    std::vector<float> predictions;
    for (const auto& sample : features) {
        // Copy sample data into a mutable vector (since we are scaling it)
        std::vector<float> scaled_sample = sample;
        // Scale the sample (using the JSON parameters)
        scale_data(scaled_sample, data_min, scaleFactor, scalerMin);
        
        // Check input tensor size. We assume the model expects a single sample per inference.
        float* input_tensor = interpreter->typed_tensor<float>(input);
        for (int i = 0; i < inputSize; ++i) {
            input_tensor[i] = scaled_sample[i];
        }
        
        // Run inference.
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Error during inference\n";
            continue;
        }
        
        // Assume output tensor is a single float value. Adjust if your model outputs a different shape.
        int output_idx = interpreter->outputs()[0];
        float* output = interpreter->typed_tensor<float>(output_idx);
        predictions.push_back(output[0]);
    }
    
    // ***** STEP 5: Compute and Output the Explained Variance Score *****
    float evs = explained_variance_score(labels, predictions);
    std::cout << "Explained Variance Score: " << evs << "\n";
    
    return 0;
}
