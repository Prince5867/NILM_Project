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

// JSON parser
#include "json.hpp"
using json = nlohmann::json;

// -------------------------------
// CSV reading utility
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
// Scale input data using scaler parameters
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
    float mean_y = std::accumulate(y_true.begin(), y_true.end(), 0.0f) / y_true.size();
    float ss_total = 0.0f;
    float ss_res = 0.0f;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_total += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    return 1.0f - (ss_res / ss_total);
}

// -------------------------------
// Main
int main() {
    // STEP 1: Load scaler parameters using nlohmann::json
    std::ifstream json_file("scaler_params.json");
    if (!json_file.is_open()) {
        std::cerr << "Cannot open scaler_params.json\n";
        return -1;
    }

    json j;
    json_file >> j;

    std::vector<float> data_min    = j["data_min"].get<std::vector<float>>();
    std::vector<float> scaleFactor = j["scale_"].get<std::vector<float>>();
    std::vector<float> scalerMin   = j["min_"].get<std::vector<float>>();

    // STEP 2: Load CSV test data
    auto csvData = read_csv("test_data.csv");
    if (csvData.empty()) {
        std::cerr << "CSV data empty or cannot be read.\n";
        return -1;
    }

    size_t featureCount = csvData[0].size() - 1;
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    for (const auto& row : csvData) {
        if (row.size() < 2) continue;
        features.emplace_back(row.begin(), row.begin() + featureCount);
        labels.push_back(row.back());
    }

    // STEP 3: Load TFLite model
    auto model = tflite::FlatBufferModel::BuildFromFile("NILM_model.tflite");
    if (!model) {
        std::cerr << "Failed to load TFLite model\n";
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!\n";
        return -1;
    }

    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int inputBatch = dims->data[0];
    int inputSize  = dims->data[1];

    if (inputSize != static_cast<int>(featureCount)) {
        std::cerr << "Mismatch in feature count: CSV (" << featureCount << ") vs Model (" << inputSize << ")\n";
        return -1;
    }

    // STEP 4: Run inference on each sample
    std::vector<float> predictions;
    for (const auto& sample : features) {
        std::vector<float> scaled_sample = sample;
        scale_data(scaled_sample, data_min, scaleFactor, scalerMin);

        float* input_tensor = interpreter->typed_tensor<float>(input);
        for (int i = 0; i < inputSize; ++i) {
            input_tensor[i] = scaled_sample[i];
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Error during inference\n";
            continue;
        }

        int output_idx = interpreter->outputs()[0];
        float* output = interpreter->typed_tensor<float>(output_idx);
        predictions.push_back(output[0]);
    }

    // STEP 5: Explained Variance Score
    float evs = explained_variance_score(labels, predictions);
    std::cout << "Explained Variance Score: " << evs << "\n";

    return 0;
}
