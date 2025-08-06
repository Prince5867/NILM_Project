# NILM: Non-Intrusive Load Monitoring using Embedded AI

This project focuses on appliance-level energy disaggregation using deep learning techniques. It aims to accurately identify and separate power consumption patterns of individual appliances from an aggregated household energy signal. The models are optimized for edge deployment on platforms such as Raspberry Pi.

## 🔍 Project Goals

- Disaggregate aggregate energy consumption into appliance-level usage
- Develop lightweight, efficient deep learning models for embedded deployment
- Support inference pipelines using TensorFlow Lite (TFLite)

---

## 📦 Project Structure

For memory purposes all the datasets and .csv files are excluded from this respository and need to be downloaded from https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/datasets/

- inference/ Edge testing setup (Python & C++)
    -  inference.py — Python inference script using TFLite
    - main.cpp — C++ inference implementation
    - datasets/ — Synthetic and real-life datasets (X and y)
    - tflite_models/ — TFLite models for inference
    - Requirements.txt — Python dependencies for inference

- Model/ Models and preprocessing tools
    - ML_pipeline/ — CNN-based NILM models and notebooks
        - NILM_pipeline.ipynb — Notebook for model training and evaluation
        - keras_model/ — Saved Keras models (.keras files)
        - tflite_models/ — TFLite quantized models (.tflite files)
        - *.json — Scaler parameter files for input/output normalization
    - Preprocessing/ — Data preprocessing scripts
        - DAD_converter.py — Dataset format conversion
        - windowing.py — Time window segmentation for time series

- Reffit/ Main dataset organized by appliance
    - Aggregate/ — Whole-house aggregate data
    - Kettle/
    - Fridge/
    - ... (other appliance folders)
    - ToolKit/ — Helper functions for data extraction
        - appliance.py
        - house_data.txt
    - Processed/ — Processed datasets ready for training/inference
        - Aggregate/
        - Dishwasher/
        - Kettle/
        - ...
        - Washing Machine/

---
## 🧠 Models Implemented

### 📦 13-Class Classification
- CNN-LSTM-based model to classify appliance from single-window sequence
- Appliance classes: *Fridge*,*Freezer*,*Washing Machine*,*Washer Dryer*,*Tumble Dryer*,*Dishwasher*,*Microwave*,*Toaster*,*Kettle*,
                *Computer*,*Electric Heater*,*Hi-Fi*,*Overhead Fan*

### ✅ Classification (ON/OFF Detection)
- **Seq2Seq CNN-LSTM model** trained to detect appliance activity
- Targeted appliances: Kettle, Dishwasher, Washer Dryer

### 📈 Regression (Power Consumption Estimation)
- **Seq2Seq CNN-LSTM models**
- Predict power usage of individual appliances from aggregate signal
- Target appliances: Kettle, Dishwasher, Washer Dryer
---
## 💡 Key Features

- Uses `MinMaxScaler` with JSON-based restoration for input/output normalization
- Inference scripts in both Python (and C++)
- Supports CLI arguments for batch inference and model selection
- Explains prediction performance with metrics like **MAE**, **Explained Variance**
- Visualization of predicted vs ground truth appliance power signals

## 🖥️ Deployment Targets

- ✅ Local evaluation using Python
- 🔄 Conversion to C++ for deployment on Raspberry Pi (TODO)
- 🔋 Future support for ESP32 using micro TFLite (TensorFlow Lite for Microcontrollers)

---

## 🚧 TODO (In Progress)

- [ ] Add C++ inference CLI with JSON scaler loading
- [ ] Optimize model architectures for latency/memory

---

## 📋 How to Run



---

## 📬 Contact

**Niranjan Kulkarni**: niranjankulkarni.fr@gmail.com
Feel free to reach out for collaboration, suggestions, or questions.
