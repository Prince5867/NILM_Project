# NILM: Non-Intrusive Load Monitoring using Embedded AI

This project focuses on appliance-level energy disaggregation using deep learning techniques. It aims to accurately identify and separate power consumption patterns of individual appliances from an aggregated household energy signal. The models are optimized for edge deployment on platforms such as Raspberry Pi.

## 🔍 Project Goals

- Disaggregate aggregate energy consumption into appliance-level usage
- Develop lightweight, efficient deep learning models for embedded deployment
- Support inference pipelines using TensorFlow Lite (TFLite)

---

## 📦 Project Structure


├── Model/ # Models and preprocessing tools
│ ├── CNN/ # CNN-based NILM models and scripts
│ │ ├── CNN model.py
│ │ ├── Conv1D_GLU.py
│ │ ├── ResidualBlock.py
│ │ ├── tf CNN model.ipynb
│ │ ├── *.keras # Saved Keras models
│ │ ├── *.tflite # TFLite quantized models
│ │ ├── *.json # Scaler parameter files
│ │ ├── test_constrained/ # Edge testing setup (Python & C++)
│ │ │ ├── inference.py
│ │ │ ├── main.cpp
│ │ │ ├── *.csv # Sample aggregate + label files
│ │ │ ├── *.tflite
│ │ │ ├── Requirements.txt
│ │ │ └── pycache/
│ │ └── former version/ # Older models/scripts

│ └── Preprocessing/ # Data preprocessing scripts
│ ├── DAD_converter.py # Dataset conversion logic
│ └── windowing.py # Time series windowing

├── Reffit/ # Main dataset (per appliance)
│ ├── Aggregate/
│ ├── Kettle/
│ ├── Fridge/
│ ├── ...
│ ├── ToolKit/ # Helper functions for data exctraction
│ │   └── appliance.py
│ │   └── house_data.txt

│ └── Processed/ # Processed versions for training/inference
│ ├── Aggregate/
│ ├── Dishwasher/
│ ├── Kettle/
│ ├── ...
│ └── Washing Machine/

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
