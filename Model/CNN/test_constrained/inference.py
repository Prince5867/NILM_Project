import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score


class InferenceModel:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.base_dir = self.model_path.parent
        self.interpreter = self.load_tflite_model()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load scalers
        self.scaler_Y = self.load_scaler(self.base_dir / 'scaler_params_Y.json')
        self.scaler_y_test = self.load_scaler(self.base_dir / 'scaler_params_y_test.json')

    def load_tflite_model(self):
        print(f"Loading TFLite model from: {self.model_path}")
        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        interpreter.allocate_tensors()
        return interpreter

    def load_scaler(self, json_path):
        print(f"Loading scaler from: {json_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Scaler params file not found: {json_path}")

        with open(json_path, 'r') as f:
            params = json.load(f)

        scaler = MinMaxScaler()
        scaler.min_ = np.array(params['min_'])
        scaler.scale_ = np.array(params['scale_'])
        scaler.data_min_ = np.array(params['data_min_'])
        scaler.data_max_ = np.array(params['data_max_'])
        scaler.data_range_ = np.array(params['data_range_'])
        return scaler

    def get_data_files(self):
        data_dict = {'X': '', 'y': '', 'X_test': '', 'y_test': ''}
        print(f"Looking for CSV files in: {self.base_dir}")
        for item in self.base_dir.iterdir():
            if item.suffix == '.csv':
                print(f"Found CSV: {item.name}")
                if item.name == 'Synthetic_Aggregate.csv':
                    data_dict['X'] = item
                elif item.name == 'IRL_Aggregate.csv':
                    data_dict['X_test'] = item
                elif item.name == 'Synthetic_Y.csv':
                    data_dict['y'] = item
                elif item.name == 'IRL_Y.csv':
                    data_dict['y_test'] = item
        return data_dict

    def load_data(self, data_dict, no_samples=1000):
        def load_csv(path):
            return np.array(pd.read_csv(path))

        try:
            X = load_csv(data_dict['X'])[:no_samples]
            y = load_csv(data_dict['y'])[:no_samples]
            X_test = load_csv(data_dict['X_test'])[:no_samples]
            y_test = load_csv(data_dict['y_test'])[:no_samples]
        except Exception as e:
            print("Error loading data:", e)
            raise e

        return X, y, X_test, y_test

    def preprocess_data(self):
        data_dict = self.get_data_files()
        X, y, X_test, y_test = self.load_data(data_dict)

        # Reshape inputs
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if X_test.ndim == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Reshape outputs
        if y.ndim == 2:
            y = y.reshape(y.shape[0], 75, 3)
        if y_test.ndim == 2:
            y_test = y_test.reshape(y_test.shape[0], 75, 3)

        return X, y, X_test, y_test

    def run_model(self, X):
        print("Running TFLite inference...")
        input_index = self.input_details[0]['index']
        output_index = self.output_details[0]['index']
        output_shape = self.output_details[0]['shape']

        predictions = []

        for i in tqdm(range(X.shape[0]), desc="Inference Progress"):
            input_tensor = X[i:i+1].astype(np.float32)  # shape: (1, timesteps, 1)
            self.interpreter.set_tensor(input_index, input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_index)
            predictions.append(output[0])  # remove batch dimension

        predictions = np.array(predictions)
        return predictions

    def inverse_scale(self, scaled_array, scaler):
        flat_scaled = scaled_array.reshape(-1, scaled_array.shape[-1])
        flat_unscaled = scaler.inverse_transform(flat_scaled)
        unscaled = flat_unscaled.reshape(scaled_array.shape)
        return unscaled

    def evaluate_model(self, predictions, y_true):
        predictions_watts = self.inverse_scale(predictions, self.scaler_y_test)
        y_true_watts = self.inverse_scale(y_true, self.scaler_y_test)

        y_true_flat = y_true_watts.reshape(-1, y_true_watts.shape[-1])
        y_pred_flat = predictions_watts.reshape(-1, predictions_watts.shape[-1])
        
        residuals = y_true_flat - y_pred_flat

        mae = np.mean(np.abs(residuals))
        residual_variance = np.var(residuals)
        explained_variance = explained_variance_score(y_true_flat, y_pred_flat)

        print(f"📊 Overall Metrics:")
        print(f"Mean Absolute Error (Watts): {mae:.4f}")
        print(f"Residual Variance: {residual_variance:.4f}")
        print(f"Explained Variance Score: {explained_variance:.4f}")

        # Per-class metrics
        print(f"\n📈 Per-Class Metrics:")
        n_classes = y_true_flat.shape[1]
        class_names = ['Washer Dryer', 'Dishwasher', 'Kettle']
        per_class_results = {}

        for i in range(n_classes):
            mae_i = np.mean(np.abs(y_true_flat[:, i] - y_pred_flat[:, i]))
            var_i = np.var(y_true_flat[:, i] - y_pred_flat[:, i])
            evs_i = explained_variance_score(y_true_flat[:, i], y_pred_flat[:, i])
            print(f"{class_names[i]}: MAE = {mae_i:.4f}, Residual Variance = {var_i:.4f}, EVS = {evs_i:.4f}")
            per_class_results[f"class_{i}"] = {
                "mae": mae_i,
                "residual_variance": var_i,
                "explained_variance": evs_i
            }

        return {
            "mae": mae,
            "residual_variance": residual_variance,
            "explained_variance": explained_variance,
            "per_class": per_class_results
        }
    def plot_prediction(self, y_true, y_pred, sample_index=0):
        """
        Plots predicted vs. true values for a given sample.
        """
        y_true_unscaled = self.inverse_scale(y_true, self.scaler_y_test)
        y_pred_unscaled = self.inverse_scale(y_pred, self.scaler_y_test)
        list_of_appliances = ['Washer Dryer', 'Dishwasher', 'Kettle']

        plt.figure(figsize=(12, 6))
        for appliance in range(len(list_of_appliances)):
            plt.plot(
                y_true_unscaled[sample_index, :, appliance],
                label=f'True - {list_of_appliances[appliance]}',
                linestyle='--'
            )
            plt.plot(
                y_pred_unscaled[sample_index, :, appliance],
                label=f'Pred - {list_of_appliances[appliance]}'
            )

        plt.title(f'Prediction vs Ground Truth for Sample {sample_index}')
        plt.xlabel('Time step')
        plt.ylabel('Power (Watts)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.base_dir}/prediction_sample_{sample_index}.png")
        print(f"Prediction plot saved as: {self.base_dir}/prediction_sample_{sample_index}.png")

def main():
    model_path = Path(__file__).resolve().parent / 'quant8model.tflite'
    inference = InferenceModel(model_path)
    X, y, X_test, y_test = inference.preprocess_data()
    predictions = inference.run_model(X)
    inference.evaluate_model(predictions, y)
    inference.plot_prediction(y, predictions, sample_index=0)


if __name__ == "__main__":
    main()
