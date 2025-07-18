import tensorflow as tf
from keras.models import load_model
import custom_fn_library
from pathlib import Path

def convert_model_to_tflite(model_path, output_path):
    print(f"🔍 Loading model from: {model_path}")
    model = load_model(model_path, custom_objects={
        'relu6_div6': custom_fn_library.relu6_div6,
        'Conv1DGLUBlock': custom_fn_library.Conv1DGLUBlock,
        'ResidualBlock': custom_fn_library.ResidualBlock,
    })

    print("🔄 Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    # Optional optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print("❌ Conversion failed:", e)
        return

    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to: {output_path}")

if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "finetuned_regression_model.keras"
    output_path = Path(__file__).resolve().parent / "model.tflite"
    convert_model_to_tflite(model_path, output_path)