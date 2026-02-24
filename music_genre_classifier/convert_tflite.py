import tensorflow as tf
import os
import numpy as np

MODEL_PATH = 'best_model.h5'
TFLITE_PATH = 'songnet.tflite'

def convert_to_tflite():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Load the Keras model
        # Note: If custom layers (SelfAttention) are used, we need to pass custom_objects
        # But we need to import SelfAttention class here.
        from model import SelfAttention
        
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'SelfAttention': SelfAttention})
        
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable Select TF Ops (Required for RNNs usually)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
        ]
        
        # Disable experimental lowering of tensor list ops (suggested by error)
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        
        # Save
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Successfully saved TFLite model to {TFLITE_PATH}")
        print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_to_tflite()
