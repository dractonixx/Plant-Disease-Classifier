import os
import numpy as np
import tensorflow as tf
import json

_model = None

def predict(image_path, top_k=1):  # Changed default to 1
    global _model
    
    if _model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
        model_path = os.path.join(base_dir, "best_model.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading model from: {model_path}")
        _model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
    
    # Load class names
    base_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    class_names_path = os.path.join(base_dir, "class_names.json")
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    
    # EfficientNet preprocessing
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    # Make prediction
    probs = _model.predict(arr, verbose=0)[0]
    top_idx = probs.argsort()[-top_k:][::-1]

    print(f"\nPrediction for {image_path}:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank}. {class_names[i]:<40} {probs[i] * 100:>6.2f}%")
    
    return [(class_names[i], float(probs[i])) for i in top_idx]