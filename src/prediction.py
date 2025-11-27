import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def load_latest_model(path="models/cifar10_model_latest.h5"):
if not os.path.exists(path):
raise FileNotFoundError("Latest model not found. Train the model first.")
return load_model(path)


def preprocess_for_predict(img_array):
x = img_array.astype('float32') / 255.0
x = np.expand_dims(x, axis=0)
return x


def decode_predictions_array(probs, top_k=3):
top_idxs = probs.argsort()[::-1][:top_k]
return [(class_names[i], float(probs[i])) for i in top_idxs]


def predict_image(model, img_array):
x = preprocess_for_predict(img_array)
probs = model.predict(x)[0]
top3 = decode_predictions_array(probs, top_k=3)
pred_idx = int(np.argmax(probs))
return {"pred_class": class_names[pred_idx], "confidence": float(probs[pred_idx]), "top3": top3}
