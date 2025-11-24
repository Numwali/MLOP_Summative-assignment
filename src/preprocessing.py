import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import io
import logging

logger = logging.getLogger(__name__)

def normalize_image_array(img_array):
    """Normalize image array to [0,1]."""
    return img_array.astype("float32") / 255.0


def preprocess_single_image(img_bytes):
    """Preprocess a single image from bytes into a 4D array."""
    img = Image.open(io.BytesIO(img_bytes)).resize((32, 32)).convert("RGB")
    x = np.array(img)
    x = normalize_image_array(x)
    x = np.expand_dims(x, axis=0)
    return x


def load_images_from_directory(directory):
    """Load all images from a folder and return arrays and labels."""
    images = []
    labels = []

    if not os.path.exists(directory):
        return np.array(images), np.array(labels)

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).resize((32, 32)).convert("RGB")
                images.append(np.array(img))
                labels.append(class_name)
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")

    return np.array(images), np.array(labels)


def preprocess_data(X, y):
    """Preprocess datasets for training: normalize X and one-hot encode y."""
    X = normalize_image_array(X)

    if y.ndim == 1 or y.shape[1] == 1:
        y = to_categorical(y, num_classes=10)

    return X, y


def _self_test():
    # Test normalization
    rand_img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    norm = normalize_image_array(rand_img)
    assert norm.max() <= 1.0 and norm.min() >= 0.0, "Normalization failed"

    # Test single image preprocess
    with io.BytesIO() as buf:
        Image.fromarray(rand_img).save(buf, format="PNG")
        b = buf.getvalue()

    pre = preprocess_single_image(b)
    assert pre.shape == (1, 32, 32, 3), f"Incorrect shape: {pre.shape}"

    # Test empty directory
    Xr, yr = load_images_from_directory("non_existent_dir_for_test")
    assert Xr.shape[0] == 0 and yr.size == 0, "Empty directory test failed"

    logger.info("preprocessing self-test passed.")


if __name__ == "__main__":
    _self_test()

