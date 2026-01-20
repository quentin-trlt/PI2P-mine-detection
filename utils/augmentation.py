"""
Utilitaires pour l'augmentation de données
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import config


def preprocess_image(image_path, label):
    """Préprocesse une image pour le training"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=config.IMG_CHANNELS)
    image = tf.image.resize(image, [config.IMG_SIZE, config.IMG_SIZE])
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, config.NUM_CLASSES)
    return image, label


def augment_image(image, label):
    """Applique l'augmentation"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2 * 255.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def preprocess_for_inference(image_path):
    """Préprocesse une image pour l'inférence (sans label)"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=config.IMG_CHANNELS)
    image = tf.image.resize(image, [config.IMG_SIZE, config.IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image


def create_dataset(file_paths, labels, training=False, batch_size=None):
    """Crée un tf.data.Dataset optimisé"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=config.RANDOM_SEED)

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_augmentation_layer():
    """Crée une couche d'augmentation pour le training"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ], name='augmentation')