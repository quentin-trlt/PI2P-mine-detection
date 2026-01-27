"""
Script 05: Export du modèle en TFLite pour déploiement mobile
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.augmentation import create_dataset


def load_data_from_directory(split_name):
    """Charge les données depuis un split"""
    split_dir = config.SPLITS_DIR / split_name

    file_paths = []
    labels = []

    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(config.CLASSES))}

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]

            for img_path in list(class_dir.glob('*.jpg'))[:10]:  # Limité pour dataset représentatif
                file_paths.append(str(img_path))
                labels.append(class_idx)

    return np.array(file_paths), np.array(labels)


def representative_dataset_gen():
    """Générateur de dataset représentatif pour quantization"""
    print("Création du dataset représentatif pour quantization...")

    file_paths, labels = load_data_from_directory('train')
    dataset = create_dataset(
        file_paths[:config.REPRESENTATIVE_DATASET_SIZE],
        labels[:config.REPRESENTATIVE_DATASET_SIZE],
        training=False,
        batch_size=1
    )

    for images, _ in dataset.take(config.REPRESENTATIVE_DATASET_SIZE):
        yield [images]


def export_tflite(model_path, quantize=True):
    """Exporte le modèle Keras en TFLite"""
    print("=" * 60)
    print("EXPORT TFLITE")
    print("=" * 60)

    # Charger modèle
    print(f"\nChargement du modèle: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Nom du modèle
    model_name = Path(model_path).parent.name
    output_dir = config.TFLITE_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conversion de base (Float32)
    print("\nConversion TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    float_model_path = output_dir / 'model_float32.tflite'
    with open(float_model_path, 'wb') as f:
        f.write(tflite_model)

    float_size = len(tflite_model) / 1024 / 1024
    print(f"✓ Modèle Float32 sauvegardé: {float_model_path}")
    print(f"  Taille: {float_size:.2f} MB")

    # Quantization si demandé
    if quantize:
        print("\nQuantization (Int8)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_quant_model = converter.convert()

        quant_model_path = output_dir / 'model_int8.tflite'
        with open(quant_model_path, 'wb') as f:
            f.write(tflite_quant_model)

        quant_size = len(tflite_quant_model) / 1024 / 1024
        print(f"✓ Modèle Int8 quantizé sauvegardé: {quant_model_path}")
        print(f"  Taille: {quant_size:.2f} MB")
        print(f"  Réduction: {(1 - quant_size / float_size) * 100:.1f}%")

    # Métadonnées
    create_metadata_file(output_dir)

    # Créer label file
    create_label_file(output_dir)

    # Test d'inférence
    test_inference(quant_model_path if quantize else float_model_path)

    print(f"\n✓ Export terminé dans: {output_dir}")

    return output_dir


def create_metadata_file(output_dir):
    """Crée un fichier de métadonnées JSON"""
    import json

    metadata = {
        'model_info': {
            'input_shape': list(config.INPUT_SHAPE),
            'num_classes': config.NUM_CLASSES,
            'classes': config.CLASSES
        },
        'preprocessing': {
            'normalization': 'divide_by_255',
            'resize': config.IMG_SIZE
        },
        'postprocessing': {
            'output_type': 'probabilities',
            'threshold': 0.5
        }
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Métadonnées sauvegardées: {metadata_path}")


def create_label_file(output_dir):
    """Crée un fichier de labels pour mobile"""
    label_path = output_dir / 'labels.txt'
    with open(label_path, 'w') as f:
        for class_name in config.CLASSES:
            f.write(f"{class_name}\n")

    print(f"✓ Labels sauvegardés: {label_path}")


def test_inference(tflite_model_path):
    """Test d'inférence avec le modèle TFLite"""
    print("\n" + "=" * 60)
    print("TEST D'INFÉRENCE")
    print("=" * 60)

    # Charger modèle TFLite
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    # Détails input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    # Test avec une image aléatoire
    if input_details[0]['dtype'] == np.uint8:
        test_input = np.random.randint(0, 256, size=input_details[0]['shape'], dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)

    # Mesurer latence
    import time
    start = time.time()
    interpreter.invoke()
    latency = (time.time() - start) * 1000

    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"\nLatence d'inférence: {latency:.2f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Somme des probabilités: {output.sum():.4f}")

    print("\n✓ Test d'inférence réussi")


def create_inference_example(output_dir):
    """Crée un exemple de code d'inférence pour mobile"""
    example_code = '''"""

Exemple d'inférence avec TFLite sur mobile
"""
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model(model_path):
    """Charge le modèle TFLite"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_shape):
    """Préprocesse une image pour l'inférence"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.uint8)  # Pour modèle quantizé
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(interpreter, image):
    """Effectue une prédiction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# Utilisation
if __name__ == "__main__":
    # Charger modèle
    interpreter = load_model('model_int8.tflite')

    # Charger labels
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f]

    # Prédire
    input_details = interpreter.get_input_details()
    image = preprocess_image('test_image.jpg', input_details[0]['shape'])
    predictions = predict(interpreter, image)

    # Afficher résultats
    top_idx = np.argmax(predictions)
    print(f"Classe prédite: {labels[top_idx]}")
    print(f"Confiance: {predictions[top_idx]:.2%}")

    # Top 3
    top3_idx = np.argsort(predictions)[-3:][::-1]
    print("\\nTop 3:")
    for idx in top3_idx:
        print(f"  {labels[idx]}: {predictions[idx]:.2%}")
'''

    example_path = output_dir / 'inference_example.py'
    with open(example_path, 'w') as f:
        f.write(example_code)

    print(f"✓ Exemple d'inférence sauvegardé: {example_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export TFLite')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle (.keras)'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Désactiver la quantization'
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Modèle non trouvé: {model_path}")
        sys.exit(1)

    output_dir = export_tflite(
        str(model_path),
        quantize=not args.no_quantize
    )

    create_inference_example(output_dir)

    print("\n=== Export terminé ===")