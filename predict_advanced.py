from typing import List, Dict
import json
import os
import argparse

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import models

from advanced_data import read_instances, load_vocabulary, index_instances, generate_batches
from util import ID_TO_CLASS


def load_pretrained_model(serialization_dir: str) -> models.Model:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.ckpt.index")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    model_path = model_path.replace(".index", "")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    model_name = config.pop("type")
    if model_name == "basic":
        from model import MyBasicAttentiveBiGRU # To prevent circular imports
        model = MyBasicAttentiveBiGRU(**config)
    elif model_name == "advanced":
        from model import MyAdvancedModel # To prevent circular imports
        model = MyAdvancedModel(config)
    else:
        raise Exception(f"model_name: {model_name} is not supported.")
    model.load_weights(model_path)

    return model


def predict(model: models.Model, instances: List[Dict], batch_size: int, save_to_file: str = None):
    batches = generate_batches(instances, batch_size)
    predicted_labels = []

    all_predicted_labels = []
    print(f"\nStarting predictions")
    for batch_inputs in tqdm(batches):
        batch_inputs.pop("labels")
        logits = model(**batch_inputs, training=False)['logits']
        predicted_labels = list(tf.argmax(logits, axis=-1).numpy())
        all_predicted_labels += predicted_labels

    if save_to_file:
        with open(save_to_file, 'w') as file:
            for predicted_label, instance in zip(all_predicted_labels, instances):
                file.write(f"{instance['sentence_id']}\t{ID_TO_CLASS[predicted_label]}\n")
    else:
        for predicted_label, instance in zip(all_predicted_labels, instances):
            print(f"{instance['sentence_id']}\t{ID_TO_CLASS[predicted_label]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set test config")

    parser.add_argument('--load-serialization-dir', type=str, help="Location of saved model")
    parser.add_argument('--data-file-path', type=str, help='Location of test file', default='data/test.txt')
    parser.add_argument('--prediction-file', type=str, help="Location of output file")
    parser.add_argument('--batch-size', type=int, help="size of batch", default=32)

    args = parser.parse_args()

    MAX_NUM_TOKENS = 250
    test_instances = read_instances(args.data_file_path, MAX_NUM_TOKENS, test=True)

    vocabulary_path = os.path.join(args.load_serialization_dir, "vocab.txt")
    vocab_token_to_id, _ = load_vocabulary(vocabulary_path)

    test_instances = index_instances(test_instances, vocab_token_to_id)

    # load config
    config_path = os.path.join(args.load_serialization_dir, "config.json")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = json.load(f)

    # load model
    model = load_pretrained_model(args.load_serialization_dir)

    predict(model, test_instances, args.batch_size, args.prediction_file)

    if args.prediction_file:
        print(f"predictions stored at: {args.prediction_file}")
