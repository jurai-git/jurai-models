from collections import Counter
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from models.datatypes import ModelArgs
from tools.dataset.manager import DatasetManager
from models.nlp_preprocessing import evaluate_model, build_tokenizer, preprocess_text


def load_and_prepare_data(dataset_manager, model_params):
    texts = dataset_manager.read_dataset(
        'probability_dataset.csv', usecols=['ementa']
    )['ementa']

    labels = dataset_manager.read_dataset(
        'probability_dataset.csv', usecols=['sumula']
    )['sumula'].to_numpy()

    tokenizer = build_tokenizer(texts, model_params.vocab_size, '<00V>')
    texts = preprocess_text(texts, tokenizer, model_params.max_seq_len)

    num_classes = len(set(labels))
    labels = to_categorical(labels - 1, num_classes=num_classes)

    return texts, labels


def split_data(texts, labels):
    return train_test_split(texts, labels, test_size=0.2, random_state=42)


def display_predictions(predictions, predicted_classes, true_classes):
    for i in range(12):
        print(f'Prediction (probabilities):\n{predictions[i]}')
        print(f'Predicted class:\n{predicted_classes[i]}')
        print(f'True class:\n{true_classes[i]}')


def display_class_distribution(predicted_classes):
    class_counts = Counter(predicted_classes)
    print('\n\nClass occurrence counts in predicted_classes:')
    for class_label, count in class_counts.items():
        print(f'Class {class_label}: {count} occurrences')


def main():
    model = load_model('./probability.keras')

    dataset_manager = DatasetManager()
    model_params = ModelArgs(max_seq_len=1024, vocab_size=32_768)

    texts, labels = load_and_prepare_data(dataset_manager, model_params)

    _, test_texts, _, test_labels = split_data(texts, labels)

    predictions, predicted_classes, true_classes = evaluate_model(model, test_texts, test_labels)

    display_predictions(predictions, predicted_classes, true_classes)
    display_class_distribution(predicted_classes)

    model.summary()


if __name__ == '__main__':
    main()
