import numpy as np
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


def evaluate_model(model, texts, labels):
    predictions = model.predict(texts)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    print(f'Accuracy on test set: {np.mean(predicted_classes == true_classes):.4f}')

    return predictions, predicted_classes, true_classes


def build_tokenizer(train_x, vocab_size, oov_token) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(train_x)
    return tokenizer


def preprocess_text(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')
