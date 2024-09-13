from keras.src.models import Sequential
from keras.src.layers import Embedding, Dense, LSTM, Dropout, GlobalMaxPooling1D, BatchNormalization
from keras import regularizers
from keras.src.utils import to_categorical
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from models.datatypes import ModelArgs
from tools.dataset.manager import DatasetManager
from codecarbon import EmissionsTracker
from models.nlp_preprocessing import build_tokenizer, preprocess_text
import matplotlib.pyplot as plt


def build_model(vocab_size, input_length, output_dim, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length))
    model.add(LSTM(
        192, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, kernel_regularizer=regularizers.l2(0.01))
    )
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def plot_training_history(history):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss Function')
    plt.ylabel('Loss Function')
    plt.xlabel('Training Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training History - Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Training Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    tracker = EmissionsTracker('probability')
    tracker.start()

    dataset_manager = DatasetManager()

    model_params = ModelArgs()
    model_params.dim = 192
    model_params.max_seq_len = 1024
    model_params.vocab_size = 32_768
    model_params.max_batch_size = 48

    train_x = dataset_manager.read_dataset(
        'probability_dataset.csv', usecols=['ementa']
    )['ementa'].to_numpy()

    tokenizer = build_tokenizer(train_x, model_params.vocab_size, '<00V>')
    train_x = preprocess_text(train_x, tokenizer, model_params.max_seq_len)

    labels = dataset_manager.read_dataset(
        'probability_dataset.csv', usecols=['sumula']
    )['sumula'].to_numpy()

    num_classes = len(set(labels))
    labels = to_categorical(labels - 1, num_classes=num_classes)

    train_texts, test_texts, train_label, test_label = train_test_split(
        train_x, labels, test_size=0.2, random_state=42
    )

    del train_x
    del labels

    model = build_model(len(tokenizer.word_index) + 1, train_texts.shape[1], model_params.dim, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_texts,
        train_label,
        epochs=8,
        batch_size=model_params.max_batch_size,
        validation_data=(test_texts, test_label),
        callbacks=[early_stopping],
    )

    model.save('./probability.keras')
    model.summary()
    tracker.stop()

    dataset_manager.save_training_history(model, history, 'probability')
    plot_training_history(history)


if __name__ == '__main__':
    main()
