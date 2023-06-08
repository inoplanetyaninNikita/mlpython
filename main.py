import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

import numpy as np
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import Callback

import threading


class CustomCallback(Callback):
    tb = int

    def __init__(self, totalbatch):
        super().__init__()
        self.tb = totalbatch

    def on_batch_begin(self, batch, logs=None):
        progress_bar["value"] = (batch / self.tb) * 100
        root.update()


df = pd.DataFrame
settings_epoch = 1
text = None

def ml(texts, labels, test_size_percent, epoch):
    global df
    data = df

    # Подготовка размеченных данных
    texts = data[texts.__getitem__(0)]
    labels = data[labels].to_numpy()

    # Препроцессинг текста
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = max([len(seq) for seq in sequences])
    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Создание и обучение LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Преобразование строковых меток в числовой формат
    y = label_encoder.transform(labels)

    # Создание и обучение модели RNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_sequence_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent, shuffle=True)
    total_samples = len(X_train)  # Общее количество образцов
    total_batches = math.ceil(total_samples / 32)

    custom_callback = CustomCallback(total_batches)

    # Создание объекта tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(32)  # Размер пакета

    predict_button.config(state='disabled')

    model.fit(dataset, epochs=epoch, callbacks=custom_callback, use_multiprocessing=True)
    _, accuracy = model.evaluate(X_test, y_test, use_multiprocessing=True)

    predict_button.config(state='normal')
    messagebox.showinfo("Информация", f"Точность составляет: {accuracy}")

    global text
    if text != None: return

    label = tk.Label(root, text="Текст:")
    label.pack(side="left")

    text = tk.Entry(root)
    text.pack(side="right")


    def button_click():
        global text
        # Пример использования обученной модели
        texts_to_check = [
            text.get()
        ]
        sequences_to_check = tokenizer.texts_to_sequences(texts_to_check)
        X_to_check = pad_sequences(sequences_to_check, maxlen=max_sequence_length)
        predictions = model.predict(X_to_check, use_multiprocessing=True)

        # Вывод результатов
        for textforprediect, prediction in zip(texts_to_check, predictions):
            predicted_label = list(label_encoder.classes_)[list(prediction).index(max(prediction))]
            messagebox.showinfo("Информация", f"Предсказание AI: {predicted_label}")
            print(f"Text: {textforprediect}")
            print()

    button = tk.Button(root, text="Предсказать", command=button_click)
    button.pack(side="right")


def browse_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        df = pd.read_csv(file_path)
        feature_listbox.delete(0, tk.END)

        current_values = list(target_combobox['values'])

        for col in df.columns:
            current_values.append(col)

        target_combobox['values'] = current_values

        for column in df.columns:
            feature_listbox.insert(tk.END, column)


def predict_selected_columns():
    features_selected = feature_listbox.curselection()
    target_to_predict = target_combobox.get()

    if features_selected and target_to_predict:
        features_to_train = [feature_listbox.get(idx) for idx in features_selected]
        thread = threading.Thread(target=
        ml(features_to_train, target_to_predict,
           float(train_percent_entry.get()),
           int(epochs_entry.get())))
        thread.start()

root = tk.Tk()
root.title("CSV Predictor")

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack()

# Количество эпох
epochs_label = tk.Label(root, text="Количество эпох:")
epochs_label.pack()

epochs_entry = ttk.Entry(root)
epochs_entry.pack()

# Процент тренировочных данных
train_percent_label = tk.Label(root, text="Процент тестовых данных:")
train_percent_label.pack()

train_percent_entry = ttk.Entry(root)
train_percent_entry.pack()

browse_button = tk.Button(root, text="Выбрать файл", command=browse_file)
browse_button.pack(pady=10)

features_label = tk.Label(root, text="Данные для обучения:")
features_label.pack()

feature_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
feature_listbox.pack()

target_label = tk.Label(root, text="Данные для предсказания:")
target_label.pack()

target_combobox = ttk.Combobox(root, state="readonly")
target_combobox.pack()

predict_button = tk.Button(root, text="Обучить", command=predict_selected_columns)
predict_button.pack(pady=10)

root.mainloop()

# for col in df.columns:
#     current_values.append(col)
#
# target_combobox['values'] = current_values
#
# for column in df.columns:
#     feature_listbox.insert(tk.END, column)
