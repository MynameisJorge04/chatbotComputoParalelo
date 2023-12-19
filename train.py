import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from keras import layers
from keras import losses
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time


class TimeCallback(Callback):

    def __init__(self):
        super().__init__()
        self.times = []
        self.start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end = time.time()
        self.times.append(end - self.start + (self.times[-1] if len(self.times) > 0 else 0))


def create_vectorize_layer(text_array):
    max_features = 10000
    sequence_length = 100
    vectoirzer_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vectoirzer_layer.adapt(text_array)
    return vectoirzer_layer
    

def create_model():
    max_features = 10000
    embedding_dim = 128
    model = tf.keras.Sequential([
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(12, activation='softmax')
    ])
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=['accuracy']
    )
    model.build((None,))
    model.summary()
    return model
    
    
def vectorize_text(text, vectoirzer_layer):
    text = tf.expand_dims(text, -1)
    return vectoirzer_layer(text)
   
    
def main(device):
    time_clallback = TimeCallback()
    columns = ['text', 'label']
    data = pd.read_csv('data.csv', encoding='utf-8', header=None, names=columns)
    X = data['text'].values
    y = data['label'].values
    clases = np.unique(y)
    df_clases = pd.DataFrame(clases, columns=['clases'])
    df_clases.to_csv('data/clases.csv', index=False)
    y_onehot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)
    vectorizer_layer = create_vectorize_layer(x_train)
    x_train_vect = vectorize_text(x_train, vectorizer_layer)
    x_test_vect = vectorize_text(x_test, vectorizer_layer)
    print(x_train_vect.shape)
    print(y_train.shape)
    model = create_model()
    vals = model.fit(x_train_vect, y_train, validation_data=(x_test_vect, y_test), epochs=10, batch_size=32, callbacks=[time_clallback])
    history = vals.history
    history['time'] = time_clallback.times
    history = pd.DataFrame(history)
    history.to_csv(f'data/history_{device}.csv', index=False)
    model.save(f'data/model_{device}.keras')
    
    
if __name__ == '__main__':
    tf.device('/cpu:0')
    main('cpu')
    
