import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')




def Original_2DCNN(input_size):

    filter_size = (3, 3)  # or your desired kernel size

    model = models.Sequential([
        layers.InputLayer(shape=input_size, name="imageinput"),
        layers.Conv2D(8, filter_size, padding='same', name="conv_1"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_1"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_1"),

        layers.Conv2D(16, filter_size, padding='same', name="conv_2"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_2"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_2"),

        layers.Conv2D(32, filter_size, padding='same', name="conv_3"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_3"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_3"),

        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(64, name="fc_2_2"),
        layers.ReLU(name="relu_5_2"),
        layers.Dropout(0.5),

        layers.Dense(6, activation='softmax', name="fc_3")  # 6-class classification
    ])

    # Compile the model
    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.85)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

### SGD -> Adam
def V1_2DCNN(input_size):

    filter_size = (3, 3)  # or your desired kernel size

    model = models.Sequential([
        layers.InputLayer(shape=input_size, name="imageinput"),

        layers.Conv2D(8, filter_size, padding='same', name="conv_1"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_1"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_1"),

        layers.Conv2D(16, filter_size, padding='same', name="conv_2"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_2"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_2"),

        layers.Conv2D(32, filter_size, padding='same', name="conv_3"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_3"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_3"),

        # Optional conv_4 block is commented out as in your MATLAB code
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(64, name="fc_2_2"),
        layers.ReLU(name="relu_5_2"),
        layers.Dropout(0.5),

        layers.Dense(6, activation='softmax', name="fc_3")  # 6-class classification
    ])

    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def ResNetStyle_CNN(input_size):
    inputs = layers.Input(shape=input_size, name="input")

    # Block 1
    x = layers.Conv2D(8, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x  # Save for skip connection
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Block 2
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, layers.Conv2D(16, (1, 1), strides=2, padding='same')(skip1)])  # Skip connection
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Block 3
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.85),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def InceptionStyle_CNN(input_size):
    inputs = layers.Input(shape=input_size)

    # Parallel convs
    conv3 = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    conv5 = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(inputs)

    x = layers.Concatenate()([conv1, conv3, conv5])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Conv + Pool
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def SeparableCNN(input_size):
    inputs = layers.Input(shape=input_size)

    x = layers.SeparableConv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def Train_model(model, X_train, y_train, X_test, y_test, set_epoch, set_batch_size, set_verbose):

    lr_schedule = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * 0.5 if epoch % 3 == 0 and epoch != 0 else lr
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=set_epoch,
        batch_size=set_batch_size,
        callbacks=[lr_schedule],
        shuffle=True,
        verbose=set_verbose
    )

    print(f"Maximum training accuracy : {np.round(float(np.max(history.history['accuracy']) * 100), 2)}%")
    print(f"Maximum validation accuracy : {np.round(float(np.max(history.history['val_accuracy']) * 100), 2)}%")

    return history, model
