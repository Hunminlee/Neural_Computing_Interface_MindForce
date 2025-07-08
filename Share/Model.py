import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')



def Original_model_V1(input_size):

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

    # Compile the model
    optimizer = optimizers.SGD(
        learning_rate=0.01,
        momentum=0.85
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def Train_model(model, X_train, y_train, X_test, y_test, set_epoch, set_batch_size, Model_name, set_verbose, save_model_set):
    # Learning rate scheduler
    lr_schedule = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * 0.5 if epoch % 3 == 0 and epoch != 0 else lr
    )

    # Optional: Early stopping or model checkpointing
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Save the best model - val accuracy
    checkpoint = ModelCheckpoint(
        f'{Model_name}.keras',  # File path to save
        monitor='val_accuracy',  # Metric to monitor
        save_best_only=True,  # Only save when val_loss improves
        mode='max',  # Lower val_loss is better
        verbose=0
    )

    print(f"Start Training (total epochs: {set_epoch})...")

    if save_model_set:  #Save model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=set_epoch,
            batch_size=set_batch_size,
            callbacks=[lr_schedule, early_stop, checkpoint],
            shuffle=True,
            verbose=set_verbose
        )
        print("Finish Training! (Model is saved)")

    else:   ####  Don't save
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=set_epoch,
            batch_size=set_batch_size,
            callbacks=[lr_schedule, early_stop],
            shuffle=True,
            verbose=set_verbose
        )
        print("Finish Training! (Model is NOT saved)\n")

    print(f"Maximum training accuracy : {np.round(float(np.max(history.history['accuracy']) * 100), 2)}%")
    print(f"Maximum validation accuracy : {np.round(float(np.max(history.history['val_accuracy']) * 100), 2)}%")

    return history, model
