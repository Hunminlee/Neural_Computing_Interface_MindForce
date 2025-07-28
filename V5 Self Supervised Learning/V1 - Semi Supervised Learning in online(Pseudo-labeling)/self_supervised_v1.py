import numpy as np
import tensorflow as tf
from decorator import append
from tensorflow.keras import layers, models, optimizers

# ------------------------------
# 1. Define CNN model
# ------------------------------
def build_model(input_shape=(4, 14, 1), num_classes=6):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),

        layers.Conv2D(8, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(64),
        layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.85),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ------------------------------
# 2. Initial Supervised Training
# ------------------------------
def train_initial_model(model, X_init, y_init, epochs=10, batch_size=32):
    model.fit(X_init, y_init, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# ------------------------------
# 3. Online Update with Pseudo-labels
# ------------------------------
def online_update(model, new_sample, pseudo_label, learning_rate=0.001):
    model.train_on_batch(new_sample, pseudo_label)
    return model

# ------------------------------
# 4. Pseudo-label generator (naive version)
# ------------------------------
def generate_pseudo_label(model, x):
    y_pred = model.predict(x, verbose=0)
    return np.argmax(y_pred, axis=1)




def detection_model():
    import lightgbm as lgb

    def detection_model():
        """
        Create and return a LightGBM binary classification model.
        """
        model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        return model


