import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# ------------------------------
# 1. Encoder (기존 CNN)
# ------------------------------
def build_encoder(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
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
    ])
    return model

# ------------------------------
# 2. Projection Head (SSL 전용)
# ------------------------------
def build_projection_head():
    return models.Sequential([
        layers.Dense(64),
        layers.ReLU(),
        layers.Dense(32)
    ])

# ------------------------------
# 3. Classifier Head
# ------------------------------
def build_classifier_head(num_classes=6):
    return models.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

# ------------------------------
# 4. Contrastive Loss (SimCLR)
# ------------------------------
def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    logits = tf.matmul(z_i, tf.transpose(z_j)) / temperature
    labels = tf.range(tf.shape(logits)[0])
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# ------------------------------
# 5. Augmentation (임시 예시)
# ------------------------------
def augment(x):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
    return x + noise

# ------------------------------
# 6. 모델 및 옵티마이저 정의
# ------------------------------
input_shape = (4, 14, 1)
encoder = build_encoder(input_shape)
projection_head = build_projection_head()
classifier_head = build_classifier_head(num_classes=6)

ssl_optimizer = optimizers.Adam(1e-3)
cls_optimizer = optimizers.Adam(1e-4)

# ------------------------------
# 7. 초기 Self-Supervised Pretraining
# ------------------------------
def pretrain_ssl(x_batch):
    with tf.GradientTape() as tape:
        x1 = augment(x_batch)
        x2 = augment(x_batch)

        z1 = projection_head(encoder(x1, training=True), training=True)
        z2 = projection_head(encoder(x2, training=True), training=True)

        loss = tf.reduce_mean(contrastive_loss(z1, z2))

    grads = tape.gradient(loss, encoder.trainable_variables + projection_head.trainable_variables)
    ssl_optimizer.apply_gradients(zip(grads, encoder.trainable_variables + projection_head.trainable_variables))
    return loss

# ------------------------------
# 8. Online Learning (with labels)
# ------------------------------
def online_update(x, y):
    with tf.GradientTape() as tape:
        features = encoder(x, training=True)
        preds = classifier_head(features, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, encoder.trainable_variables + classifier_head.trainable_variables)
    cls_optimizer.apply_gradients(zip(grads, encoder.trainable_variables + classifier_head.trainable_variables))
    return loss, preds
