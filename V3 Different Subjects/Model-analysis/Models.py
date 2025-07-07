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


def SE_CNN(input_size):
    inputs = layers.Input(shape=input_size)

    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Squeeze-and-Excitation block
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(16, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 16))(se)
    x = layers.Multiply()([x, se])

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def Attention_CNN(input_size):
    inputs = layers.Input(shape=input_size)

    # Base Conv
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Attention mask
    attn = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attn])  # Apply attention to feature map

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def Gated_CNN(input_size):
    inputs = layers.Input(shape=input_size)

    # Convolution
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x_act = layers.ReLU()(x)

    # Gate
    gate = layers.Conv2D(16, (3, 3), padding='same', activation='sigmoid')(inputs)

    # Gated output
    x = layers.Multiply()([x_act, gate])

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


from tensorflow.keras import layers, models, optimizers

def Hybrid_CNN(input_size):
    inputs = layers.Input(shape=input_size)

    ### 1. Multi-scale inception-style parallel conv
    conv3 = layers.Conv2D(8, (3, 1), padding='same', activation='relu')(inputs)
    conv5 = layers.Conv2D(8, (5, 1), padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(inputs)
    x = layers.Concatenate()([conv1, conv3, conv5])
    x = layers.BatchNormalization()(x)

    ### 2. Gated activation
    gate = layers.Conv2D(24, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, gate])

    ### 3. Residual block + SE
    res_input = x
    x = layers.Conv2D(32, (3, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Squeeze-and-Excitation
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(32, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 32))(se)
    x = layers.Multiply()([x, se])

    # Residual connection (res_input has 24 channels; match with 32 using 1x1 conv)
    res_proj = layers.Conv2D(32, (1, 1), padding='same')(res_input)
    x = layers.Add()([x, res_proj])
    x = layers.ReLU()(x)

    ### 4. Spatial attention block
    attn = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attn])

    ### 5. Classification head
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def BiLSTM_Attention_14features(input_size=(4, 14)):
    inputs = layers.Input(shape=input_size)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(64)(attention)
    attention = layers.Permute([2, 1])(attention)
    x = layers.Multiply()([x, attention])

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def Conv2D_then_LSTM(input_size=(4, 14, 1)):
    inputs = layers.Input(shape=input_size)

    # Conv2D layers to extract spatial features across 14 features (width)
    x = layers.TimeDistributed(
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')
    )(inputs)  # TimeDistributed applies conv1d over each time step

    x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # Now x shape is (batch, time_steps, features_flat)
    x = layers.LSTM(64)(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



from tensorflow.keras import layers, models, optimizers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def Transformer_Model(input_size=(4, 14), head_size=64, num_heads=4, ff_dim=128, num_layers=2, num_classes=6):
    inputs = layers.Input(shape=input_size)
    x = inputs

    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def Hybrid_CNN_Transformer(input_size=(4, 14, 1), num_classes=6):
    inputs = layers.Input(shape=input_size)

    # Conv2D block to extract spatial features per time step
    x = layers.TimeDistributed(layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)  # Now shape: (batch, time_steps, features)

    # Transformer encoder layers
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def V1_2DCNN_with_many_params(input_size):

    filter_size = (3, 3)  # or your desired kernel size

    model = models.Sequential([
        layers.InputLayer(shape=input_size, name="imageinput"),

        layers.Conv2D(128, filter_size, padding='same', name="conv_1"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_1"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_1"),

        layers.Conv2D(128, filter_size, padding='same', name="conv_2"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_2"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_2"),

        layers.Conv2D(128, filter_size, padding='same', name="conv_3"),
        layers.BatchNormalization(),
        layers.ReLU(name="relu_3"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name="maxpool_3"),

        # Optional conv_4 block is commented out as in your MATLAB code
        #layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(512, name="fc_2_2"),
        layers.ReLU(name="relu_5_2"),
        layers.Dropout(0.5),
        layers.Dense(256, name="fc_2_3"),
        layers.ReLU(name="relu_5_3"),
        layers.Dropout(0.5),

        layers.Dense(6, activation='softmax', name="fc_3")  # 6-class classification
    ])

    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
