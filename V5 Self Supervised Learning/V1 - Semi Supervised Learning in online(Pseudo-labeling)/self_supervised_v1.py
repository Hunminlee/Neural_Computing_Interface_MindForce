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


def build_model_after_detection(input_shape=(4, 14, 1), num_classes=5):
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




def detection_model_LGBM():
    import lightgbm as lgb

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

def detection_model_XGB():
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,  # 최신 버전에서는 필요
        eval_metric='logloss',   # 경고 방지용
        random_state=42,
        n_jobs=-1
    )
    return model


def grid_search_model_LGBM(X_train, y_train, X_test, y_test):
    from sklearn.model_selection import GridSearchCV
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score

    best_model = None
    best_acc = 0
    best_params = None

    # 모델 정의
    model = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

    # 그리드 정의
    param_grid = {
        'num_leaves': [31, 63, 96],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10]
    }

    # GridSearchCV 수행
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # 평가
    candidate_model = grid.best_estimator_
    y_pred = candidate_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Best Parameters from CV:", grid.best_params_)
    print("Best CV Accuracy:", grid.best_score_)
    print("Test Accuracy of Best Model:", acc)

    # 최고 성능 갱신 여부 확인 및 저장
    if acc > best_acc:
        best_acc = acc
        best_model = candidate_model
        best_params = grid.best_params_

    print(f"Selected Best Test Accuracy: {best_acc:.4f}")
    return best_model, best_params, best_acc

