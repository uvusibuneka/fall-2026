import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)


# реализовать вычисление отступа объекта (визуализировать, проанализировать);

import numpy as np
import matplotlib.pyplot as plt

from autograd import Tensor

X = df.drop(columns=["Outcome"]).to_numpy(dtype=float)
y = df["Outcome"].to_numpy(dtype=float)

y = np.where(y == 1, 1.0, -1.0)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1.0

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_test = np.column_stack((np.ones(len(X_test)), X_test))


def g(X: Tensor, w: Tensor) -> Tensor:
    return X @ w


def a(X: Tensor, w: Tensor) -> np.ndarray:
    return np.where(g(X, w).data >= 0, 1.0, -1.0)


def M(X: Tensor, y: Tensor, w: Tensor) -> Tensor:
    return g(X, w) * y


def L(X: np.ndarray, y: np.ndarray, w: Tensor) -> np.ndarray:
    return a(Tensor(X), w) != y


def Q(X: np.ndarray, y: np.ndarray, w: Tensor) -> float:
    return L(X, y, w).mean()


def quadratic_loss(X: Tensor, y: Tensor, w: Tensor) -> Tensor:
    margin = M(X, y, w)
    residual = 1.0 + (-margin)
    return (residual * residual).mean()


def plot_margins(X: np.ndarray, y: np.ndarray, w: Tensor):
    margins = y * (X @ w.data)
    margins = np.sort(margins)

    plt.figure(figsize=(10, 5))
    plt.plot(margins, label="Отступы")
    plt.axhline(0, linestyle="--", label="Граница ошибки")
    plt.xlabel("Объекты")
    plt.ylabel("Отступ M")
    plt.title("Отступы объектов")
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_DIR / "margins.png")
    plt.close()


def analyze_margins(X: np.ndarray, y: np.ndarray, w: Tensor):
    margins = y * (X @ w.data)

    print("Ошибочные объекты (M < 0):", np.sum(margins < 0))
    print("Правильные объекты (M > 0):", np.sum(margins > 0))
    print("Минимальный отступ:", margins.min())
    print("Средний отступ:", margins.mean())
    print("Максимальный отступ:", margins.max())


# реализовать вычисление градиента функции потерь;

def grad(result: Tensor, w: Tensor) -> np.ndarray:
    result.backward()
    return w.grad.copy()


# реализовать рекуррентную оценку функционала качества;

def recursive_Q(previous_Q, current_loss, lambda_=0.01):
    if previous_Q is None:
        return current_loss

    return lambda_ * current_loss + (1.0 - lambda_) * previous_Q


# реализовать метод стохастического градиентного спуска с инерцией;

from optimizers import SGDMomentum


def train_sgd_momentum(X, y, w, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.0, lambda_=0.01, order_mode="random", random_state=42):
    rng = np.random.default_rng(random_state)

    optimizer = SGDMomentum([w], lr=lr, momentum=momentum, weight_decay=weight_decay)

    quality = None
    history = []

    for epoch in range(epochs):
        if order_mode == "random":
            order = rng.permutation(len(X))
        elif order_mode == "margin":
            margins = y * (X @ w.data)
            order = np.argsort(np.abs(margins))
        else:
            raise ValueError("Неизвестный способ предъявления объектов")

        for i in order:
            x_i = Tensor(X[i:i + 1])
            y_i = Tensor(y[i:i + 1])

            optimizer.zero_grad()

            loss = quadratic_loss(x_i, y_i, w)

            loss.backward()
            optimizer.step()

            quality = recursive_Q(quality, loss.data.item(), lambda_)

        history.append(quality)

    return w, history


# реализовать L2 регуляризацию;

def l2_regularization(w: Tensor, weight_decay: float) -> float:
    return 0.5 * weight_decay * np.sum(w.data[1:] ** 2)


# реализовать скорейший градиентный спуск;

def find_best_step(X, y, w, gradient, weight_decay):
    steps = np.logspace(-5, 0, 20)

    original_weights = w.data.copy()

    best_step = steps[0]
    best_loss = float("inf")

    for step in steps:
        w.data = original_weights - step * gradient

        loss = quadratic_loss(Tensor(X), Tensor(y), w)

        regularization = l2_regularization(w, weight_decay)
        loss_value = loss.data.item() + regularization

        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step

    w.data = original_weights

    return best_step


def train_steepest(X, y, w, epochs=100, weight_decay=0.001, lambda_=0.01):
    quality = None
    history = []

    for epoch in range(epochs):
        w.zero_grad()
        loss = quadratic_loss(Tensor(X), Tensor(y), w)
        loss.backward()

        gradient = w.grad.copy()
        regularization = weight_decay * w.data
        regularization[0] = 0.0
        gradient += regularization

        step = find_best_step(X, y, w, gradient, weight_decay)
        w.data -= step * gradient

        quality = recursive_Q(quality, loss.data.item(), lambda_)

        history.append(quality)

    return w, history


# реализовать предъявление объектов по модулю отступа;

def margin_order(X, y, w):
    margins = y * (X @ w.data)
    return np.argsort(np.abs(margins))


# обучить линейный классификатор на выбранном датасете;

from weights import random_init

#     обучить со случайным предъявлением;

initial_w = random_init(X_train.shape[1], rng=np.random.default_rng(42))

w = Tensor(initial_w)

w, history = train_sgd_momentum(X_train, y_train, w, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.001, order_mode="random")


#     обучить с инициализацией весов через корреляцию;

from weights import correlation_init

correlation_w = correlation_init(X_train, y_train)

w_correlation = Tensor(correlation_w)

w_correlation, history_correlation = train_sgd_momentum(X_train, y_train, w_correlation, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.001, order_mode="random")


#     обучить со случайной инициализацией весов через мультистарт;

best_multistart_w = None
best_multistart_Q = float("inf")
best_multistart_history = None

rng = np.random.default_rng(42)

for start in range(10):
    initial_w = random_init(X_train.shape[1], rng=rng)

    current_w = Tensor(initial_w)

    current_w, current_history = train_sgd_momentum(X_train, y_train, current_w, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.001, order_mode="random", random_state=start)

    current_Q = Q(X_train, y_train, current_w)

    if current_Q < best_multistart_Q:
        best_multistart_Q = current_Q
        best_multistart_w = current_w
        best_multistart_history = current_history


#     обучить с предъявлением объектов по модулю отступа;

initial_w = random_init(X_train.shape[1], rng=np.random.default_rng(42))

w_margin = Tensor(initial_w)

w_margin, history_margin = train_sgd_momentum(X_train, y_train, w_margin, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.001, order_mode="margin")


initial_w = random_init(X_train.shape[1], rng=np.random.default_rng(42))

w_steepest = Tensor(initial_w)

w_steepest, history_steepest = train_steepest(X_train, y_train, w_steepest, epochs=100, weight_decay=0.001)


# оценить качество классификации;

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(X, y, w, name):
    prediction = a(Tensor(X), w)

    accuracy = accuracy_score(y, prediction)
    precision = precision_score(y, prediction, pos_label=1, zero_division=0)
    recall = recall_score(y, prediction, pos_label=1, zero_division=0)
    f1 = f1_score(y, prediction, pos_label=1, zero_division=0)

    print()
    print(name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


results = []

results.append(evaluate(X_test, y_test, w, "SGD Momentum"))
results.append(evaluate(X_test, y_test, w_correlation, "Correlation initialization"))
results.append(evaluate(X_test, y_test, best_multistart_w, "Random multistart"))
results.append(evaluate(X_test, y_test, w_margin, "Margin presentation"))
results.append(evaluate(X_test, y_test, w_steepest, "Steepest gradient descent"))


# Визуализировать и проанализировать отступы;

custom_models = [
    ("SGD Momentum", w),
    ("Correlation initialization", w_correlation),
    ("Random multistart", best_multistart_w),
    ("Margin presentation", w_margin),
    ("Steepest gradient descent", w_steepest)
]

best_name = None
best_w = None
best_train_accuracy = -1.0

for name, current_w in custom_models:
    prediction = a(Tensor(X_train), current_w)
    accuracy = accuracy_score(y_train, prediction)

    if accuracy > best_train_accuracy:
        best_train_accuracy = accuracy
        best_name = name
        best_w = current_w

print()
print("Лучшая собственная модель:", best_name)

analyze_margins(X_test, y_test, best_w)
plot_margins(X_test, y_test, best_w)


# сравнить лучшую реализацию с эталонной;

from sklearn.linear_model import RidgeClassifier

reference_model = RidgeClassifier(alpha=0.001)
reference_model.fit(X_train[:, 1:], y_train)

reference_prediction = reference_model.predict(X_test[:, 1:])

reference_result = {
    "model": "sklearn RidgeClassifier",
    "accuracy": accuracy_score(y_test, reference_prediction),
    "precision": precision_score(y_test, reference_prediction, pos_label=1, zero_division=0),
    "recall": recall_score(y_test, reference_prediction, pos_label=1, zero_division=0),
    "f1": f1_score(y_test, reference_prediction, pos_label=1, zero_division=0)
}

results.append(reference_result)

print()
print("Эталонная модель")
print("Accuracy:", reference_result["accuracy"])
print("Precision:", reference_result["precision"])
print("Recall:", reference_result["recall"])
print("F1:", reference_result["f1"])


# Итоговая таблица результатов;

results_df = pd.DataFrame(results)

print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))


# Визуализировать рекуррентную оценку функционала качества;

plt.figure(figsize=(10, 5))

plt.plot(history, label="SGD Momentum")
plt.plot(history_correlation, label="Correlation")
plt.plot(best_multistart_history, label="Multistart")
plt.plot(history_margin, label="Margin")
plt.plot(history_steepest, label="Steepest")

plt.xlabel("Эпоха")
plt.ylabel("Q")
plt.title("Рекуррентная оценка функционала качества")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "quality.png")
plt.close()


# подготовить отчет. - см. README
