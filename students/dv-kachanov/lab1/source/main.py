import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent

EPOCHS = int(os.environ.get("LAB1_EPOCHS", "100"))
MULTISTART_N = int(os.environ.get("LAB1_MULTISTART_N", "10"))

import matplotlib

matplotlib.use("Agg")

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


# Канон обозначений — лекция 1: X^l = {(x_i, y_i)}, y_i in {-1, +1};
# a(x, w) = sign(<w, x>); M_i(w) = g(x_i, w) * y_i;
# Q(M) = (1 - M)^2; Q(w) = SUM_i L(w, x_i) -> min_w (сумма без 1/l).

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


# Константный признак f_0(x) = -1 (лекция 1, модель нейрона МакКаллока-Питтса).
X_train = np.column_stack((-np.ones(len(X_train)), X_train))
X_test = np.column_stack((-np.ones(len(X_test)), X_test))


def g(X: Tensor, w: Tensor) -> Tensor:
    return X @ w


def a(X: Tensor, w: Tensor) -> np.ndarray:
    return np.where(g(X, w).data >= 0, 1.0, -1.0)


def M_i(X: Tensor, y: Tensor, w: Tensor) -> Tensor:
    """Отступ M_i(w) = g(x_i, w) * y_i; M_i(w) < 0 <=> ошибка на x_i."""
    return g(X, w) * y


def err_rate(X: np.ndarray, y: np.ndarray, w: Tensor) -> float:
    """Доля ошибок (бинарная потеря, НЕ эмпирический риск Q)."""
    return float(np.mean(a(Tensor(X), w) != y))


def Q(X: np.ndarray, y: np.ndarray, w: Tensor) -> float:
    """Эмпирический риск Q(w) = SUM_i (1 - M_i(w))^2 (сумма без 1/l)."""
    margins = y * (X @ w.data)
    return float(np.sum((1.0 - margins) ** 2))


def tilde_L(x_i: Tensor, y_i: Tensor, w: Tensor, tau: float) -> Tensor:
    """Регуляризованная потеря одного объекта в графе автодифференцирования:

    ~L(w, x_i) = L(w, x_i) + tau/2 * ||w||^2,
    L(w, x_i) = Q(M_i) = (1 - M_i)^2, M_i = <w, x_i> * y_i.

    Штраф tau/2 * SUM_{j=0}^{n} w_j^2 считается внутри графа по ВСЕМ
    компонентам, включая w_0, поэтому grad содержит слагаемое tau * w
    и шаг w := w - h * grad эквивалентен w := w(1 - h*tau) - h*grad L.
    """
    margin = M_i(x_i, y_i, w)
    residual = 1.0 + (-margin)
    loss = (residual * residual).sum()
    penalty = (w * w).sum() * (tau / 2.0)
    return loss + penalty


def per_object_tilde_L(x: np.ndarray, label: float, w: np.ndarray, tau: float) -> float:
    """То же ~L(w, x_i), но числом (для инициализации оценки Q)."""
    margin = float(x @ w) * label
    return float((1.0 - margin) ** 2 + (tau / 2.0) * float(w @ w))


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


# Рекуррентная оценка функционала (экспоненциальное скользящее среднее).

def recursive_Q(previous_Q, xi, lam):
    """Q := lam * xi + (1 - lam) * Q, xi := ~L(w, x_i)."""
    if previous_Q is None:
        return xi

    return lam * xi + (1.0 - lam) * previous_Q


def init_Q(X, y, w: Tensor, tau: float, rng, subset_size=50) -> float:
    """Q := среднее ~L(w, x_i) по случайному подмножеству (лекция 1, шаг 2 SG)."""
    subset_size = min(subset_size, len(X))
    idx = rng.choice(len(X), size=subset_size, replace=False)
    values = [
        per_object_tilde_L(X[k], float(y[k]), w.data, tau)
        for k in idx
    ]
    return float(np.mean(values))


# Порядок предъявления объектов (лекция 1).

def sample_order(rng, margins: np.ndarray, mode: str, mu_plus=1.0, mu_minus=-1.0, delta=0.1):
    """Порядок предъявления объектов.

    shuffling: равномерная перетасовка (попеременно объекты разных классов
    в среднем за счет перемешивания).

    margin: вероятностное предпочтение объектов с малым M_i (больше ошибка)
    и малым |M_i| (меньше уверенность): вес u_i = (mu_plus - M_i) / (|M_i| + delta),
    объекты с M_i > mu_plus ("хорошие") и M_i < mu_minus ("выбросы")
    в текущей эпохе вообще не предъявляются. Параметры mu_plus, mu_minus
    подбираются; delta = 0.1 фиксирует вес уверенности при M_i = 0.
    """
    if mode == "shuffling":
        return rng.permutation(len(margins))

    if mode != "margin":
        raise ValueError("Неизвестный способ предъявления объектов")

    eligible = np.where((margins >= mu_minus) & (margins <= mu_plus))[0]
    if len(eligible) == 0:
        return np.empty(0, dtype=int)

    m = margins[eligible]
    u = (mu_plus - m) / (np.abs(m) + delta)
    u = np.maximum(u, 0.0)
    if u.sum() <= 0.0:
        u = np.ones_like(u)
    p = u / u.sum()
    return rng.choice(eligible, size=len(eligible), replace=False, p=p)


# Стохастический градиент с инерцией (лекция 1):
# v := gamma * v + (1 - gamma) * grad ~L(w, x_i); w := w - h * v.

from optimizers import SGDMomentum


def train_sgd_momentum(X, y, w, h=0.01, gamma=0.9, tau=0.001, lam=None, epochs=100, order_mode="shuffling", mu_plus=1.0, mu_minus=-1.0, random_state=42):
    rng = np.random.default_rng(random_state)
    if lam is None:
        lam = 1.0 / len(X)

    optimizer = SGDMomentum([w], h=h, gamma=gamma)

    Q_estimate = init_Q(X, y, w, tau, rng)
    history = []

    for epoch in range(epochs):
        margins = y * (X @ w.data)
        order = sample_order(rng, margins, order_mode, mu_plus=mu_plus, mu_minus=mu_minus)

        for i in order:
            x_i = Tensor(X[i:i + 1])
            y_i = Tensor(y[i:i + 1])

            optimizer.zero_grad()

            xi = tilde_L(x_i, y_i, w, tau)

            xi.backward()
            optimizer.step()

            Q_estimate = recursive_Q(Q_estimate, float(xi.data), lam)

        history.append(Q_estimate)

    return w, history


# Скорейший градиентный спуск (лекция 1):
# L(w - h * grad L(w, x_i), x_i) -> min_h по ОДНОМУ объекту;
# при квадратичной потере h* = ||x_i||^{-2}.
# Шаг делается по регуляризованному градиенту grad ~L(w, x_i)
# (эквивалент w := w(1 - h*tau) - h*grad L), h* — из формулы лекции
# для квадратичной потери.

def train_steepest(X, y, w, tau=0.001, lam=None, epochs=100, random_state=42):
    rng = np.random.default_rng(random_state)
    if lam is None:
        lam = 1.0 / len(X)

    Q_estimate = init_Q(X, y, w, tau, rng)
    history = []

    for epoch in range(epochs):
        order = rng.permutation(len(X))

        for i in order:
            x_i = Tensor(X[i:i + 1])
            y_i = Tensor(y[i:i + 1])

            w.zero_grad()

            xi = tilde_L(x_i, y_i, w, tau)
            xi.backward()

            gradient = w.grad.copy()
            h_star = 1.0 / float(X[i] @ X[i])
            w.data -= h_star * gradient

            Q_estimate = recursive_Q(Q_estimate, float(xi.data), lam)

        history.append(Q_estimate)

    return w, history


# Обучение линейного классификатора на датасете Pima Indians Diabetes.

from weights import zeros_init, random_init, correlation_init

h = 0.01
gamma = 0.9
tau = 0.001
lam = 1.0 / len(X_train)
print(f"h = {h}, gamma = {gamma}, tau = {tau}, lam = {lam:.6f} (1/l, l = {len(X_train)})")

# Обучение со случайным предъявлением (shuffling), случайная инициализация.

w = Tensor(random_init(X_train.shape[1], rng=np.random.default_rng(42)))

w, history = train_sgd_momentum(X_train, y_train, w, h=h, gamma=gamma, tau=tau, lam=lam, epochs=EPOCHS, order_mode="shuffling")


# Обучение с инициализацией весов через корреляцию (точная дробь).

w_correlation = Tensor(correlation_init(X_train, y_train))

w_correlation, history_correlation = train_sgd_momentum(X_train, y_train, w_correlation, h=h, gamma=gamma, tau=tau, lam=lam, epochs=EPOCHS, order_mode="shuffling")


# Обучение со случайной инициализацией весов через мультистарт.

best_multistart_w = None
best_multistart_Q = float("inf")
best_multistart_history = None

rng = np.random.default_rng(42)

for start in range(MULTISTART_N):
    current_w = Tensor(random_init(X_train.shape[1], rng=rng))

    current_w, current_history = train_sgd_momentum(X_train, y_train, current_w, h=h, gamma=gamma, tau=tau, lam=lam, epochs=EPOCHS, order_mode="shuffling", random_state=start)

    current_Q = Q(X_train, y_train, current_w)

    if current_Q < best_multistart_Q:
        best_multistart_Q = current_Q
        best_multistart_w = current_w
        best_multistart_history = current_history


# Обучение с вероятностным предъявлением по отступу (малые M_i и |M_i|,
# отсечения mu_plus / mu_minus), нулевая инициализация.

w_margin = Tensor(zeros_init(X_train.shape[1]))

w_margin, history_margin = train_sgd_momentum(X_train, y_train, w_margin, h=h, gamma=gamma, tau=tau, lam=lam, epochs=EPOCHS, order_mode="margin", mu_plus=1.0, mu_minus=-1.0)


# Скорейший спуск h* = ||x_i||^{-2} по одному объекту, нулевая инициализация.

w_steepest = Tensor(zeros_init(X_train.shape[1]))

w_steepest, history_steepest = train_steepest(X_train, y_train, w_steepest, tau=tau, lam=lam, epochs=EPOCHS)


# Оценка качества классификации.

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
    print("Q (train sum):", Q(X_train, y_train, w))

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


# Визуализировать и проанализировать отступы.

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


# Сравнить лучшую реализацию с эталонной.

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


# Итоговая таблица результатов.

results_df = pd.DataFrame(results)

print()
print("Итоговые результаты:")
print(results_df.to_string(index=False))


# Визуализировать рекуррентную оценку функционала качества.

plt.figure(figsize=(10, 5))

plt.plot(history, label="SGD Momentum")
plt.plot(history_correlation, label="Correlation")
plt.plot(best_multistart_history, label="Multistart")
plt.plot(history_margin, label="Margin")
plt.plot(history_steepest, label="Steepest")

plt.xlabel("Эпоха")
plt.ylabel("Q")
plt.yscale("log")
plt.title("Рекуррентная оценка функционала качества (лог. шкала)")
plt.legend()
plt.grid()

plt.savefig(OUTPUT_DIR / "quality.png")
plt.close()


# Подготовить отчет. - см. README
