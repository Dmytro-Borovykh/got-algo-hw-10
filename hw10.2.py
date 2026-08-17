"""
Обчислення визначеного інтеграла функції f(x) = x^2 на відрізку [0, 2]
методом Монте-Карло з перевіркою через scipy.integrate.quad.
"""

import numpy as np
import scipy.integrate as spi
import matplotlib
matplotlib.use("Agg")           # робота без графічного дисплея
import matplotlib.pyplot as plt


def f(x):
    return x ** 2


A, B = 0, 2                          # межі інтегрування
ANALYTICAL = (B ** 3 - A ** 3) / 3   # аналітичне значення: x^3/3 від 0 до 2 = 8/3


# ------------------------------------------------- метод Монте-Карло

def monte_carlo_hit_or_miss(func, a: float, b: float, n: int, seed: int = None) -> float:
    """
    Метод «влучив або промахнувся».

    Будуємо прямокутник [a, b] x [0, y_max], кидаємо в нього n випадкових точок
    і рахуємо частку тих, що потрапили під криву. Площа під кривою дорівнює
    площі прямокутника, помноженій на цю частку.
    """
    rng = np.random.default_rng(seed)

    x_grid = np.linspace(a, b, 10_000)
    y_max = func(x_grid).max() * 1.05        # невеликий запас над максимумом

    x = rng.uniform(a, b, n)
    y = rng.uniform(0, y_max, n)

    hits = np.count_nonzero(y < func(x))
    return (b - a) * y_max * hits / n


def monte_carlo_mean_value(func, a: float, b: float, n: int, seed: int = None) -> float:
    """
    Метод середнього значення.

    Інтеграл дорівнює довжині відрізка, помноженій на середнє значення функції
    у випадкових точках цього відрізка. Точніший за hit-or-miss за тієї ж
    кількості точок, оскільки використовує саме значення функції,
    а не бінарний факт влучання.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, n)
    return (b - a) * func(x).mean()


# -------------------------------------------------------- перевірка

def compare(n_values: list) -> None:
    """Порівняння точності обох варіантів Монте-Карло при зростанні n."""
    quad_result, quad_error = spi.quad(f, A, B)

    print(f"Аналітичне значення:  {ANALYTICAL:.10f}   (8/3)")
    print(f"scipy quad:           {quad_result:.10f}   (похибка {quad_error:.2e})")
    print()
    print(f"{'n':>10} | {'hit-or-miss':>14} | {'похибка, %':>11} | "
          f"{'середнє знач.':>14} | {'похибка, %':>11}")
    print("-" * 74)

    for n in n_values:
        hm = monte_carlo_hit_or_miss(f, A, B, n, seed=42)
        mv = monte_carlo_mean_value(f, A, B, n, seed=42)
        err_hm = abs(hm - ANALYTICAL) / ANALYTICAL * 100
        err_mv = abs(mv - ANALYTICAL) / ANALYTICAL * 100
        print(f"{n:>10} | {hm:>14.6f} | {err_hm:>10.4f}% | {mv:>14.6f} | {err_mv:>10.4f}%")


def convergence_check(n: int, trials: int = 200) -> None:
    """
    Емпірична перевірка швидкості збіжності.

    Теорія стверджує, що похибка Монте-Карло спадає як 1/sqrt(n).
    Порівнюємо стандартне відхилення оцінок при n та при 4n:
    воно має зменшитись приблизно вдвічі.
    """
    print(f"\n{'=' * 74}")
    print(f"ПЕРЕВІРКА ЗБІЖНОСТІ ({trials} незалежних запусків на кожне n)")
    print(f"{'=' * 74}")
    print(f"{'n':>10} | {'середнє':>12} | {'ст. відхилення':>16} | {'відношення':>12}")
    print("-" * 60)

    prev_std = None
    for k in range(4):
        current_n = n * (4 ** k)
        estimates = [monte_carlo_mean_value(f, A, B, current_n, seed=s) for s in range(trials)]
        mean = np.mean(estimates)
        std = np.std(estimates)
        ratio = f"{prev_std / std:.2f}x" if prev_std else "-"
        print(f"{current_n:>10} | {mean:>12.6f} | {std:>16.6f} | {ratio:>12}")
        prev_std = std


def make_plot(filename: str = "monte_carlo_plot.png", n: int = 2000) -> None:
    """Графік функції з областю інтегрування та випадковими точками."""
    rng = np.random.default_rng(42)

    x = np.linspace(-0.5, 2.5, 400)
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "r", linewidth=2, label="f(x) = x²")

    ix = np.linspace(A, B)
    ax.fill_between(ix, f(ix), color="gray", alpha=0.3, label="область інтегрування")

    # Випадкові точки: під кривою та над нею
    y_max = f(np.linspace(A, B, 1000)).max() * 1.05
    px = rng.uniform(A, B, n)
    py = rng.uniform(0, y_max, n)
    under = py < f(px)
    ax.scatter(px[under], py[under], s=1, color="green", alpha=0.4, label="влучили")
    ax.scatter(px[~under], py[~under], s=1, color="blue", alpha=0.4, label="промахнулись")

    ax.axvline(x=A, color="gray", linestyle="--")
    ax.axvline(x=B, color="gray", linestyle="--")
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, max(y) + 0.1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Метод Монте-Карло: інтеграл f(x) = x² від {A} до {B}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)

    fig.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"\nГрафік збережено: {filename}")


if __name__ == "__main__":
    compare([100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    convergence_check(1_000)
    make_plot()
