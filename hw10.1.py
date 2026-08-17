"""
Дві реалізації видачі решти монетами номіналів [50, 25, 10, 5, 2, 1]:
    - find_coins_greedy: жадібний алгоритм, O(m)
    - find_min_coins: динамічне програмування, O(amount * m)

Використання:
    python cash_register.py
"""

import timeit

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount: int, coins: list = None) -> dict:
    """
    Жадібний алгоритм видачі решти.

    На кожному кроці береться найбільший номінал, що не перевищує залишок.
    Складність O(m), де m це кількість номіналів. Від суми не залежить.
    """
    if coins is None:
        coins = COINS
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    result = {}
    remainder = amount

    for coin in sorted(coins, reverse=True):
        if remainder >= coin:
            count, remainder = divmod(remainder, coin)
            result[coin] = count
        if remainder == 0:
            break

    return result


def find_min_coins(amount: int, coins: list = None) -> dict:
    """
    Динамічне програмування: мінімальна кількість монет на задану суму.

    min_coins[s] це мінімальна кількість монет для суми s,
    last_coin[s] це номінал, доданий останнім (потрібен для відновлення набору).
    Складність O(amount * m) за часом і O(amount) за пам'яттю.
    """
    if coins is None:
        coins = COINS
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    min_coins = [0] + [float("inf")] * amount
    last_coin = [0] * (amount + 1)

    # Заповнюємо таблицю знизу вгору: від суми 1 до потрібної
    for current in range(1, amount + 1):
        for coin in coins:
            if coin <= current and min_coins[current - coin] + 1 < min_coins[current]:
                min_coins[current] = min_coins[current - coin] + 1
                last_coin[current] = coin

    if min_coins[amount] == float("inf"):
        return {}  # суму неможливо скласти наявними номіналами

    # Відновлюємо набір монет, рухаючись назад по last_coin
    result = {}
    remainder = amount
    while remainder > 0:
        coin = last_coin[remainder]
        result[coin] = result.get(coin, 0) + 1
        remainder -= coin

    return dict(sorted(result.items(), reverse=True))


def total_coins(result: dict) -> int:
    """Загальна кількість монет у наборі."""
    return sum(result.values())


def benchmark(amounts: list) -> None:
    """Порівняння часу виконання обох алгоритмів на зростаючих сумах."""
    print(f"\n{'=' * 74}")
    print("ПОРІВНЯННЯ ЧАСУ ВИКОНАННЯ")
    print(f"{'=' * 74}")
    print(f"{'Сума':>10} | {'Жадібний, с':>14} | {'ДП, с':>14} | {'ДП / жадібний':>16}")
    print("-" * 74)

    for amount in amounts:
        t_greedy = min(timeit.Timer(lambda: find_coins_greedy(amount)).repeat(3, 1))
        t_dp = min(timeit.Timer(lambda: find_min_coins(amount)).repeat(3, 1))
        print(f"{amount:>10} | {t_greedy:>14.8f} | {t_dp:>14.8f} | {t_dp / t_greedy:>15.0f}x")


def compare_results(amounts: list) -> None:
    """Перевіряє, чи збігаються набори монет обох алгоритмів."""
    print(f"\n{'=' * 74}")
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
    print(f"{'=' * 74}")
    print(f"{'Сума':>8} | {'Жадібний':>10} | {'ДП':>6} | {'Збіг':>6}")
    print("-" * 42)

    for amount in amounts:
        g = total_coins(find_coins_greedy(amount))
        d = total_coins(find_min_coins(amount))
        print(f"{amount:>8} | {g:>10} | {d:>6} | {'так' if g == d else 'НІ':>6}")


if __name__ == "__main__":
    print(f"Набір номіналів: {COINS}\n")

    for amount in (113, 99, 6):
        print(f"Сума {amount}:")
        print(f"  Жадібний: {find_coins_greedy(amount)}")
        print(f"  ДП:       {find_min_coins(amount)}")

    compare_results([113, 99, 63, 6, 30, 40])
    benchmark([100, 1_000, 10_000, 100_000, 500_000])

    # Контрприклад: набір номіналів, на якому жадібний алгоритм помиляється
    print(f"\n{'=' * 74}")
    print("КОНТРПРИКЛАД: номінали [1, 3, 4], сума 6")
    print(f"{'=' * 74}")
    greedy = find_coins_greedy(6, [1, 3, 4])
    optimal = find_min_coins(6, [1, 3, 4])
    print(f"Жадібний: {greedy}  ->  {total_coins(greedy)} монети")
    print(f"ДП:       {optimal}  ->  {total_coins(optimal)} монети")
