import numpy as np
from typing import Sequence

def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
    total_days = len(p)
    dp = np.zeros(total_days + 1)
    dp[0] = 1.0  # probability of 0 rainy days to start

    for pi in p:
        new_dp = dp.copy()
        for j in range(total_days, 0, -1):  # bottom-up dp
            new_dp[j] = dp[j] * (1 - pi) + dp[j - 1] * pi
        new_dp[0] = dp[0] * (1 - pi)
        dp = new_dp

    # Probability of more than n rainy days
    return np.sum(dp[n+1:])

if __name__ == "__main__":
    p = np.random.uniform(0.1, 0.4, size=365)  # 365 random rain probabilities
    n = 100 # number of rainy days
    result = prob_rain_more_than_n(p, n)
    print(f"Probability of more than {n} rainy days: {result}")