import numpy as np
from scipy.stats import truncnorm


def calculate_table(mu, S, alpha, B):
    percentiles_T = 1. - np.exp(np.linspace(np.log(0.01), np.log(1.), 21)).round(3)[::-1] + 0.01 # percentiles for T from 5% to 95%
    T_values = mu + np.sqrt(S) * truncnorm.ppf(percentiles_T, (0. - mu) / np.sqrt(S),
                                                  (1 - mu) / np.sqrt(S)).round(3)  # calculate corresponding T values
    E_percentiles = truncnorm.cdf((T_values - alpha) / np.sqrt(B), (0.1 - alpha) / np.sqrt(B),
                                  (1 - alpha) / np.sqrt(B)).round(3)  # calculate corresponding E percentiles
    return np.column_stack((np.arange(T_values.size), percentiles_T, E_percentiles))  # combine and return as a 2D array


if __name__ == '__main__':
    mu = 0.5  # fill in with your desired value
    S = 0.25 ** 2  # fill in with your desired value
    alpha = 0.6
    B = 0.4 ** 2

    table = calculate_table(mu, S, alpha, B)
    print(table)
