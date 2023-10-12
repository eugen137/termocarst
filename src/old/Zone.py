import numpy as np
from scipy import optimize


class Zone:
    def __init__(self, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([-1, 1]),
                 beta=np.array([-1, 1]), ksi=np.array([-0.5, 0.5])):
        self.alpha = alpha.astype(np.float64)
        self.beta = beta.astype(np.float64)
        self.ksi = ksi.astype(np.float64)

        self.square = square.astype(np.float64)
        self.years_square = years_square
        self.temperature = temperature.astype(np.float64)
        self.precipitation = precipitation.astype(np.float64)
        self.years = years
        y = years_square - np.min(years)
        y = y.astype(int)
        print(y)

        # нормировка
        self.norm_precipitation = (precipitation - np.min(precipitation)) / \
                                  (np.max(precipitation) - np.min(precipitation))
        self.norm_temp = (temperature - np.min(temperature)) / \
                         (np.max(temperature) - np.min(temperature))
        self.norm_square = (square - np.min(square)) / \
                           (np.max(square) - np.min(square))

        self.temp_y = self.norm_temp[y]
        self.perecip_y = self.norm_precipitation[y]
        self.theta = None
        r = 100
        # print(self.norm_square)
        # print(self.temp_y)
        # print(self.perecip_y)
        self.bounds = optimize.Bounds(-r * np.ones_like(self.norm_square),
                                      r * np.ones_like(self.norm_square))

    def g_m(self, theta_m):
        return (np.exp(-self.ksi[0] * theta_m) * theta_m) * (self.ksi[0] * theta_m + 1) - \
               (np.exp(-self.ksi[1] * theta_m) * theta_m) * (self.ksi[1] * theta_m + 1) / \
               theta_m * (np.exp(-self.ksi[0] * theta_m) -
                          np.exp(-self.ksi[1] * theta_m))

    def k(self, theta):
        hr = (self.h_r(theta)).astype(np.float64)
        return (np.exp((-self.beta[0] * hr)) * (self.beta[0] * hr + 1) -
                np.exp((-self.beta[1] * hr)) * (self.beta[1] * hr + 1)) / \
               hr * (np.exp(-self.beta[0] * hr) -
                     np.exp(-self.beta[1] * hr))

    def l(self, theta):
        lr = (self.l_r(theta))
        return (np.exp(-self.alpha[0] * lr) * (self.alpha[0] * lr + 1) -
                np.exp(-self.alpha[1] * lr) * (self.alpha[1] * lr + 1)) / \
               lr * (np.exp(-self.alpha[0] * lr) -
                     np.exp(-self.alpha[1] * lr))

    def h_r(self, theta):
        return np.sum(np.multiply(theta, self.perecip_y))

    def l_r(self, theta):
        return np.sum(np.multiply(theta, self.temp_y))

    def func(self, theta):
        n_points = self.square.shape
        f = np.zeros_like(self.square)
        # f = 0
        for i in range(0, n_points[0]):
            f[i] = np.abs(self.l(theta) * self.temp_y[i] + self.k(theta) * self.perecip_y[i] +
                        self.g_m(theta[i]) - self.norm_square[i])
        return f

    def theta_calc(self):
        sol = optimize.root(self.func, np.ones_like(self.square), method="lm")
        print(sol.success)
        print(self.func(sol.x))
        if sol.success:
            self.theta = sol.x
            return sol.x
        else:
            return None

    def modeling(self, n):
        pass

    def value_from_prv(self, t):
