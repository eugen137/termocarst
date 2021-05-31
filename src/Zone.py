from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize


class TypesOfParameters(Enum):
    TEMPERATURE = 1
    PRECIPITATIONS = 2
    ERRORS = 3


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
        self.num_years_with_square = y

        # нормировка
        self.norm_precipitation = (precipitation - np.min(precipitation)) / \
                                  (np.max(precipitation) - np.min(precipitation))
        self.norm_temp = (temperature - np.min(temperature)) / \
                         (np.max(temperature) - np.min(temperature))
        self.norm_square = (square - np.min(square)) / \
                           (np.max(square) - np.min(square))

        self.temp_y = self.norm_temp[y]
        self.precip_y = self.norm_precipitation[y]
        self.theta = None
        r = 100
        self.bounds = optimize.Bounds(-r * np.ones_like(self.norm_square),
                                      r * np.ones_like(self.norm_square))
        self.restored_square = None

    def g_m(self, theta_m):
        return (np.exp(-self.ksi[0] * theta_m) * theta_m) * (self.ksi[0] * theta_m + 1) - \
               (np.exp(-self.ksi[1] * theta_m) * theta_m) * (self.ksi[1] * theta_m + 1) / \
               theta_m * (np.exp(-self.ksi[0] * theta_m) -
                          np.exp(-self.ksi[1] * theta_m))

    def k(self, theta):
        hr = (self.h_r(theta)).astype(np.float64)
        return (np.exp((-self.beta[0] * hr)) * (self.beta[0] * hr + 1) -
                np.exp((-self.beta[1] * hr)) * (self.beta[1] * hr + 1)) / hr * (np.exp(-self.beta[0] * hr) -
                                                                                np.exp(-self.beta[1] * hr))

    def l(self, theta):
        lr = (self.l_r(theta))
        return (np.exp(-self.alpha[0] * lr) * (self.alpha[0] * lr + 1) -
                np.exp(-self.alpha[1] * lr) * (self.alpha[1] * lr + 1)) / lr * (np.exp(-self.alpha[0] * lr) -
                                                                                np.exp(-self.alpha[1] * lr))

    def h_r(self, theta):
        return np.sum(np.multiply(theta, self.precip_y))

    def l_r(self, theta):
        return np.sum(np.multiply(theta, self.temp_y))

    def func(self, theta):
        n_points = self.square.shape
        f = np.zeros_like(self.square)
        # f = 0
        for i in range(0, n_points[0]):
            f[i] = np.abs(self.l(theta) * self.temp_y[i] + self.k(theta) * self.precip_y[i] +
                          self.g_m(theta[i]) - self.norm_square[i])
        return f

    def theta_calc(self):
        sol = optimize.root(self.func, np.ones_like(self.square), method="lm")
        print("success=", sol.success)
        if sol.success:
            self.theta = sol.x
            return sol.x
        else:
            return None

    def modeling(self, n=100):
        s_mean = np.zeros_like(self.norm_temp)
        s_m = np.ones((len(self.norm_temp), n))
        for j in range(0, n):
            for i in range(0, len(self.norm_temp)):
                s_m[i, j] = self.value_from_prv(TypesOfParameters.TEMPERATURE) * self.norm_temp[i] + \
                            self.value_from_prv(TypesOfParameters.PRECIPITATIONS) * self.norm_temp[i] + \
                            self.value_from_prv(TypesOfParameters.ERRORS, i)

        for i in range(0, len(self.temperature)):
            for j in range(0, n):
                if i in self.num_years_with_square:
                    s_mean[i] += self.norm_square[np.where(self.num_years_with_square == i)]
                else:
                    s_mean[i] += s_m[i, j]
            s_mean[i] = s_mean[i] / n
        self.restored_square = np.min(self.square) + s_mean * (np.max(self.square) - np.min(self.square))

    def value_from_prv(self, type_of_parameter: TypesOfParameters, num=0):
        n = None
        w = None
        if self.theta is None:
            return None
        if type_of_parameter == TypesOfParameters.TEMPERATURE:
            d = self.temp_y
            alpha_beta = self.alpha
        elif type_of_parameter == TypesOfParameters.PRECIPITATIONS:
            d = self.precip_y
            alpha_beta = self.beta
        elif type_of_parameter == TypesOfParameters.ERRORS:
            n = np.max(np.where(np.array(self.num_years_with_square <= num)))
            d = None
            alpha_beta = self.ksi
            w = np.exp(-alpha_beta[0] * self.theta[n]) * self.theta[n] / (np.exp(-alpha_beta[0] * self.theta[n]) -
                                                                          np.exp(-alpha_beta[1] * self.theta[n]))
        else:
            return None

        if type_of_parameter != TypesOfParameters.ERRORS:
            l_r = np.sum(np.multiply(self.theta, d))
            ro = (np.exp(-alpha_beta[0] * l_r) - np.exp(-alpha_beta[1] * l_r)) / l_r
            w = np.exp(-alpha_beta[0] * l_r) / ro
        x1 = None
        f = False
        while not f:
            x1 = np.random.rand()
            x2 = np.random.rand()
            x1_ = alpha_beta[0] + x1 * (alpha_beta[1] - alpha_beta[0])
            x2_ = w * x2
            if type_of_parameter == TypesOfParameters.ERRORS:
                f = x2_ <= np.exp(-x1_ * self.theta[n]) * self.theta[n] / (np.exp(-self.ksi[0] * self.theta[n]) -
                                                                           np.exp(-self.ksi[1] * self.theta[n]))
            else:
                f = x2_ <= np.exp(-x1_ * l_r) / ro
        return x1

    def draw(self):
        plt.plot(self.restored_square, c="b")
        q = []
        for i in range(0, len(self.temperature)):
            if i not in self.num_years_with_square:
                q.append(i)
        plt.plot(q,
                 self.restored_square[q], 'g^')

        plt.show()