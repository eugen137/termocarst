import numpy as np
from scipy import optimize


class Zone:
    def __init__(self, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([-1, 1]),
                 beta=np.array([-1, 1]), ksi=np.array([-0.05, 0.05])):
        self.alpha = alpha
        self.beta = beta
        self.ksi = ksi

        self.square = square
        self.years_square = years_square
        self.temperature = temperature
        self.precipitation = precipitation
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
        self.recip_y = self.norm_precipitation[y]
        self.theta = None
        r = 1000
        self.bounds = optimize.Bounds(-r * np.ones_like(self.norm_square),
                                      r * np.ones_like(self.norm_square))

    def g_m(self, theta_m):
        return (np.exp((-self.ksi[0] * theta_m).astype(np.float128)) * theta_m) * (self.ksi[0] * theta_m + 1) - \
               (np.exp((-self.ksi[1] * theta_m).astype(np.float128)) * theta_m) * (self.ksi[1] * theta_m + 1) / \
               theta_m * (np.exp((-self.ksi[0] * theta_m).astype(np.float128)) -
                          np.exp((-self.ksi[1] * theta_m).astype(np.float128)))

    def k(self, theta):
        return (np.exp((-self.beta[0] * self.h_r(theta)).astype(np.float128)) * (self.beta[0] * self.h_r(theta) + 1) -
                np.exp((-self.beta[1] * self.h_r(theta)).astype(np.float128)) * (self.beta[1] * self.h_r(theta) + 1)) / \
               self.h_r(theta) * (np.exp((-self.beta[0] * self.h_r(theta)).astype(np.float128)) -
                                  np.exp((-self.beta[1] * self.h_r(theta)).astype(np.float128)))

    def l(self, theta):
        return (np.exp((-self.alpha[0] * self.l_r(theta)).astype(np.float128)) * (self.alpha[0] * self.l_r(theta) + 1) -
                np.exp((-self.alpha[1] * self.l_r(theta)).astype(np.float128)) * (self.alpha[1] * self.l_r(theta) + 1)) / \
               self.l_r(theta) * (np.exp((-self.alpha[0] * self.l_r(theta)).astype(np.float128)) -
                                  np.exp((-self.alpha[1] * self.l_r(theta)).astype(np.float128)))

    def h_r(self, theta):
        return np.sum(np.multiply(theta, self.recip_y))

    def l_r(self, theta):
        return np.sum(np.multiply(theta, self.temp_y))

    def func(self, theta):
        n_points = len(self.square)
        l_theta = self.l(theta)
        k_beta = self.k(theta)
        f = np.zeros_like(self.square)
        for i in range(0, n_points):
            f[i] = l_theta * self.temp_y[i] + k_beta * self.recip_y[i] + \
                   self.g_m(theta[i]) - self.norm_square[i]
        return np.sum(np.abs(f))

    def theta_calc(self):
        sol = optimize.differential_evolution(self.func,
                                              bounds=self.bounds,
                                              maxiter=int(10e+10),
                                              tol=10e-6,
                                              workers=-1,
                                              popsize=70,
                                              updating='deferred',
                                              strategy='best2exp')
        print(sol)
