import logging
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from src.utils import normalize


class TypesOfParameters(Enum):
    TEMPERATURE = 1
    PRECIPITATIONS = 2
    ERRORS = 3


class RandomizeRestoring:
    def __init__(self, task_id, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([-1, 1]),
                 beta=np.array([-1, 1]), ksi=np.array([-0.5, 0.5])):
        """
        Класс рандомизированного восстановления пропусков в данных о площади озер.
        :param square: Вектор с данными о площади
        :param years_square: Вектор с данными годов, в которых есть значения площади
        :param temperature: Вектор с данными о температуре
        :param precipitation: Вектор с данными об осадках
        :param years: Вектор с данными годов
        :param alpha: Промежуток параметра Температуры
        :param beta: Промежуток параметра Осадков
        :param ksi: Промежуток параметра Ошибки
        """
        self.id = task_id
        logging.info("ID={}. Начато восстановление RandomizeRestoring".format(self.id), extra={"task_id": self.id})
        self.alpha = alpha.astype(np.float64)
        self.beta = beta.astype(np.float64)
        self.ksi = ksi.astype(np.float64)

        self.square = square.astype(np.float64)
        self.years_square = years_square
        self.temperature = temperature.astype(np.float64)
        self.precipitation = precipitation.astype(np.float64)
        self.years = years
        self.num_years_with_square = (years_square - np.min(years)).astype(int)

        logging.info("ID={}. Нормировка данных".format(self.id), extra={"task_id": self.id})
        # нормировка
        self.norm_precipitation = normalize(precipitation)
        self.norm_temp = normalize(temperature)
        self.norm_square = normalize(square)
        self.temp_norm_yws = self.norm_temp[self.num_years_with_square]
        self.precip_norm_yws = self.norm_precipitation[self.num_years_with_square]
        self.theta = None
        r = 100
        self.bounds = optimize.Bounds(-r * np.ones_like(self.norm_square),
                                      r * np.ones_like(self.norm_square))
        self.restored_square = None
        self.params_gaps()

    def params_gaps(self):
        logging.info("ID={}. Начато вычисление промежутков параметров".format(self.id))
        # создание матрицы для вычисления промежутков
        x = np.ones_like(self.temp_norm_yws).reshape(-1, 1)
        x = np.hstack((x, self.temp_norm_yws.reshape(-1, 1), self.precip_norm_yws.reshape(-1, 1)))
        # вычисляются оценки МНК
        a = np.linalg.inv(np.mat(np.transpose(x)) * np.mat(np.mat(x))) * np.mat(np.transpose(x)) * \
            (np.mat(self.norm_square.reshape(-1, 1)))
        e = np.mat(x) * np.mat(a)
        s2 = np.sum(np.power(e - self.norm_square.reshape(-1, 1), 2)) / len(self.square - 2)
        q = np.mat(np.transpose(x)) * np.mat(np.mat(x))
        self.alpha = [a[0] - 3 * np.sqrt(s2 * q[0, 0]), a[0] + 3 * np.sqrt(s2 * q[0, 0])]
        logging.info("ID={}. Промежутки параметров для температуры {}".format(self.id, self.alpha))
        self.beta = [a[1] - 3 * np.sqrt(s2 * q[1, 1]), a[1] + 3 * np.sqrt(s2 * q[1, 1])]
        logging.info("ID={}. Промежутки параметров для осадков {}".format(self.id, self.beta))
        self.ksi = [a[2] - 3 * np.sqrt(s2 * q[2, 2]), a[2] + 3 * np.sqrt(s2 * q[2, 2])]
        logging.info("ID={}. Промежутки параметров для ошибки {}".format(self.id, self.ksi))
        logging.info("ID={}. Окончено вычисление промежутков параметров".format(self.id))

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

    def el(self, theta):
        lr = (self.l_r(theta))
        return (np.exp(-self.alpha[0] * lr) * (self.alpha[0] * lr + 1) -
                np.exp(-self.alpha[1] * lr) * (self.alpha[1] * lr + 1)) / lr * (np.exp(-self.alpha[0] * lr) -
                                                                                np.exp(-self.alpha[1] * lr))

    def h_r(self, theta):
        return np.sum(np.multiply(theta, self.precip_norm_yws))

    def l_r(self, theta):
        return np.sum(np.multiply(theta, self.temp_norm_yws))

    def func(self, theta):
        n_points = self.square.shape
        f = np.zeros_like(self.square)
        for i in range(0, n_points[0]):
            f[i] = np.abs(self.el(theta) * self.temp_norm_yws[i] + self.k(theta) * self.precip_norm_yws[i] +
                          self.g_m(theta[i]) - self.norm_square[i])
        return f

    def theta_calc(self):
        logging.info("ID={}. Вычисление множителей Лагранжа".format(self.id))
        sol = optimize.root(self.func, np.ones_like(self.square), method="lm")
        if sol.success:
            self.theta = sol.x
            logging.info("ID={}. Успешно окончено вычисление множителей Лагранжа".format(self.id))
            return sol.x
        else:
            logging.error("ID={}. Вычисление множителей Лагранжа окончено неудачей".format(self.id))
            return None

    def modeling(self, n=100):
        logging.info("ID={}. Начато моделирование".format(self.id))
        s_mean = np.zeros_like(self.norm_temp)
        s_m = np.ones((len(self.norm_temp), n))
        for j in range(0, n):
            # logging.info("Сэмплирование, шаг {}".format(j))
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
        square_dict = dict()
        for i in range(0, len(self.years)):
            square_dict[self.years[i]] = self.restored_square[i]
        logging.info("ID={}. Окончено моделирование".format(self.id))
        return square_dict

    def value_from_prv(self, type_of_parameter: TypesOfParameters, num=0):
        n = None
        w = None
        if self.theta is None:
            return None
        if type_of_parameter == TypesOfParameters.TEMPERATURE:
            d = self.temp_norm_yws
            alpha_beta = self.alpha
        elif type_of_parameter == TypesOfParameters.PRECIPITATIONS:
            d = self.precip_norm_yws
            alpha_beta = self.beta
        elif type_of_parameter == TypesOfParameters.ERRORS:
            n = np.max(np.where(np.array(self.num_years_with_square <= num)))
            d = None
            alpha_beta = self.ksi
            def p(x): return np.exp(-x * self.theta[n]) * self.theta[n] / (np.exp(-self.ksi[0] * self.theta[n]) -
                                                                           np.exp(-self.ksi[1] * self.theta[n]))
            z = -1000
            c = 0
            for i in np.arange(-1, 1, 0.001):
                c = p(i)
                if z < c:
                    z = c
            w = c
        else:
            return None

        if type_of_parameter != TypesOfParameters.ERRORS:

            l_r = np.sum(np.multiply(self.theta, d))
            ro = (np.exp(-alpha_beta[0] * l_r) - np.exp(-alpha_beta[1] * l_r)) / l_r

            def p(x): return np.exp(-x * l_r) / ro
            z = -1000
            for i in np.arange(-1, 1, 0.001):
                c = p(i)
                if z < c:
                    z = c
            w = c
        while True:
            x1 = np.random.rand()
            x2 = np.random.rand()
            x1_ = alpha_beta[0] + x1 * (alpha_beta[1] - alpha_beta[0])
            if w * x2 <= p(x1_):
                break
        return x1

    def draw(self):
        q = []
        for i in range(0, len(self.temperature)):
            if i not in self.num_years_with_square:
                q.append(i)
        p_sf = np.polyfit(self.years_square, self.square, 1)
        f_sf = np.polyval(p_sf, self.years)

        p_middle_f = np.polyfit(self.years, self.restored_square, 1)
        f_middle_f = np.polyval(p_middle_f, self.years)

        fig, axs = plt.subplots(3)
        plt.subplots_adjust(hspace=0.6)

        axs[0].plot(self.years, self.temperature)
        axs[0].set_title("Температура")

        axs[1].plot(self.years, self.precipitation)
        axs[1].set_title("Осадки")

        axs[2].plot(self.years, self.restored_square, 'g', self.years, f_sf, 'r', self.years, f_middle_f, 'g',
                    q + self.years[0], self.restored_square[q], '*r')
        axs[2].set_title("Площадь")
        axs[2].set_xlabel('x_label', size=10)
        plt.show()
