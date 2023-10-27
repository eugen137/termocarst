import logging
import numpy as np
from src.utils import Randomize, TypesOfParameters


class RandomizeRestoring(Randomize):
    def __init__(self, task_id, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([0, 1]),
                 beta=np.array([0, 1]), ksi=np.array([-0.15, 0.15])):
        super().__init__(task_id, square, years_square, temperature,
                         precipitation, years, alpha, beta, ksi)
        self.params_gaps()

    def params_gaps(self):
        logging.info("ID={}. Начато вычисление промежутков параметров".format(self.id))
        # создание матрицы для вычисления промежутков
        # x = np.ones_like(self.temp_norm_yws).reshape(-1, 1)
        x = np.hstack((self.temp_norm_yws.reshape(-1, 1), self.precip_norm_yws.reshape(-1, 1)))
        # вычисляются оценки МНК
        a = np.linalg.inv(np.mat(np.transpose(x)) * np.mat(np.mat(x))) * np.mat(np.transpose(x)) * \
            (np.mat(self.norm_square.reshape(-1, 1)))
        e = np.mat(x) * np.mat(a)
        s2 = np.sum(np.power(e - self.norm_square.reshape(-1, 1), 2)) / (len(self.square) - 2)
        print(a)
        q = np.mat(np.transpose(x)) * np.mat(np.mat(x))
        self.alpha = [a[0,0] - 3 * np.sqrt(s2 * q[0, 0]), a[0,0] + 3 * np.sqrt(s2 * q[0, 0])]
        logging.info("ID={}. Промежутки параметров для температуры {}".format(self.id, self.alpha))
        self.beta = [a[1,0] - 3 * np.sqrt(s2 * q[1, 1]), a[1,0] + 3 * np.sqrt(s2 * q[1, 1])]
        logging.info("ID={}. Промежутки параметров для осадков {}".format(self.id, self.beta))
        self.ksi = [- 3 * np.sqrt(s2), 3 * np.sqrt(s2)]
        logging.info("ID={}. Промежутки параметров для ошибки {}".format(self.id, self.ksi))
        logging.info("ID={}. Окончено вычисление промежутков параметров".format(self.id))

    def func(self, theta):
        n_points = self.square.shape
        f = np.zeros_like(self.square)
        for i in range(0, n_points[0]):
            f[i] = np.abs(self.el(theta) * self.temp_norm_yws[i] + self.k(theta) * self.precip_norm_yws[i] +
                          self.g_m(theta[i]) - self.norm_square[i])
        return f

    def calculate_static_param(self):
        logging.info("ID={}. calculate_static_param".format(self.id))
        self.calculated_l_r = np.sum(np.multiply(self.theta, self.temp_norm_yws))
        self.calculated_ro = (np.exp(-self.alpha[0] * self.calculated_l_r) - np.exp(-self.alpha[1] * self.calculated_l_r)) / self.calculated_l_r
        self.calculated_h_r = np.sum(np.multiply(self.theta, self.precip_norm_yws))
        self.calculated_fo = (np.exp(-self.alpha[0] * self.calculated_h_r) - np.exp(-self.alpha[1] * self.calculated_h_r)) / self.calculated_h_r

        self.calculated_q_err = None
        self.calculated_mean_theta = np.mean(self.theta)

    def modeling(self, n=100):
        self.calculate_static_param()
        logging.info("ID={}. Начато моделирование".format(self.id))
        s_mean = np.zeros_like(self.norm_temp)
        s_m = np.ones((len(self.norm_temp), n))
        for j in range(0, n):
            logging.info("Сэмплирование, шаг {}".format(j))
            for i in range(0, len(self.norm_temp)):
                if i in self.num_years_with_square:
                    continue
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
        self.draw()
        return square_dict


class RandomizeForecasting(Randomize):
    def __init__(self, task_id, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([0, 1]),
                 beta=np.array([0, 1]), ksi=np.array([-0.15, 0.15])):
        self.square_param = []
        super().__init__(task_id, square, years_square, temperature,
                         precipitation, years, alpha, beta, ksi)
        self.order_calc()
        self.params_gaps()

    def order_calc(self):
        logging.info("ID={}. Начато вычисление порядка.".format(self.id))
        # вычисление порядка модели
        s_mean = np.mean(self.norm_square)
        len_s = len(self.norm_square)
        k = 0
        c_0 = np.sum(np.power(self.norm_square - s_mean, 2)) / len_s
        for i in range(0, len_s):
            c_k = 0
            for j in range(0, len_s - i):
                c_k += (self.norm_square[j] - s_mean) * (self.norm_square[j + i] - s_mean)
            c_k = c_k / len_s
            if abs(c_k / c_0) < 0.1:
                k = i
                break
        self.p = k
        logging.info("ID={}. Порядок вычислен и равен {}".format(self.id, self.p))

    def params_gaps(self):
        logging.info("ID={}. Начато вычисление промежутков параметров".format(self.id))
        n = len(self.norm_square) - self.p

        # создание матрицы для вычисления промежутков
        x = np.ones_like(self.norm_square[self.p:]).reshape(-1, 1)
        for i in range(0, self.p):
            x = np.hstack((x, self.norm_square[i:n + i]))
        x = np.hstack((self.temp_norm_yws[self.p:].reshape(-1, 1), self.precip_norm_yws[self.p:].reshape(-1, 1)))

        # вычисляются оценки МНК
        a = np.linalg.inv(np.mat(np.transpose(x)) * np.mat(np.mat(x))) * np.mat(np.transpose(x)) * \
            (np.mat(self.norm_square.reshape(-1, 1)))
        e = np.mat(x) * np.mat(a)
        s2 = np.sum(np.power(e - self.norm_square.reshape(-1, 1), 2)) / (len(self.square) - 2)
        q = np.mat(np.transpose(x)) * np.mat(np.mat(x))

        self.square_param = []
        for i in range(0, self.p):
            self.square_param.append([a[i] - 3 * np.sqrt(s2 * q[i, i]),
                                      a[i] + 3 * np.sqrt(s2 * q[i, i])])

        self.alpha = [a[self.p + 1] - 3 * np.sqrt(s2 * q[self.p + 1, self.p + 1]),
                      a[self.p + 1] + 3 * np.sqrt(s2 * q[self.p + 1, self.p + 1])]
        logging.info("ID={}. Промежутки параметров для температуры {}".format(self.id, self.alpha))
        self.beta = [a[self.p + 1] - 3 * np.sqrt(s2 * q[self.p + 1, self.p + 1]),
                     a[self.p + 1] + 3 * np.sqrt(s2 * q[self.p + 1, self.p + 1])]
        logging.info("ID={}. Промежутки параметров для осадков {}".format(self.id, self.beta))
        self.ksi = [- 3 * np.sqrt(s2), 3 * np.sqrt(s2)]
        logging.info("ID={}. Промежутки параметров для ошибки {}".format(self.id, self.ksi))
        logging.info("ID={}. Окончено вычисление промежутков параметров".format(self.id))

    def ex(self, theta, p):
        lr = np.sum(np.multiply(theta, self.norm_square[p:len(self.norm_square) - p]))
        return (np.exp(-self.square_param[p][0] * lr) * (self.square_param[p] * lr + 1) -
                np.exp(-self.square_param[p][1] * lr) * (self.square_param[p][1] * lr + 1)) / \
            (lr * (np.exp(-self.square_param[p][0] * lr) - np.exp(-self.square_param[p][1] * lr)))

    def func(self, theta):
        n_points = self.square.shape
        f = np.zeros_like(self.square)
        for i in range(0, n_points[0]):
            for j in range(0, self.p):
                f[i] += self.ex(theta, j)
            f[i] = np.abs(f[i] + self.el(theta) * self.temp_norm_yws[i] + self.k(theta) * self.precip_norm_yws[i] +
                          self.g_m(theta[i]) - self.norm_square[i])
        return f

    def modeling(self, n=100, k=10):
        logging.info("ID={}. Начато моделирование".format(self.id))
        s_mean = np.zeros_like(self.norm_temp)
        s_m = np.ones((len(self.norm_temp), n))
        for j in range(0, n):
            logging.info("Сэмплирование, шаг {}".format(j))
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
        self.draw()
        return square_dict
