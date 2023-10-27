import json
import logging
import uuid
from abc import ABC
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


class TypesOfParameters(Enum):
    TEMPERATURE = 1
    PRECIPITATIONS = 2
    ERRORS = 3


class Processing(ABC):
    def __init__(self, task_id, square, temperature, precipitation):
        self.id = task_id
        self._precipitation = precipitation
        self._temperature = temperature
        self._square = square
        self.type = None

    def import_from_message(self, message):
        message = json.loads(message)
        if "square" in message.keys() and "precipitation" in message.keys() and "temperature" in message.keys():
            self.id = message["id"] if "id" in message.keys() else uuid.uuid4()
            self.input_square(message["square"])
            self.input_precipitation(message["precipitation"])
            self.input_temperature(message["temperature"])
            logging.info("ID={}, импортированы данные square, precipitation, temperature".format(self.id))
            self.type = message["type"] if "type" in message.keys() else "randomize_modeling"
            logging.info("ID={}. Тип восстановления сменен на {}".format(message["type"], self.id))
            return True
        else:
            return False

    def input_precipitation(self, precipitation):
        self._precipitation = precipitation

    def input_temperature(self, temperature):
        self._temperature = temperature

    def input_square(self, square):
        self._square = square

    def _test_data(self):
        if type(self._precipitation) != dict or type(self._temperature) != dict:
            return False
        if self._precipitation.keys() != self._temperature.keys():
            return False
        return True


class Randomize(ABC):

    def __init__(self, task_id, square: np.ndarray, years_square: np.ndarray, temperature: np.ndarray,
                 precipitation: np.ndarray, years: np.ndarray, alpha=np.array([0, 1]),
                 beta=np.array([0, 1]), ksi=np.array([-0.15, 0.15])):
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
        self.p = 0
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
        self.calculated_l_r = None
        self.calculated_ro = None
        self.calculated_h_r = None
        self.calculated_fo = None
        self.calculated_q_err = None
        self.calculated_mean_theta = None

    def func(self, theta):
        pass

    def params_gaps(self):
        pass

    def g_m(self, theta_m):
        return (np.exp(-self.ksi[0] * theta_m) * (self.ksi[0] * theta_m + 1) -
                np.exp(-self.ksi[1] * theta_m) * (self.ksi[1] * theta_m + 1)) / \
            (theta_m * (np.exp(-self.ksi[0] * theta_m) -
                        np.exp(-self.ksi[1] * theta_m)))

    def k(self, theta):
        hr = (self.h_r(theta)).astype(np.float64)
        return (np.exp((-self.beta[0] * hr)) * (self.beta[0] * hr + 1) -
                np.exp((-self.beta[1] * hr)) * (self.beta[1] * hr + 1)) / (hr * (np.exp(-self.beta[0] * hr) -
                                                                                 np.exp(-self.beta[1] * hr)))

    def theta_calc(self):
        logging.info("ID={}. Вычисление множителей Лагранжа".format(self.id))
        sol = optimize.root(self.func, np.ones_like(self.square), method="hybr")
        if sol.success:
            self.theta = sol.x
            logging.info("ID={}. Успешно окончено вычисление множителей Лагранжа".format(self.id))
            return sol.x
        else:
            logging.error("ID={}. Вычисление множителей Лагранжа окончено неудачей".format(self.id))
            return None

    def el(self, theta):
        lr = self.l_r(theta)
        return (np.exp(-self.alpha[0] * lr) * (self.alpha[0] * lr + 1) -
                np.exp(-self.alpha[1] * lr) * (self.alpha[1] * lr + 1)) / (lr * (np.exp(-self.alpha[0] * lr) -
                                                                                 np.exp(-self.alpha[1] * lr)))

    def h_r(self, theta):
        return np.sum(np.multiply(theta, self.precip_norm_yws))

    def l_r(self, theta):
        return np.sum(np.multiply(theta, self.temp_norm_yws))

    def value_from_prv(self, type_of_parameter: TypesOfParameters, num=0):
        n = None
        if self.theta is None:
            return None

        if type_of_parameter == TypesOfParameters.TEMPERATURE:
            edges = self.alpha
            lh_r = self.calculated_l_r
            rf_o = self.calculated_ro

        elif type_of_parameter == TypesOfParameters.PRECIPITATIONS:
            edges = self.beta
            lh_r = self.calculated_h_r
            rf_o = self.calculated_fo

        else:
            # значит type_of_parameter == TypesOfParameters.ERRORS:
            edges = self.ksi
            lh_r = None

        if type_of_parameter != TypesOfParameters.ERRORS:
            def prv(x, p_lh_r):

                return np.exp(-x * p_lh_r) / rf_o
        else:
            def prv(x, p_lh_r):
                q_err_k = (np.exp(-self.ksi[0] * self.calculated_mean_theta) - np.exp(
                    -self.ksi[1] * self.calculated_mean_theta)) / self.calculated_mean_theta
                return np.exp(-x * self.calculated_mean_theta) / q_err_k
        return generator(edges, prv, lh_r)

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

        axs[2].plot(self.years, self.restored_square, 'g', self.years, f_sf, 'g', self.years, f_middle_f, 'r',
                    q + self.years[0], self.restored_square[q], '*r')
        axs[2].set_title("Площадь")
        axs[2].set_xlabel('x_label', size=10)
        plt.savefig('foo.png')


def normalize(arr):
    return (arr - np.min(arr)) / \
        (np.max(arr) - np.min(arr))


def generator(edges, prv, lh_r):
    max_value_prv = max(prv(edges[0], lh_r), prv(edges[1], lh_r))
    while True:
        x1 = np.random.rand()
        x2 = np.random.rand()
        x1_ = edges[0] + x1 * (edges[1] - edges[0])
        if max_value_prv * x2 <= prv(x1_, lh_r):
            break
    return x1


def is_none(a):
    if a is None:
        return True
    if np.isnan(a):
        return True
    return False
