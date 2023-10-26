import logging
from abc import ABC
import numpy as np

from src.utils import is_none


class RandomizeNew(ABC):

    def __init__(self, task_id, data: np.ndarray):
        """
        Класс рандомизированного восстановления пропусков в данных о площади озер.
        :param task_id: ID задачи
        :param data: Матрица с данными, первый столбец - площадь, если есть пропуски в данных - они равны None
        """

        self.normalized_data = None
        self.p = 0
        self.id = task_id
        logging.info("ID={}. Начата работа Рандомизированного алгоритма".format(self.id), extra={"task_id": self.id})
        self.data = data
        # подготовка данных

        self.operating_data = np.array([])
        self.normalization_data()
        self.data_gaps = []
        self.data_analysis()
        self.theta = None
        self.need_for_restoration = False

    def data_analysis(self):
        logging.info("ID={}. Анализ входных данных.".format(self.id), extra={"task_id": self.id})
        logging.debug("ID={}. проверка размерности входящих данных.".format(self.id), extra={"task_id": self.id})
        if len(self.data.shape) != 2:
            logging.error("ID={}. Проверка размерности не прошла.".format(self.id),
                          extra={"task_id": self.id})
            return False

        logging.debug("ID={}. Проверка всех данных, кроме первого столбца.".format(self.id), extra={"task_id": self.id})
        for i in range(1, self.data.shape[0]):
            for j in range(0, self.data.shape[1]):
                if is_none(self.data[i, j]):
                    logging.error("ID={}. Проверка всех данных не прошла.".format(self.id),
                                  extra={"task_id": self.id})
                    return False

        logging.debug("ID={}. Поиск пропусков в первом столбце".format(self.id), extra={"task_id": self.id})
        for i in range(0, self.data.shape[0]):
            if is_none(self.data[0, i]):
                self.data_gaps.append(i)
                self.need_for_restoration = True
            else:
                try:
                    self.operating_data = np.hstack((self.operating_data, self.data[:, i]))
                except ValueError:
                    self.operating_data = self.data[:, i]

    def normalization_data(self):
        logging.info("ID={}. Нормировка данных".format(self.id), extra={"task_id": self.id})
        min_data = np.min(self.operating_data, axis=0)
        max_data = np.max(self.operating_data, axis=0)
        self.normalized_data = (self.operating_data - min_data) / (max_data - min_data)

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
            lh_r = self.__l_r
            rf_o = self.__ro

        elif type_of_parameter == TypesOfParameters.PRECIPITATIONS:
            edges = self.beta
            lh_r = self.__h_r
            rf_o = self.__fo

        else:
            # значит type_of_parameter == TypesOfParameters.ERRORS:
            edges = self.ksi

        if type_of_parameter != TypesOfParameters.ERRORS:
            def prv(x):
                return np.exp(-x * lh_r) / rf_o
        else:
            def prv(x):
                q_err_k = (np.exp(-self.ksi[0] * self.__mean_theta) - np.exp(
                    -self.ksi[1] * self.__mean_theta)) / self.__mean_theta
                return np.exp(-x * self.__mean_theta) / q_err_k
        return generator(edges, prv)

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