import logging
from abc import ABC
import numpy as np
from scipy.optimize import optimize

from src.utils import is_none, generator


class RandomizeNew(ABC):

    def __init__(self, task_id, data: np.ndarray):
        """
        Класс рандомизированного восстановления пропусков в данных о площади озер.
        :param task_id: ID задачи
        :param data: Матрица с данными, первый столбец - целевой аргумент(например, площадь), если есть пропуски в
            данных - они равны None, второй (в случае прогнозирования/восстановления площади) - температура,
            третий (в случае прогнозирования/восстановления площади) - осадки
        """

        self.normalized_data = None
        self.memory_param = 0
        self.id = task_id
        logging.info("ID={}. Начата работа Рандомизированного алгоритма".format(self.id), extra={"task_id": self.id})
        self.data = data

        # подготовка данных
        self.data_gaps = []
        self.operating_data = np.array([])
        self.data_analysis()  # анализ данных
        self.memory_param_calc()   # вычисление порядка
        self.operation_data_preparation()  # подготовка данных с учетом порядка памяти
        self.min_max = dict()
        self.normalization_data()  # нормализация данных
        self.theta = None
        self.need_for_restoration = False
        self.param_limits = []
        self.error_limits = []
        self.param_limits_calc()

    def data_analysis(self):
        """
        Метод анализа данных, проверка размерности, наличия данных, поиск пропусков.\n
        После работы метода:
         - данные без пропусков сохраняются в self.operating_data
         - признак потребности восстановления сохраняется в self.need_for_restoration
        :return: True|False - пройдена или не пройдена проверка
        """
        logging.info("ID={}. Анализ входных данных.".format(self.id), extra={"task_id": self.id})
        # logging.debug("ID={}. проверка размерности входящих данных.".format(self.id), extra={"task_id": self.id})
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
            if is_none(self.data[i, 0]):
                self.data_gaps.append(i)
                self.need_for_restoration = True
            else:
                try:
                    self.operating_data = np.vstack((self.operating_data, self.data[i, :]))
                except ValueError:
                    self.operating_data = self.data[i, :]
        logging.debug("ID={}. Проверка данных окончена".format(self.id), extra={"task_id": self.id})
        return True

    def memory_param_calc(self):
        """
        Метод вычисления параметра "памяти".
        :return: Параметр памяти self.memory_param
        """

        logging.info("ID={}. Начато вычисление порядка.".format(self.id))
        arg = self.operating_data[:, 0]   # целевой аргумент - первый столбец
        s_mean = np.mean(arg)
        len_s = len(arg)
        k = 0
        c_0 = np.sum(np.power(arg - s_mean, 2)) / len_s
        for i in range(0, len_s):
            c_k = 0
            for j in range(0, len_s - i):
                c_k += (arg[j] - s_mean) * (arg[j + i] - s_mean)
            c_k = c_k / len_s
            if abs(c_k / c_0) < 0.1:
                k = i
                break
        self.memory_param = k
        logging.info("ID={}. Порядок вычислен и равен {}".format(self.id, self.memory_param))

    def operation_data_preparation(self):
        """Подготовка данных с учетом параметра памяти"""
        data = self.operating_data.copy()
        arr = data[:, 0]
        n = data.shape
        # составляем матрицу целевого
        self.operating_data = np.array([])
        for i in range(0, self.memory_param):
            if self.operating_data.shape[0] == 0:
                self.operating_data = arr[i:n[0]+i-self.memory_param].reshape(-1, 1)
            else:
                self.operating_data = np.hstack((self.operating_data, arr[i:n[0]+i-self.memory_param].reshape(-1, 1)))
        for i in range(1, n[1]):
            self.operating_data = np.hstack((self.operating_data, data[0:n[0]-self.memory_param, i].reshape(-1, 1)))

    def normalization_data(self):
        """
        Нормализация данных мин-макс - 0..1
        :return: self.normalized_data нормализованные данные
        """
        # TODO: нужно добавить формирование матрицы параметров
        logging.info("ID={}. Нормировка данных".format(self.id), extra={"task_id": self.id})
        min_data = np.min(self.operating_data, axis=0)
        max_data = np.max(self.operating_data, axis=0)
        self.operating_data = (self.operating_data - min_data) / (max_data - min_data)
        self.min_max["min"] = min_data
        self.min_max["max"] = max_data

    def param_func(self, theta, p):
        """
        Множители параметров целевой функции

        :param theta: Множители Лагранжа
        :param p: номер параметра
        :return: множители параметра
        """
        hr = np.sum(np.multiply(theta, self.normalized_data[p, :]))
        return (np.exp((-self.param_limits[p][0] * hr)) * (self.param_limits[p][0] * hr + 1) -
                np.exp((-self.param_limits[p][1] * hr)) * (self.param_limits[p][1] * hr + 1)) / \
            (hr * (np.exp(-self.param_limits[p][0] * hr) - np.exp(-self.param_limits[p][1] * hr)))

    def param_vector(self, theta):
        par_mx = np.ones_like(self.normalized_data[:,0])
        for i in range(0, len(par_mx)):
            par_mx[i] = self.param_func(theta, i)
        return par_mx

    def param_error(self, theta_m):
        """
        Вычисление параметра измерительной ошибки
        :param theta_m: множитель Лагранжа на шаге m
        :return: Параметр ошибки для вычисления целевой функции для оптимизации
        """
        return (np.exp(-self.error_limits[0] * theta_m) * (self.error_limits[0] * theta_m + 1) -
                np.exp(-self.error_limits[1] * theta_m) * (self.error_limits[1] * theta_m + 1)) / \
            (theta_m * (np.exp(-self.error_limits[0] * theta_m) -
                        np.exp(-self.error_limits[1] * theta_m)))

    def func(self, theta):
        """
        Целевая функция для оптимизации
        :param theta: множители Лагранжа
        :return: значение целевой функции
        """
        f = np.ones_like(theta)
        param_vector = self.param_vector(theta)
        for i in range(0, len(theta)):
            f[i] = np.abs(np.sum(param_vector*self.normalized_data[i, :]) + self.param_error(theta[i]))
        return f

    def theta_calc(self):
        """
        Метод вычисления множителей Лагранжа через оптимизацию целевой функции
        :return: множителей Лагранжа
        """
        logging.info("ID={}. Вычисление множителей Лагранжа".format(self.id))
        sol = optimize.root(self.func, np.ones_like(self.normalized_data[0, :]), method="hybr")
        if sol.success:
            self.theta = sol.x
            logging.info("ID={}. Успешно окончено вычисление множителей Лагранжа".format(self.id))
            return sol.x
        else:
            logging.error("ID={}. Вычисление множителей Лагранжа окончено неудачей".format(self.id))
            return None

    def param_limits_calc(self):
        logging.info("ID={}. Начато вычисление промежутков параметров".format(self.id))
        # вычисляются оценки МНК
        y = self.operating_data[:, 0].reshape(-1, 1)

        x = np.hstack((np.ones_like(y), self.operating_data[:, 1:]))
        m = self.memory_param + 3  # 2 - температура и осадки, 1 - свободный

        a = np.linalg.inv(np.mat(np.transpose(x)) * np.mat(np.mat(x))) * np.mat(np.transpose(x)) * (np.mat(y))
        e = np.mat(x) * np.mat(a)
        s2 = np.sum(np.power(e - y, 2)) / (len(y) - m)
        q = np.mat(np.transpose(x)) * np.mat(np.mat(x))

        for i in range(0, a.shape[0]):
            self.param_limits.append([a[i, 0] - 3 * np.sqrt(s2 * q[i, i]), a[i, 0] + 3 * np.sqrt(s2 * q[i, i])])
        self.error_limits = [- 3 * np.sqrt(s2), 3 * np.sqrt(s2)]
        print(self.param_limits, self.error_limits)

    # def value_from_prv(self, num=0):
    #     n = None
    #     if self.theta is None:
    #         return None
    #
    #     if type_of_parameter == TypesOfParameters.TEMPERATURE:
    #         edges = self.alpha
    #         lh_r = self.__l_r
    #         rf_o = self.__ro
    #
    #     elif type_of_parameter == TypesOfParameters.PRECIPITATIONS:
    #         edges = self.beta
    #         lh_r = self.__h_r
    #         rf_o = self.__fo
    #
    #     else:
    #         # значит type_of_parameter == TypesOfParameters.ERRORS:
    #         edges = self.ksi
    #
    #     if type_of_parameter != TypesOfParameters.ERRORS:
    #         def prv(x):
    #             return np.exp(-x * lh_r) / rf_o
    #     else:
    #         def prv(x):
    #             q_err_k = (np.exp(-self.ksi[0] * self.__mean_theta) - np.exp(
    #                 -self.ksi[1] * self.__mean_theta)) / self.__mean_theta
    #             return np.exp(-x * self.__mean_theta) / q_err_k
    #     return generator(edges, prv)

