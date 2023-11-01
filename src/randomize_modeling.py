import logging
from abc import ABC
import numpy as np
from scipy import optimize

from config import config
from src.utils import is_none, generator_param


class RandomizeParent(ABC):

    def __init__(self, task_id, data: np.ndarray, param_names=None):
        """
        Класс рандомизированного прогнозирования.
        :param task_id: ID задачи
        :param data: Матрица с данными, первый столбец - целевой аргумент(например, площадь), если есть пропуски в
            данных - они равны None, второй (в случае прогнозирования/восстановления площади) - температура,
            третий (в случае прогнозирования/восстановления площади) - осадки
        :param param_names - названия параметров (например, ["Площадь, Осадки, Температура"])
        """
        if param_names is None:
            param_names = ["Площадь", "Температура", "Осадки"]
        else:
            if type(param_names) != list:
                param_names = [param_names]
        self.param_names = param_names
        self.memory_param = 0
        self.id = task_id
        logging.info("ID={}. Целевой параметр: {}. "
                     "Начата работа Рандомизированного алгоритма".format(self.id, self.param_names[0]),
                     extra={"task_id": self.id, })
        self.data = data
        self.theta = None
        self.need_for_restoration = False
        self.param_limits = []
        self.error_limits = []

        # подготовка данных
        self.min_max = dict()
        self.data_gaps = []
        self.data_for_restore = np.array([])
        self.operating_data = np.array([])
        self.data_analysis()  # анализ данных

        self.target_array = self.operating_data[:, 0]  # целевой параметр
        self.memory_param_calc()  # вычисление порядка
        self.operation_data_preparation()  # подготовка данных с учетом порядка памяти
        self.param_limits_calc()

    def learning(self):
        # обучение модели (вычисление theta)
        self.theta_calc()

    def data_analysis(self):
        """
        Метод анализа данных, проверка размерности, наличия данных, поиск пропусков.\n
        После работы метода:
         - данные без пропусков сохраняются в self.operating_data
         - признак потребности восстановления сохраняется в self.need_for_restoration
        :return: True|False - пройдена или не пройдена проверка
        """
        logging.info("ID={}. Целевой параметр: {}. Анализ входных данных.".format(self.id, self.param_names[0],
                                                                                  self.param_names[0]),
                     extra={"task_id": self.id, })
        if len(self.data.shape) != 2:
            logging.error(
                "ID={}. Целевой параметр: {}. Проверка размерности не прошла.".format(self.id, self.param_names[0]),
                extra={"task_id": self.id, })
            return False

        logging.debug("ID={}. Целевой параметр: {}. "
                      "Проверка всех данных, кроме первого столбца.".format(self.id, self.param_names[0]),
                      extra={"task_id": self.id, })
        for i in range(0, self.data.shape[0]):
            for j in range(1, self.data.shape[1]):
                if is_none(self.data[i, j]):
                    logging.error("ID={}. Целевой параметр: {}. "
                                  "Проверка всех данных не прошла.".format(self.id, self.param_names[0]),
                                  extra={"task_id": self.id, })
                    return False

        logging.debug(
            "ID={}. Целевой параметр: {}. Поиск пропусков в первом столбце".format(self.id, self.param_names[0]),
            extra={"task_id": self.id, })
        for i in range(0, self.data.shape[0]):
            if is_none(self.data[i, 0]):
                if self.__class__.__name__ == "RandomizeForecast":
                    recover = RandomizeRecover(self.id, self.data, self.param_names)
                    recover.learning()
                    n = int(config["RANDOMIZE_CONFIG"]["randomize.number_of_random_trajectories_restoring"])
                    recovered_data = recover.modeling(n)
                    self.data[:, 0] = recovered_data
                    self.need_for_restoration = False
                    self.data_gaps = []
                    self.data_for_restore = np.array([])
                    self.operating_data = np.array([])
                    self.data_analysis()
                # Если есть пропуски в целевом параметре - номер добавляем в self.data_gaps номер
                # и в self.data_for_restore добавляем данные
                self.data_gaps.append(i)
                self.need_for_restoration = True

                try:
                    # при добавлении пропускаем self.data[i, 0] так как он является пропуском в данных
                    self.data_for_restore = np.vstack((self.data_for_restore, self.data[i, 1:]))
                except ValueError:
                    self.data_for_restore = self.data[i, 1:]
            else:
                try:
                    self.operating_data = np.vstack((self.operating_data, self.data[i, :]))
                except ValueError:
                    self.operating_data = self.data[i, :]
        logging.debug("ID={}. Целевой параметр: {}. Проверка данных окончена".format(self.id, self.param_names[0]),
                      extra={"task_id": self.id, })
        return True

    def memory_param_calc(self):
        """
        Метод вычисления параметра "памяти". В случае восстановления данных - self.memory_param = 0
        :return: Параметр памяти self.memory_param
        """
        if self.__class__.__name__ != "RandomizeForecast":
            return
        logging.info("ID={}. Целевой параметр: {}. Вычисление порядка.".format(self.id, self.param_names[0]))
        arg = self.target_array
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
        logging.info("ID={}. Целевой параметр: {}. Порядок вычислен и равен {}".format(self.id, self.param_names[0],
                                                                                       self.memory_param))

    def operation_data_preparation(self):
        """Подготовка данных с учетом параметра памяти"""
        data = self.operating_data.copy()
        arr = self.target_array
        n = self.operating_data.shape
        # составляем матрицу целевого параметра
        self.operating_data = np.array([])

        for i in range(0, self.memory_param):  # пропускаем текущее значение
            if self.operating_data.shape[0] == 0:
                self.operating_data = arr[i:n[0] + i - self.memory_param].reshape(-1, 1)
            else:
                self.operating_data = np.hstack(
                    (self.operating_data, arr[i:n[0] + i - self.memory_param].reshape(-1, 1)))

        # добавляем дополнительные параметры
        for i in range(1, n[1]):
            try:
                # при добавлении пропускаем self.data[i, 0] так как он является пропуском в данных
                self.operating_data = np.hstack((self.operating_data, data[self.memory_param:, i].reshape(-1, 1)))
            except ValueError:
                self.operating_data = data[self.memory_param:, i].reshape(-1, 1)

        # добавляем свободное слагаемое в случае если это прогнозирование
        if self.__class__.__name__ == "RandomizeForecast":
            self.operating_data = np.hstack((np.ones_like(data[self.memory_param:, 0].reshape(-1, 1)),
                                             self.operating_data))
        self.target_array = self.target_array[self.memory_param:]

        self.normalization_data()  # нормализация данных

    def normalization_data(self):
        """
        Нормализация данных мин-макс - 0..1
        :return: self.operating_data - нормализованные данные,
            self.min_max - dict с минимальным и максимальным значением
        """
        logging.info("ID={}. Целевой параметр: {}. Нормировка данных".format(self.id, self.param_names[0]),
                     extra={"task_id": self.id, })
        min_data = np.min(self.operating_data, axis=0)
        max_data = np.max(self.operating_data, axis=0)
        self.operating_data = (self.operating_data - min_data) / (max_data - min_data)
        self.min_max["min"] = min_data
        self.min_max["max"] = max_data
        self.min_max["min_target"] = self.target_array.min()
        self.min_max["max_target"] = self.target_array.max()
        min_data = self.target_array.min()
        max_data = self.target_array.max()
        self.target_array = (self.target_array - min_data) / \
                            (max_data - min_data)
        self.operating_data = np.nan_to_num(self.operating_data, nan=1)

        if self.__class__.__name__ == "RandomizeRecover":
            self.data_for_restore = (self.data_for_restore - self.min_max["min"]) / \
                                    (self.min_max["max"] - self.min_max["min"])

    def param_func(self, theta, p):
        """
        Множители параметров целевой функции
        :param theta: Множители Лагранжа
        :param p: номер параметра
        :return: множитель параметра
        """
        hr = np.sum(np.multiply(theta, self.operating_data[:, p]))
        logging.debug(
            "ID={}. Целевой параметр: {}. self.operating_data[:, p] = {}".format(self.id, self.param_names[0],
                                                                                 self.operating_data[:, p]),
            extra={"task_id": self.id, })
        logging.debug("ID={}. Целевой параметр: {}. theta = {}".format(self.id, self.param_names[0], theta),
                      extra={"task_id": self.id, })
        logging.debug("ID={}. Целевой параметр: {}. hr = {}".format(self.id, self.param_names[0], hr),
                      extra={"task_id": self.id, })
        return (np.exp((-self.param_limits[p][0] * hr)) * (self.param_limits[p][0] * hr + 1) -
                np.exp((-self.param_limits[p][1] * hr)) * (self.param_limits[p][1] * hr + 1)) / \
            (hr * (np.exp(-self.param_limits[p][0] * hr) - np.exp(-self.param_limits[p][1] * hr)))

    def param_vector(self, theta):
        """Вычисление вектора множителей параметров целевой функции"""
        par_mx = np.ones_like(self.operating_data[0, :])
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
            f[i] = np.abs((param_vector * self.operating_data[i, :]).sum() + self.param_error(theta[i]))
        return f

    def theta_calc(self):
        """
        Метод вычисления множителей Лагранжа через оптимизацию целевой функции
        :return: множителей Лагранжа
        """
        logging.info("ID={}. Целевой параметр: {}. Вычисление множителей Лагранжа".format(self.id, self.param_names[0]))
        sol = optimize.root(self.func, np.ones_like(self.operating_data[:, 0]), method="lm")
        if sol.success:
            self.theta = sol.x
            logging.info("ID={}. Целевой параметр: {}. "
                         "Успешно окончено вычисление множителей Лагранжа".format(self.id, self.param_names[0]))
            logging.debug(
                "ID={}. Целевой параметр: {}. Множители Лагранжа = {}".format(self.id, self.param_names[0], self.theta))
            return sol.x
        else:
            logging.error(
                "ID={}. Целевой параметр: {}. "
                "Вычисление множителей Лагранжа окончено неудачей".format(self.id, self.param_names[0]))
            return None

    def param_limits_calc(self):
        """
        Вычисление пределов множителей параметров
        :return:
        """
        logging.info("ID={}. Целевой параметр: {}. "
                     "Вычисление промежутков параметров".format(self.id, self.param_names[0]))
        # вычисляются оценки МНК
        y = self.target_array  # целевое значение
        y = y.reshape(-1, 1)  # превращаем строку в столбец

        x = self.operating_data
        m = self.operating_data.shape[1]

        a_ls = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        e_ls = y - x.dot(a_ls)
        s2 = np.power(e_ls, 2).sum() / (len(y) - m)
        q = np.linalg.inv(x.T.dot(x))

        for i in range(0, a_ls.shape[0]):
            self.param_limits.append([a_ls[i, 0] - 3 * np.sqrt(s2 * q[i, i]), a_ls[i, 0] + 3 * np.sqrt(s2 * q[i, i])])
        self.error_limits = [- 3 * np.sqrt(s2), 3 * np.sqrt(s2)]

    def generate_random_value_with_prv(self, hr_vector: np.ndarray, ro_vector: np.ndarray):
        """
        Генерирует случайное (множитель параметра) число с заданной ПРВ на рассчитанном интервале.

        :param hr_vector: Поэлементное произведение theta и векторов параметров (с учетом памяти)
        :param ro_vector: Делитель для каждого параметра
        :return: случайное число с заданной ПРВ на рассчитанном интервале
        """
        if self.theta is None:
            return None
        ans = np.zeros_like(hr_vector)
        for p in range(0, len(hr_vector)):
            def prv(x):
                return np.exp(-x * hr_vector[p]) / ro_vector[p]
            ans[p] = generator_param(self.param_limits[p], prv)
        return ans

    def generate_random_error_prv(self):
        if self.theta is None:
            return None

        def prv(x):
            q_err_k = (np.exp(-self.error_limits[0] * self.theta.mean()) -
                       np.exp(-self.error_limits[1] * self.theta.mean())) / self.theta.mean()
            return np.exp(-x * self.theta.mean()) / q_err_k
        return generator_param(self.error_limits, prv)


class RandomizeForecast(RandomizeParent):
    def __init__(self, task_id, data: np.ndarray, param_names=None, time_parameter_values=None):

        super().__init__(task_id, data, param_names)
        if time_parameter_values is None:
            time_parameter_values = {"short": int(config["RANDOMIZE_CONFIG"]["randomize.short_time_period"]),
                                     "middle": int(config["RANDOMIZE_CONFIG"]["randomize.middle_time_period"]),
                                     "long": int(config["RANDOMIZE_CONFIG"]["randomize.long_time_period"])}
        self.time_parameter_values = time_parameter_values

    def modeling(self, n=100, period_type="short", secondary=False):
        logging.info("ID={}. Целевой параметр: {}. "
                     "Начато моделирование - количество итераций - {}".format(self.id, self.param_names[0], n))
        if period_type in self.time_parameter_values.keys():
            forecast_years = self.time_parameter_values[period_type]
        else:
            forecast_years = 0
        forecast_matrix = np.zeros((forecast_years, self.memory_param))
        forecast_matrix = np.hstack((np.ones((forecast_years, 1)), forecast_matrix))

        # прогнозирование вспомогательных параметров
        logging.info("ID={}. Целевой параметр: {}. "
                     "Прогнозирование вспомогательных параметров".format(self.id, self.param_names[0]))
        for p in range(1, self.data.shape[1]):
            logging.info("ID={}. Целевой параметр: {}. "
                         "Прогнозирование вспомогательного параметра {}".format(self.id, self.param_names[0],
                                                                                self.param_names[p]))
            param_model = RandomizeForecast(self.id, self.data[:, p].reshape(-1, 1), [self.param_names[p]])
            param_model.learning()
            param_values = param_model.modeling(n=n, secondary=True)
            forecast_matrix = np.hstack((forecast_matrix, param_values[-forecast_years:].reshape(-1, 1)))
        logging.info("ID={}. Целевой параметр: {}. "
                     "Прогнозирование вспомогательных параметров окончено".format(self.id, self.param_names[0]))
        # forecast_matrix - матрица, у которой слева - нули, справа - предсказанные дополнительные параметры (в
        # случае температуры и осадков - только нулевая матрица)

        # вычисляем параметры функции ПРВ
        hr_vector = ro_vector = np.ones(self.operating_data.shape[1])
        for p in range(1, self.operating_data.shape[1]):
            hr_vector[p] = np.sum(np.multiply(self.theta, self.operating_data[:, p]))
            ro_vector[p] = (np.exp(-self.param_limits[p][0] * hr_vector[p]) -
                            np.exp(-self.param_limits[p][1] * hr_vector[p])) / hr_vector[p]

        # объединяем матрицы forecast_matrix и operating_data
        forecast_matrix = np.vstack((self.operating_data, forecast_matrix))
        work_matrix = np.zeros((forecast_matrix.shape[0], forecast_matrix.shape[1], n))
        logging.info("ID={}. Целевой параметр: {}. "
                     "Начато моделирование".format(self.id, self.param_names[0]))
        # заполняем work_matrix известными значениями
        for i in range(0, n):
            work_matrix[:, :, i] = forecast_matrix
        forecasted_target_param = np.zeros((forecast_years + len(self.target_array), n))
        for step in range(0, n):
            logging.debug("ID={}. Целевой параметр: {}. "
                          "Моделирование - шаг {}".format(self.id, self.param_names[0], step))
            # заполним forecasted_target_param известными значениями
            forecasted_target_param[:len(self.target_array), step] = self.target_array
            for year in range(self.operating_data.shape[0], forecast_matrix.shape[0]):
                # формируем строку с данными памяти
                for j in range(1, self.memory_param + 1):
                    work_matrix[year, j, step] = forecasted_target_param[year - j, step]

                # предсказываем на шаге step значение целевого параметра
                forecasted_target_param[year, step] = \
                    np.sum((self.generate_random_value_with_prv(hr_vector, ro_vector)*work_matrix[year, :, step])) + \
                    self.generate_random_error_prv()

        ans = forecasted_target_param.mean(1)
        ans = ans[-forecast_years:]
        if not secondary:
            ans = ans * (self.min_max["max_target"] - self.min_max["min_target"]) + self.min_max["min_target"]
            ans = np.hstack((self.data[:, 0], ans))
        else:

            addition = (self.data[:, 0] - self.min_max["min_target"]) / \
                       (self.min_max["max_target"] - self.min_max["min_target"])
            ans = np.hstack((addition, ans))
        return ans


class RandomizeRecover(RandomizeParent):
    def modeling(self, n=100):
        logging.info("ID={}. Целевой параметр: {}. "
                     "Начато моделирование рандомизированного восстановления пропусков. "
                     "Количество итераций - {}".format(self.id, self.param_names[0], n))

        # вычисляем параметры функции ПРВ
        hr_vector = ro_vector = np.ones(self.operating_data.shape[1])
        for p in range(1, self.operating_data.shape[1]):
            hr_vector[p] = np.sum(np.multiply(self.theta, self.operating_data[:, p]))
            ro_vector[p] = (np.exp(-self.param_limits[p][0] * hr_vector[p]) -
                            np.exp(-self.param_limits[p][1] * hr_vector[p])) / hr_vector[p]
        recovered_target_param = np.zeros((len(self.data_gaps), n))
        for step in range(0, n):
            logging.debug("ID={}. Целевой параметр: {}. "
                          "Моделирование - шаг {}".format(self.id, self.param_names[0], step))

            for year in range(0, self.data_for_restore.shape[0]):
                # предсказываем на шаге step значение целевого параметра
                recovered_target_param[year, step] = \
                    np.sum((self.generate_random_value_with_prv(hr_vector, ro_vector)*self.data_for_restore[year, :])) \
                    + self.generate_random_error_prv()

        ans = recovered_target_param.mean(1)
        ans = ans * (self.min_max["max_target"] - self.min_max["min_target"]) + self.min_max["min_target"]
        ans2 = []
        k = 0
        for i in range(0, self.data.shape[0]):
            if i in self.data_gaps:
                ans2.append(ans[k])
                k += 1
            else:
                ans2.append(self.data[i, 0])
        logging.info("ID={}. Целевой параметр: {}. Восстановление окончено ".format(self.id, self.param_names[0]))
        return np.array(ans2)
