import json
import logging
import uuid
from abc import ABC
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt


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
        self.calculated_square = {}

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

    # def draw(self):
    #     q = []
    #     for i in range(0, len(self.temperature)):
    #         if i not in self.num_years_with_square:
    #             q.append(i)
    #     p_sf = np.polyfit(self.years_square, self.square, 1)
    #     f_sf = np.polyval(p_sf, self.years)
    #
    #     p_middle_f = np.polyfit(self.years, self.restored_square, 1)
    #     f_middle_f = np.polyval(p_middle_f, self.years)
    #
    #     fig, axs = plt.subplots(3)
    #     plt.subplots_adjust(hspace=0.6)
    #
    #     axs[0].plot(self.years, self.temperature)
    #     axs[0].set_title("Температура")
    #
    #     axs[1].plot(self.years, self.precipitation)
    #     axs[1].set_title("Осадки")
    #
    #     axs[2].plot(self.years, self.restored_square, 'g', self.years, f_sf, 'g', self.years, f_middle_f, 'r',
    #                 q + self.years[0], self.restored_square[q], '*r')
    #     axs[2].set_title("Площадь")
    #     axs[2].set_xlabel('x_label', size=10)
    #     plt.savefig('foo.png')


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
    return x1_


def generator_param(edges, prv):

    max_value_prv = max(prv(edges[0]), prv(edges[1]))

    while True:
        x1 = edges[0] + np.random.uniform(0, 1) * (edges[1] - edges[0])
        x2 = np.random.uniform(0, 1)
        if x2 <= prv(x1) / max_value_prv:
            break
    return x1


def is_none(a):
    if a is None:
        return True
    if np.isnan(a):
        return True
    return False


def make_data_matrix(square: dict, temperature: dict, precipitation: dict):
    years = list(temperature.keys())
    years.sort()
    data_ = np.ones((len(years), 3))
    for i in range(0, len(years)):
        data_[i, 0] = square[years[i]] if years[i] in square.keys() else None
        data_[i, 1] = temperature[years[i]]
        data_[i, 2] = precipitation[years[i]]
    return data_
