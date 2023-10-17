import json
import logging
from abc import ABC

import numpy as np


class Processing(ABC):
    def __init__(self, square, temperature, precipitation):
        self._precipitation = precipitation
        self._temperature = temperature
        self._square = square
        self.type = None

    def import_from_message(self, message):
        message = json.loads(message)
        if "square" in message.keys() and "precipitation" in message.keys() and "temperature" in message.keys():
            self.input_square(message["square"])
            self.input_precipitation(message["precipitation"])
            self.input_temperature(message["temperature"])
            logging.info("Импортированы данные square, precipitation, temperature")
            self.type = message["type"]
            logging.info("Тип восстановления сменен на {}".format(message["type"]))
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


def normalize(arr):
    return (arr - np.min(arr)) / \
        (np.max(arr) - np.min(arr))
