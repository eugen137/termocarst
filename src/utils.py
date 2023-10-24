import json
import logging
import uuid
from abc import ABC

import numpy as np


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


def normalize(arr):
    return (arr - np.min(arr)) / \
        (np.max(arr) - np.min(arr))
