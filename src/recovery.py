import json
import logging
from scipy import interpolate


class Recovery:
    def __init__(self, type_recovery="randomize_modeling", precipitation=None, temperature=None, square=None):
        self.type = type_recovery
        self.__precipitation = precipitation
        self.__temperature = temperature
        self.__square = square

    def import_from_message(self, message):
        message = json.loads(message)
        if "square" in message.keys() and "precipitation" in message.keys() and "temperature" in message.keys():
            self.input_square(message["square"])
            self.input_precipitation(message["precipitation"])
            self.input_temperature(message["temperature"])
            logging.info("Импортированны данные square, precipitation, temperature")
            if "type" in message.keys():
                if message["type"] in ("randomize_modeling", "polynomial"):
                    self.type = message["type"]
                    logging.info("Тип восстановления сменен на {}".format(message["type"]))
                else:
                    logging.info("Тип {} не распознан".format(message["type"]))
            return True
        else:
            return False

    def input_precipitation(self, precipitation):
        self.__precipitation = precipitation

    def input_temperature(self, temperature):
        self.__temperature = temperature

    def input_square(self, square):
        self.__square = square

    def __test_data(self):
        if type(self.__precipitation) != dict or type(self.__temperature) != dict:
            return False
        if self.__precipitation.keys() != self.__temperature.keys():
            return False
        return True

    def __recovery_polynom(self):
        logging.info("Начато восстановление методом polynom")
        if self.__test_data:
            years = list(self.__precipitation.keys())
            logging.info("Тест пройден")
        else:
            return False
        square_years = list(self.__square.keys())
        square_years = [float(y) for y in square_years]
        orig_sq = [float(s) for s in self.__square.values()]
        interpol = interpolate.CubicSpline(square_years, orig_sq)
        logging.info("Начато интерполирование CubicSpline")
        square = dict()
        for y in years:
            square[y] = interpol(float(y)).tolist()
        return square

    def __randomize_modeling(self):
        return self.__square

    def get_recovered_square(self):
        if self.type == "polynomial":
            self.__square = self.__recovery_polynom()
        else:
            self.__square = self.__randomize_modeling()
        return self.__square
