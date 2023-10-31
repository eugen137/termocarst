import logging
from uuid import uuid4
from scipy import interpolate

from src.randomize_modeling import RandomizeRecover
from src.utils import Processing, make_data_matrix


class Recovering(Processing):
    def __init__(self, type_recovery="randomize_modeling", precipitation=None, temperature=None, square=None,
                 task_id=None):
        task_id = task_id if task_id else uuid4()
        super().__init__(task_id, precipitation=precipitation, temperature=temperature, square=square)
        self.type = type_recovery
        self.__restored_square = None

    def __recovery_polynom(self):
        logging.info("ID={}, Начато восстановление методом polynom".format(self.id))
        if self._test_data:
            years = list(self._precipitation.keys())
            logging.info("ID={}, Тест пройден".format(self.id))
        else:
            return False
        square_years = list(self._square.keys())
        square_years = [float(y) for y in square_years]
        orig_sq = [float(s) for s in self._square.values()]
        interpol = interpolate.CubicSpline(square_years, orig_sq)
        logging.info("ID={}, Начато интерполирование CubicSpline".format(self.id))
        square = dict()
        for y in years:
            square[y] = interpol(float(y)).tolist()
        return square

    def __randomize_modeling(self):

        data_ = make_data_matrix(self._square, self._temperature, self._precipitation)
        rand_restoring = RandomizeRecover(self.id, data_)
        rand_restoring.learning()
        square = rand_restoring.modeling(1000)
        return square

    def get_recovered_square(self):
        if self.type == "polynomial":
            self._square = self.__recovery_polynom()
        else:
            self._square = self.__randomize_modeling()
        return self._square
