import logging
import numpy as np
from scipy import interpolate
from src.Zone import RandomizeRestoring
from src.utils import Processing


class Recovering(Processing):
    def __init__(self, type_recovery="randomize_modeling", precipitation=None, temperature=None, square=None):
        super().__init__(precipitation=precipitation, temperature=temperature, square=square)
        self.type = type_recovery
        self.__restored_square = None

    def __recovery_polynom(self):
        logging.info("Начато восстановление методом polynom")
        if self._test_data:
            years = list(self._precipitation.keys())
            logging.info("Тест пройден")
        else:
            return False
        square_years = list(self._square.keys())
        square_years = [float(y) for y in square_years]
        orig_sq = [float(s) for s in self._square.values()]
        interpol = interpolate.CubicSpline(square_years, orig_sq)
        logging.info("Начато интерполирование CubicSpline")
        square = dict()
        for y in years:
            square[y] = interpol(float(y)).tolist()
        return square

    def __randomize_modeling(self):
        print(self._square)
        square = np.array(list(self._square.values()))
        temperature = np.array(list(self._temperature.values()))
        precipitation = np.array(list(self._precipitation.values()))
        years = np.array(list(self._precipitation.keys())).astype(int)
        years_square = np.array(list(self._square.keys())).astype(int)
        rand_restoring = RandomizeRestoring(square=square, years_square=years_square, temperature=temperature,
                                            precipitation=precipitation, years=years, alpha=np.array([-2.7, 10]),
                                            beta=np.array([-1.9, 10]), ksi=np.array([-0.05, 0.05]))
        rand_restoring.theta_calc()
        square = rand_restoring.modeling(200)
        return square
        # return self.__square

    def get_recovered_square(self):
        if self.type == "polynomial":
            self._square = self.__recovery_polynom()
        else:
            self._square = self.__randomize_modeling()
        return self._square
