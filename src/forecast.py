import json

import numpy as np

from src.randomize_modeling import RandomizeForecasting
from src.utils import Processing


class Forecasting(Processing):
    def __init__(self, forecast_type="randomize_modeling", precipitation=None, temperature=None, square=None,
                 period_type=None, task_id=None):
        super().__init__(task_id=task_id, precipitation=precipitation, temperature=temperature, square=square)
        self.type = forecast_type
        self.period_type = period_type

    def import_from_message(self, message):
        if super().import_from_message(message):
            message = json.loads(message)
            if "period_type" in message.keys():
                self.period_type = message["period_type"]
                return True
        return False

    def forecast(self):
        square = np.array(list(self._square.values()))
        temperature = np.array(list(self._temperature.values()))
        precipitation = np.array(list(self._precipitation.values()))
        years = np.array(list(self._precipitation.keys())).astype(int)
        years_square = np.array(list(self._square.keys())).astype(int)
        rand_forecasting = RandomizeForecasting(self.id, square=square, years_square=years_square,
                                                temperature=temperature,
                                                precipitation=precipitation, years=years, alpha=np.array([-2.7, 10]),
                                                beta=np.array([-1.9, 10]), ksi=np.array([-0.15, 0.15]))
