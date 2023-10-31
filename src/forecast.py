import json
import numpy as np
from src.new_randomize import RandomizeForecast
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
        years = list(self._temperature.keys())
        years.sort()
        data_ = np.ones((len(years), 3))
        for i in range(0, len(years)):
            data_[i, 0] = self._square[years[i]] if years[i] in self._square.keys() else None
            data_[i, 1] = self._temperature[years[i]]
            data_[i, 2] = self._precipitation[years[i]]
        rand_forecasting = RandomizeForecast(self.id, data_)
        rand_forecasting.learning()
        square = rand_forecasting.modeling(500)
        return square
