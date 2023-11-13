import json

from src.config import config
from src.randomize_modeling import RandomizeForecast
from src.utils import Processing, make_data_matrix


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
        data_ = make_data_matrix(self._square, self._temperature, self._precipitation)
        rand_forecasting = RandomizeForecast(self.id, data_)
        rand_forecasting.learning()
        n = int(config["RANDOMIZE_CONFIG"]["randomize.number_of_random_trajectories_forecasting"])
        square = rand_forecasting.modeling(n, self.period_type)

        # найдем максимальный год
        years = self._temperature.keys()
        years = [int(y) for y in years]
        years.sort()
        forecast_year = rand_forecasting.time_parameter_values[self.period_type] + max(years)
        self.calculated_square = self._square.copy()
        for year in range(max(years) + 1, forecast_year + 1):
            n = len(years) + year - max(years) - 1
            self.calculated_square[str(year)] = square[n]
        return square
