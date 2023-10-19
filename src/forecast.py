import json

from src.utils import Processing


class Forecasting(Processing):
    def __init__(self, forecast_type="randomize_modeling", precipitation=None, temperature=None, square=None,
                 period_type=None):
        super().__init__(precipitation=precipitation, temperature=temperature, square=square)
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
        pass


