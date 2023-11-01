import json
import logging
import uuid
from abc import ABC

import numpy as np


class Task(ABC):
    def __init__(self, task_id=None, main_param_array=None, secondary_param_matrix=None, count_of_trajectories=None):
        self.main_param_array = main_param_array
        self.secondary_param_matrix = secondary_param_matrix
        self.id = task_id
        self.secondary_param_matrix = np.array([])
        self.secondary_param_names = ["temperature", "precipitation"]
        self.main_param_name = "square"

        if count_of_trajectories is None:
            self.count_of_trajectories = count_of_trajectories
        else:
            self.count_of_trajectories = count_of_trajectories

    def import_from_message(self, message):
        message = json.loads(message)

        # забираем описание задачи
        self.id = message["id"] if "id" in message.keys() else uuid.uuid4()
        self.main_param_name = message["main_param"] if "main_param" in message.keys() else "square"
        self.count_of_trajectories = message["count_of_trajectories"] \
            if "count_of_trajectories" in message.keys() else self.count_of_trajectories

        # собираем годы
        years = []
        param_data = message["data"]
        for param in param_data.keys():
            for y in param_data[param].keys():
                if y not in years:
                    years.append(y)

        # сортируем года
        years.sort()

        # определим второстепенные параметры
        secondary_params = []
        for param in param_data.keys():
            if param != self.main_param_name:
                secondary_params.append(param)

        # соберем данные
        self.secondary_param_matrix = np.zeros((len(years), len(secondary_params)))
        self.main_param_array = np.zeros_like(np.array(years))
        for i in range(0, len(years)):
            for j in range(0, len(secondary_params)):
                self.secondary_param_matrix[i, j] = param_data[secondary_params[j]][years[i]]
                self.main_param_array[i] = param_data[self.main_param_name][years[i]] \
                    if years[i] in param_data[self.main_param_name].keys() else None

    def _test_data(self):
        pass


class ForecastTask(Task):
    def __init__(self, forecast_years, main_param: np.ndarray, secondary_param=None, count_of_trajectories=None):
        super().__init__(main_param, secondary_param, count_of_trajectories)
        self.forecast_years = forecast_years

    def import_from_message(self, message):
        super().import_from_message(message)
        self.forecast_years = 5  # TODO: сделать конфига
