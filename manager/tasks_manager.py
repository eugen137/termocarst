
import logging
import uuid
import numpy as np

from manager.configuration import static_config
from manager.tasks import RecoveryTask, ForecastTask
from manager.utils import State


class TaskManager:
    """
    Элемент менеджера задач
    """
    def __init__(self, task_id=None, main_param_array=None, secondary_param_matrix=None, count_of_trajectories=None):
        self.main_param_array = main_param_array
        self.secondary_param_matrix = secondary_param_matrix
        self.id = task_id
        self.secondary_param_matrix = np.array([])
        self.secondary_param_names = ["temperature", "precipitation"]
        self.main_param_name = "square"
        self.task = None
        self.type = None
        self.forecast_years = None
        self.finished = False

        if count_of_trajectories is None:
            self.count_of_trajectories = count_of_trajectories
        else:
            self.count_of_trajectories = count_of_trajectories

    def import_from_message(self, message: dict, type_task: str):
        """Импорт данных из сообщения
        :param message: сообщение прочитанное из кафки
        :param type_task : тип главной задачи (прогнозирование(recovery) или восстановление(forecast))
        """

        # забираем описание задачи
        self.id = message["id"] if "id" in message.keys() else str(uuid.uuid4())
        self.main_param_name = message["main_param_name"] if "main_param_name" in message.keys() else "square"
        self.count_of_trajectories = message["count_of_trajectories"] \
            if "count_of_trajectories" in message.keys() else self.count_of_trajectories
        self.type = type_task

        if self.type == "forecast":
            if "period_type" in message.keys():
                period_type = message["period_type"]
                if period_type == "short":
                    self.forecast_years = int(static_config["RANDOMIZE_CONFIG"]["randomize.short_time_period"])
                elif period_type == "middle":
                    self.forecast_years = int(static_config["RANDOMIZE_CONFIG"]["randomize.middle_time_period"])
                elif period_type == "long":
                    self.forecast_years = int(static_config["RANDOMIZE_CONFIG"]["randomize.long_time_period"])
            else:
                self.forecast_years = 5

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
        self.main_param_array = np.zeros(len(years))
        for i in range(0, len(years)):
            self.main_param_array[i] = param_data[self.main_param_name][years[i]]
            for j in range(0, len(secondary_params)):
                logging.info("Импорт второстепенного параметра {}".format(secondary_params[j]))
                self.secondary_param_matrix[i, j] = param_data[secondary_params[j]][years[i]]

                    # \
                    # if years[i] in param_data[self.main_param_name].keys() else None
        print(self.main_param_array)

    def _test_data(self):
        # TODO: добавить тестирование
        return True

    def make_task(self):
        if self.type == "recovery":
            self.task = RecoveryTask(id_=self.id, parent_ids=[self.id], main_param=self.main_param_array,
                                     second_param=self.secondary_param_matrix, main_param_name=self.main_param_name,
                                     count_of_trajectories=self.count_of_trajectories)
        else:
            self.task = ForecastTask(id_=self.id, parent_ids=[self.id], main_param=self.main_param_array,
                                     second_param=self.secondary_param_matrix, main_param_name=self.main_param_name,
                                     second_param_names=self.secondary_param_names,
                                     count_of_trajectories=self.count_of_trajectories)

    def receive_message(self, message: dict):
        logging.info("Пришло сообщение в таск менеджер")
        message["parent_ids"].pop(0)
        self.task.receive_message(message)

    def make_message(self):
        message = self.task.make_message()
        if self.task.state == State.finished:
            self.finished = True
        return message
