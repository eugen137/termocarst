import datetime
import logging
import uuid
import numpy as np
from configuration import static_config
from tasks import RecoveryTask, ForecastTask
from utils import State


class TaskManager:
    """
    Элемент менеджера задач
    """
    def __init__(self,  message: dict, type_task: str, task_id=None, main_param_array=None, secondary_param_matrix=None,
                 count_of_trajectories=None):
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
        self.start_time = datetime.datetime.now()

        if count_of_trajectories is None:
            self.count_of_trajectories = count_of_trajectories
        else:
            self.count_of_trajectories = count_of_trajectories

        self.import_from_message(message, type_task)

    def import_from_message(self, message: dict, type_task: str):
        """Импорт данных из сообщения
        :param message: сообщение прочитанное из кафки
        :param type_task : тип главной задачи (прогнозирование(recovery) или восстановление(forecast))
        """
        # забираем описание задачи
        if self.id is None:
            self.id = message["id"] if "id" in message.keys() else str(uuid.uuid4())
        logging.info("id={}. Импорт таск".format(self.id))
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
                logging.info("id={}. Определен период прогнозирования - {}".format(self.id, period_type))
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
            self.main_param_array[i] = param_data[self.main_param_name][years[i]] \
                if years[i] in param_data[self.main_param_name].keys() else None
            for j in range(0, len(secondary_params)):
                self.secondary_param_matrix[i, j] = param_data[secondary_params[j]][years[i]]

    def make_task(self):
        logging.info("***********************Начато создание задач****************************")
        if self.type == "recovery":
            logging.info("Создается основная задача восстановления")
            self.task = RecoveryTask(id_=self.id, parent_ids=[self.id], main_param=self.main_param_array,
                                     second_param=self.secondary_param_matrix, main_param_name=self.main_param_name,
                                     count_of_trajectories=self.count_of_trajectories)
            logging.info("Создана основная задача восстановления")
        else:
            logging.info("Создается основная задача прогнозирования")
            self.task = ForecastTask(id_=self.id, parent_ids=[self.id], main_param=self.main_param_array,
                                     second_param=self.secondary_param_matrix, main_param_name=self.main_param_name,
                                     second_param_names=self.secondary_param_names,
                                     count_of_trajectories=self.count_of_trajectories,
                                     forecast_years=self.forecast_years)
            logging.info("Создана основная задача прогнозирования")
        logging.info("***********************Окончено создание задач****************************")

    def receive_message(self, message: dict):
        logging.info("Пришло сообщение в таск менеджер")
        message["parent_ids"].pop(0)
        self.task.receive_message(message)

    def make_message(self):
        message = self.task.make_message()
        if self.task.state == State.finished:
            self.finished = True
            second_param = self.task.second_param.tolist() if type(self.task.second_param) is np.ndarray else None
            message = {"id": self.id,
                       "main_param_name": self.main_param_name,
                       "result": self.task.result,
                       "secondary_param_names": self.task.second_param_names,
                       "secondary_param": second_param
                       }
        return message
