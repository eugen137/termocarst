import logging
import uuid

import numpy as np

from randomize_modeling import RandomizeForecast, RandomizeRecover
from utils import send_worker_list


class Worker:
    main_param = np.array([])
    secondary_matrix = np.array([])
    ids_path = []
    type = "ForecastTask"
    state = 'free'
    step = 'learning'
    theta = []

    def __init__(self, id_=None):
        self.count_of_trajectories = None
        self.forecast_years = None
        self.id = id_ if id_ else str(uuid.uuid4())

    async def send_first_message(self):
        await send_worker_list(mess={"status": "free"}, key=self.id)

    def import_from_message(self, message):
        logging.info("Импорт данных задачи из сообщения")
        self.main_param = np.array(message['main_param'])
        self.secondary_matrix = np.array(message['second_param']) if 'second_param' in message.keys() else None
        self.type = message["type"]
        self.step = message["step"]
        self.ids_path = message["ids_path"]
        if self.step == "sampling":
            self.count_of_trajectories = message["count_of_trajectories"]
            self.forecast_years = message["forecast_years"] if 'forecast_years' in message.keys() else None
            self.theta = np.array(message["theta"])
            print(self.theta)
        logging.info("Импорт данных из сообщения закончен")
        logging.info(f"Параметры задачи: main_param={self.main_param}, type={self.type}, step={self.step} "
                     f"forecast_years={self.forecast_years}, "
                     f"count_of_trajectories={self.count_of_trajectories}")

    def calc(self):
        logging.info("Начало вычислений")
        self.state = 'busy'
        rand_type = RandomizeForecast if self.type == "ForecastTask" else RandomizeRecover
        rand_model = rand_type(self.ids_path[0], self.main_param, self.secondary_matrix)
        ans = {"parent_ids": self.ids_path, "step": self.step}
        if self.step == "learning":
            learning_status = rand_model.learning()
            self.state = 'free'
            if learning_status:
                ans["result"] = list(rand_model.theta)
            else:
                self.state = 'failed'
                ans["result"] = list()
                ans["error"] = "Не удалось вычислить множители Лагранжа",
        else:
            rand_model.theta = self.theta
            if rand_type == RandomizeForecast:
                result = rand_model.modeling_mult(n=self.count_of_trajectories, forecast_years=self.forecast_years)
            else:
                result = rand_model.modeling(n=self.count_of_trajectories)
            self.state = 'free'
            ans["result"] = list(result)
        ans["state"] = self.state
        return ans
