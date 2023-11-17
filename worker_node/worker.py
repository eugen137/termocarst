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
        self.ids_path = message["ids_path"]
        self.count_of_trajectories = message["count_of_trajectories"]
        self.forecast_years = message["forecast_years"] if 'forecast_years' in message.keys() else None
        logging.info("Импорт данных из сообщения закончен")
        logging.info("Параметры задачи: main_param={}, type={}, "
                     "forecast_years={}, "
                     "count_of_trajectories={}".format(self.main_param, self.type,
                                                       self.forecast_years, self.count_of_trajectories))

    def calc(self):
        logging.info("Начало вычислений")
        self.state = 'busy'
        rand_type = RandomizeForecast if self.type == "ForecastTask" else RandomizeRecover
        rand_model = rand_type(self.ids_path[0], self.main_param, self.secondary_matrix)
        learning_status = rand_model.learning()
        if learning_status:
            if rand_type == RandomizeForecast:
                result = rand_model.modeling_mult(n=self.count_of_trajectories, forecast_years=self.forecast_years)
            else:
                result = rand_model.modeling(n=self.count_of_trajectories)
            self.state = 'free'
            ans = {"parent_ids": self.ids_path, "result": list(result), "state": "free"}
        else:
            self.state = 'failed'
            ans = {"parent_ids": self.ids_path, "result": list(), "error": "Не удалось вычислить множители Лагранжа",
                   "state": "failed"}
        return ans
