import logging
import uuid
import numpy as np
from utils import State


class Task:
    def __init__(self, parent_ids, main_param, second_param, main_param_name, second_param_names=None, id_=None,
                 count_of_trajectories=100):
        self.id = id_ if id_ is not None else str(uuid.uuid4())
        self.parent_ids = parent_ids if type(parent_ids) is list else [parent_ids]
        self.main_param = main_param
        self.second_param = second_param
        self.second_param_names = second_param_names
        self.state = State.created
        self.main_param_name = main_param_name
        self.result = None
        self.child_tasks = []
        self.count_of_trajectories = count_of_trajectories
        logging.info("Создана задача с типом {}, главный параметр {}".format(self.__class__.__name__, main_param_name))

    def all_children_is_finished(self):
        logging.info("id={}, Проверка детей на выполнение".format(self.id))
        finished = True
        for task in self.child_tasks:
            logging.info("id={}, Дочерняя задача {} со статусом {}".format(self.id, task.id, task.state))
            if task.state != State.finished:
                finished = False
                break
        return finished

    def receive_result(self, result):
        logging.info("id={}, Получен результат работы задачи {}. Результат - {}".format(self.id, self.id, result))
        self.result = result
        self.state = State.finished

    def receive_message(self, message: dict):
        logging.info("id={}, Пришло сообщение в таск".format(self.id))
        # проверим есть ли "parent_ids"
        if len(message["parent_ids"]) == 0:
            # значит пришло мне
            logging.info("id={}, Пришло сообщение для текущей задачи {}".format(self.id, self.id))
            return self.receive_result(message["result"])
        # значит пришло не нам, а моим дочерним задачам
        child_task_id = message["parent_ids"].pop(0)
        logging.info("id={}, Пришло сообщение для дочерней задачи {}, путь {}".format(self.id, child_task_id, message["parent_ids"]))
        # проверяем пришло ли нашим детям сообщение
        for task in self.child_tasks:
            if child_task_id == task.id:
                logging.info("id={}, Найдено для кого предназначено сообщение".format(self.id))
                return task.receive_message(message)

    def make_message(self):
        logging.info("id={}, Запущена генерация сообщения задачи {}".format(self.id, self.id))
        messages = []
        # если задача выполняется - то нам нечего возвращать
        if self.state == State.running or self.state == State.finished or self.result is not None:
            logging.info("id={}, Задача {} в процессе выполнения, сообщение не готово".format(self.id, self.id))
            return messages
        parent_ids = self.parent_ids.copy()
        parent_ids.append(self.id)
        second_param = self.second_param.tolist() if type(self.second_param) is np.ndarray else None
        task_message = {
            "ids_path": parent_ids,
            "main_param": list(self.main_param),
            "second_param": second_param,
            "count_of_trajectories": self.count_of_trajectories,
            "type": self.__class__.__name__
        }
        self.state = State.running
        messages.append(task_message)
        logging.info("id={}, Сообщение сгенерировано {}".format(self.id, messages))
        return messages


class RecoveryTask(Task):
    def __init__(self, parent_ids, main_param, second_param, main_param_name, id_=None, count_of_trajectories=100):
        super().__init__(parent_ids=parent_ids, main_param=main_param, second_param=second_param,
                         main_param_name=main_param_name,  id_=id_, count_of_trajectories=count_of_trajectories)


class ForecastTask(Task):
    def __init__(self, parent_ids, main_param, second_param, main_param_name, second_param_names=None, id_=None,
                 forecast_years=5, count_of_trajectories=100):
        super().__init__(parent_ids=parent_ids, main_param=main_param, second_param=second_param,
                         main_param_name=main_param_name, second_param_names=second_param_names, id_=id_,
                         count_of_trajectories=count_of_trajectories)
        self.forecast_years = forecast_years
        self.create_subtasks()

    def create_subtasks(self):
        logging.info("id={}, Запущено создание подзадач".format(self.id))
        # проведем анализ, нужны ли нам подзадачи
        parent_ids = self.parent_ids.copy()
        parent_ids.extend([self.id])

        # если тип прогнозирование - значит может быть нужно восстановление
        if self.check_need_for_recover():
            logging.info("id={}, Для выполнения задачи нужно восстановление".format(self.id))
            recovery_task = RecoveryTask(parent_ids=parent_ids, main_param=self.main_param,
                                         second_param=self.second_param, main_param_name=self.main_param_name)
            self.child_tasks.append(recovery_task)

        # проверим нужно ли прогнозирование второстепенных параметров
        if self.second_param is not None:
            logging.info("id={}, Для задачи {} нужно прогнозирование второстепенных параметров".format(self.id, self.id))
            count_secondary_param = self.second_param.shape[1]
            for i in range(0, count_secondary_param):
                second_param = self.second_param[:, i]
                forecast_task = ForecastTask(parent_ids=parent_ids, main_param=second_param,
                                             second_param=None, main_param_name=self.second_param_names[i])
                self.child_tasks.append(forecast_task)
        if len(self.child_tasks) != 0:
            self.state = State.wait

    def check_need_for_recover(self):
        if np.count_nonzero(np.isnan(self.main_param)) == 0:
            return False
        return True

    def collect_child_data(self):
        # собираем результат работы дочерних задач
        logging.info("id={}, Сбор данных с подзадач".format(self.id))
        if len(self.child_tasks) == 0:
            logging.info("id={}, Подзадач нет".format(self.id))
            return

        # проверяем - было ли восстановление
        print([t.__class__.__name__ for t in self.child_tasks])
        recovery_task = list(filter(lambda x: x.__class__.__name__ == "RecoveryTask", self.child_tasks))
        if len(recovery_task) != 0:
            logging.info("id={}, Было восстановление, забираем восстановленные параметры {}".format(self.id, self.id))
            # если было восстановление - то заменяем старые данные восстановленными
            print(recovery_task[0].main_param)
            self.main_param = recovery_task[0].result

        if self.second_param.shape == (0,):
            logging.info("id={}, Второстепенных параметров в задаче нет".format(self.id))
            return

        forecasted_second_param = np.zeros((self.second_param.shape[0]+self.forecast_years, self.second_param.shape[1]))
        forecast_tasks = list(filter(lambda x: x.__class__.__name__ == "ForecastTask", self.child_tasks))
        logging.info("id={}, Количество задач прогнозирования {}".format(self.id, len(forecast_tasks)))
        for task in forecast_tasks:
            task_name = task.main_param_name
            logging.info("id={}, Сбор данных с задачи {}, параметр {}".format(self.id, task.id, task_name))
            num = self.second_param_names.index(task_name)
            forecasted_second_param[:, num] = task.main_param
        self.second_param = forecasted_second_param

    def receive_result(self, result):
        logging.info("id={}, Получен результат прогнозирования".format(self.id))
        self.result = result
        self.main_param = result
        self.state = State.finished

    def make_message(self):
        logging.info("id={}, Запущена генерация сообщения задачи прогнозирования {}".format(self.id, self.id))
        messages = []

        if self.state == State.running or self.state == State.finished or self.result is not None:
            return messages

        # проверим есть ли не оконченные задачи
        if self.all_children_is_finished():
            logging.info("id={}, Все дочерние процессы выполнены".format(self.id))
            self.collect_child_data()
            messages.extend(super().make_message())
            if self.state == State.wait:
                self.state = State.running

        # проверим сообщения от детей
        for task in self.child_tasks:
            child_message = task.make_message()
            messages.extend(child_message)
        messages_new = []
        for m in messages:
            m["forecast_years"] = self.forecast_years
            messages_new.append(m)
        return messages_new
