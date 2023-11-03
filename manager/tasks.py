import uuid
import numpy as np
from manager.utils import State


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

    def all_children_is_finished(self):
        finished = True
        for task in self.child_tasks:
            if task.state != State.finished:
                finished = False
                break
        return finished

    def receive_result(self, result):
        self.result = result
        self.state = State.finished

    def receive_message(self, message: dict):
        # проверим есть ли "parent_ids"
        if len(message["parent_ids"]) == 0:
            # значит пришло мне
            return self.receive_result(message["result"])

        # значит пришло не нам, а моим дочерним задачам
        child_task_id = message["parent_ids"].pop(0)
        # проверяем пришло ли нашим детям сообщение
        for task in self.child_tasks:
            if child_task_id == task.id:
                return task.receive_message(message)

    def make_message(self):
        messages = []
        # если задача выполняется - то нам нечего возвращать
        if self.state == State.running:
            print("State.running")
            return messages
        parent_ids = self.parent_ids.copy()
        parent_ids.append(self.id)
        task_message = {
            "ids_path": parent_ids,
            "main_param": list(self.main_param),
            "second_param": self.second_param,
            "type": self.__class__.__name__
        }
        messages.append(task_message)
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
        # проведем анализ, нужны ли нам подзадачи
        parent_ids = self.parent_ids.copy()
        parent_ids.extend([self.id])

        # если тип прогнозирование - значит может быть нужно восстановление
        if self.check_need_for_recover():
            recovery_task = RecoveryTask(parent_ids=parent_ids, main_param=self.main_param,
                                         second_param=self.second_param, main_param_name=self.main_param_name)
            self.child_tasks.append(recovery_task)

        # проверим нужно ли прогнозирование второстепенных параметров
        if self.second_param is not None:
            count_secondary_param = self.second_param.shape[1]
            for i in range(0, count_secondary_param):
                second_param = self.second_param[:, i]
                forecast_task = ForecastTask(parent_ids=parent_ids, main_param=second_param,
                                             second_param=None, main_param_name=self.second_param_names[i])
                self.child_tasks.append(forecast_task)
        if len(self.child_tasks) != 0:
            self.state = State.wait

    def check_need_for_recover(self):
        if len(np.where(self.main_param == None)[0]) == 0:
            return False
        return True

    def collect_child_data(self):
        # собираем результат работы дочерних задач

        if len(self.child_tasks) == 0:
            return

        # проверяем - было ли восстановление
        recovery_task = list(filter(lambda x: x.__class__.__name__ == "RecoveryTask", self.child_tasks))
        if len(recovery_task) != 0:
            # если было восстановление - то заменяем старые данные восстановленными
            self.main_param = recovery_task[0].main_param

        if self.second_param == np.array([]):
            return

        forecasted_second_param = np.zeros((self.second_param.shape[0]+self.forecast_years, self.second_param.shape[1]))
        forecast_tasks = list(filter(lambda x: x.__class__.__name__ == "RecoveryTask", self.child_tasks))

        for task in forecast_tasks:
            task_name = task.main_param_name
            num = self.second_param_names.index(task_name)
            forecasted_second_param[:, num] = task.main_param
        self.second_param = forecasted_second_param

    def receive_result(self, result):
        self.main_param = result
        self.state = State.finished

    def make_message(self):
        messages = []

        if self.state == State.running:
            return messages

        # проверим есть ли не оконченные задачи
        if self.all_children_is_finished():
            self.collect_child_data()
            messages.extend(super().make_message())
            self.state = State.running

        # проверим сообщения от детей
        for task in self.child_tasks:
            child_message = task.make_message()
            messages.extend(child_message)
        return messages
