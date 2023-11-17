import datetime
import logging
import uuid
import jsonschema
from jsonschema import validate
from manager_worker import Worker
from tasks_manager import TaskManager
from utils import WorkerState, send_answer


class Queue:
    workers = dict()
    messages_for_workers = []
    task_managers = dict()
    message_schema = {
        "type": "object",
        "properties": {
            "data": {
                "description": "Данные для задачи в формате {[имя_параметра:{[год:значение]}]}",
                "type": "object",
                "patternProperties": {
                    "^.*$": {
                        "type": "object",
                        "patternProperties": {
                            "^[0-9]*$": {"type": "number"}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "period_type": {
                "description": "Временной промежуток прогнозирования (short, middle, long)",
                "enum": ["short", "middle", "long"]},
            "count_of_trajectories": {
                "description": "Количество сэмплированных траекторий",
                "type": "number"},
            "id": {
                "description": "Уникальный идентификатор задачи",
                "type": "string"},
            "main_param_name": {
                "description": "Имя целевого параметра",
                "type": "string"},
        },
        "required": ["id", "data"]
    }

    async def add_main_task_message(self, message, type_task):
        logging.info("Добавление сообщения {}".format(message))
        logging.info("Перед добавлением {}".format(self.task_managers))
        id_ = message["id"] if "id" in message.keys() else str(uuid.uuid4())
        test_data = await self.test_message(message)
        if not test_data["test_status"]:
            logging.error("Ошибка распознания данных")
            logging.error("{}".format(test_data["Error"]))
            answer = {"id": id_,
                      "result": "Данные в неверном формате, ошибка {}".format(test_data["Error"])}
            t = type_task
            topic = "RecoveryAnswer" if t == "recovery" else "ForecastAnswer"
            return await send_answer(answer, "answer", topic)

        self.task_managers[id_] = TaskManager(message, type_task=type_task, task_id=id_)
        self.task_managers[id_].make_task()
        logging.info("После добавления {}".format(self.task_managers))
        mess = self.task_managers[id_].make_message()
        await self.add_task_message(mess)

    async def add_task_message(self, message: list):
        self.messages_for_workers.extend(message)
        await self.distribution_of_messages()

    async def add_worker(self, worker: Worker):
        logging.info("Добавлен в очередь воркер {}".format(worker.id))
        self.workers[worker.id] = worker
        await self.distribution_of_messages()

    async def receive_result_of_workers(self, result, failed=False):
        # получим id задачи
        id_ = result["parent_ids"].pop(0)
        logging.info("Для задачи {} получен результат работы {}".format(id_, result))
        if failed:
            logging.info("Задача {} закончена со статусом {}".format(id_, result["state"]))
            answer = {"id": id_,
                      "result": "Ошибка {}".format(result["error"])}
            t = self.task_managers[id_].type
            topic = "RecoveryAnswer" if t == "recovery" else "ForecastAnswer"
            return await send_answer(answer, "answer", topic)
            pass
        else:
            self.task_managers[id_].receive_message(result)
            answer = self.task_managers[id_].make_message()
            if not answer:
                return
            if self.task_managers[id_].finished:
                logging.info("!!!!!!!!!!!!!!Работа по задаче {} завершена. "
                             "Время на выполнение задачи {}!!!!!!!!!!!!!!!!!!!!!".format(id_, datetime.datetime.now() -
                                                                                         self.task_managers[
                                                                                             id_].start_time))
                t = self.task_managers[id_].type
                topic = "RecoveryAnswer" if t == "recovery" else "ForecastAnswer"
                await send_answer(answer, "answer", topic)
            else:
                # если задача не выполнена, то задача продолжается, добавляем в очередь вычисления
                await self.add_task_message(answer)

    async def add_worker_message(self, worker_id, message):
        logging.info("От воркера {} пришло сообщение {}".format(worker_id, message))
        if worker_id in self.workers.keys():
            # нужно проверить были ли задачи у воркера
            if self.workers[worker_id].state == WorkerState.busy:
                logging.info("Воркер был занят, ожидаем результат")
                # значит воркер был занят

                if message["state"] == WorkerState.free.name:
                    # если он стал свободен, значит закончил вычисления
                    self.workers[worker_id].state = WorkerState.free
                    await self.receive_result_of_workers(message)
                if message["state"] == WorkerState.failed.name:
                    # если он стал свободен, значит закончил вычисления
                    await self.receive_result_of_workers(message, failed=True)
            else:
                # значит воркер не был занят, меняем статус TODO: сделать другие статусы
                self.workers[worker_id].state = WorkerState.busy

        else:
            await self.add_worker(Worker(worker_id))

    async def distribution_of_messages(self):
        logging.info("Запущено распределение задача по воркерам")
        # нужно найти свободных воркеров
        free_workers = []
        for worker in self.workers.values():
            if worker.state == WorkerState.free:
                logging.info("Воркер {} свободен".format(worker.id))
                free_workers.append(worker.id)
        logging.info("Количество задач для воркеров {}".format(len(self.messages_for_workers)))
        for worker in free_workers:
            if len(self.messages_for_workers) == 0:
                break

            message = self.messages_for_workers.pop(0)
            self.workers[worker].task_message = message
            logging.info("Задача {} назначена воркеру {}".format(message["ids_path"], worker))
            await self.workers[worker].start_work()

    async def test_message(self, message):
        # проверка наличия данных
        ans = {"test_status": True,
               "Error": ''}
        try:
            validate(instance=message, schema=self.message_schema)
        except jsonschema.exceptions.ValidationError as err:
            ans["test_status"] = False
            ans["Error"] = str(err)
            return ans
        main_param_name = message["main_param_name"] if "main_param_name" in message.keys() else "square"
        data = message["data"]
        years = []
        for prop in data.keys():
            if prop == main_param_name:
                continue
            if years == []:
                years = list(data[prop].keys())
                years.sort()
                continue
            years2 = list(data[prop].keys())
            years2.sort()
            if years != years2:
                ans["test_status"] = False
                ans["Error"] = "Не равное количество годов у второстепенных сообщений"
                return ans
        logging.info("Проверка данных прошла успешно")
        return ans
