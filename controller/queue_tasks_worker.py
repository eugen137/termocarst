import logging
import uuid

from manager_worker import Worker
from tasks_manager import TaskManager
from utils import WorkerState, send_answer


class Queue:
    workers = dict()
    messages_for_workers = []
    task_managers = dict()

    async def add_main_task_message(self, message, type_task):
        logging.info("Добавление сообщения {}".format(message))
        logging.info("Перед добавлением {}".format(self.task_managers))
        id_ = message["id"] if "id" in message.keys() else str(uuid.uuid4())
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

    async def receive_result_of_workers(self, result):
        # получим id задачи
        id_ = result["parent_ids"].pop(0)
        logging.info("Для задачи {} получен результат работы {}".format(id_, result))
        self.task_managers[id_].receive_message(result)
        answer = self.task_managers[id_].make_message()
        if self.task_managers[id_].finished:
            print("задача выполнена")
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
                self.workers[worker_id].state = WorkerState.free
                if message["state"] == WorkerState.free.name:
                    # если он стал свободен, значит закончил вычисления
                    await self.receive_result_of_workers(message)
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
