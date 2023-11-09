import logging

from manager.manager_worker import Worker
from manager.tasks_manager import TaskManager
from manager.utils import WorkerState


class Queue:
    workers = dict()
    messages_for_workers = []
    task_managers = dict()

    async def add_main_task_message(self, message, type_task):
        task_manager = TaskManager()
        task_manager.import_from_message(message, type_task=type_task)
        self.task_managers[task_manager.id] = task_manager
        self.task_managers[task_manager.id].make_task()
        mess = self.task_managers[task_manager.id].make_message()
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
            # TODO: задача выполнена, нужно отправить результат
            print("задача выполнена")
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
                free_workers.append(worker.id)
        for worker in free_workers:
            if len(self.messages_for_workers) == 0:
                break
            message = self.messages_for_workers.pop(0)
            logging.info("Задача {} назначена воркеру {}".format(message, worker))
            self.workers[worker].task_message = message
            await self.workers[worker].start_work()
