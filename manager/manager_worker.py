from manager.utils import WorkerState, send_to_current_task


class Worker:
    def __init__(self, id_):
        self.id = id_
        self.state = WorkerState.free
        self.task_message = {}

    async def start_work(self):
        self.state = WorkerState.busy
        await send_to_current_task(self.task_message, self.id)