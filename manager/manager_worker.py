from manager.utils import ServerState, send_to_current_task


class Worker:
    def __init__(self, id_):
        self.id = id_
        self.state = ServerState.free
        self.task_message = {}

    def start_work(self):
        await send_to_current_task(self.task_message, self.id)
        self.state = ServerState.busy
