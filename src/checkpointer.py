import orbax.checkpoint as ocp


class Checkpointer:
    # a class to make saving/loading to a single path easier, and to help update to newer APIs later
    def __init__(self, checkpoint_path):
        self.checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        self.checkpoint_path = checkpoint_path

    def save(self, state, force=True):
        self.checkpointer.save(self.checkpoint_path, state, force=force)

    def restore(self, state):
        self.checkpointer.wait_until_finished()  # finish saving checkpoints first
        return self.checkpointer.restore(self.checkpoint_path, state)
