import os


class Logger:
    """Basic logger. """

    def __init__(self, path):
        self.path = path

    def log_scalar(self, tag, value, step):
        """Log single value.
        Attrs:
            tag (str): name of the value.
            value (float): value to log.
            step (int): step for the x-axis.
        """
        path = os.path.join(self.path, tag)
        flag = "a" if os.path.isfile(path) else "w"
        with open(path, flag) as f:
            f.write(f"{value}\n")
