import os
import time
import logging
from shutil import copyfile


class Logger:
    def __init__(
        self,
        log_level: str,
        resume: bool = True,
        log_dir: str = None,
        tag: str = None,
        backup_list: list = [],
        _init_logger: bool = True,
    ) -> None:
        """
        The log handler for training

        Parameters:
            log_level: str
                the log level which must be set as `DEBUG`, `INFO` or `ERROR`
            resume: bool
                overwrite the log files if true else do not
            log_dir: str
                the directory path to save the log file
            tag: str
                the log tag
            backup_list: list
                backup important files
            _init_logger: bool
                whether to create logger upon class initialization, useful when re-creating loggers
                in child processes
        """
        if log_level == "DEBUG":
            self.level = logging.DEBUG
        elif log_level == "INFO":
            self.level = logging.INFO
        elif log_level == "ERROR":
            self.level = logging.ERROR
        else:
            raise NotImplementedError("log_level must be set as `DEBUG`, `INFO` or `ERROR`")

        self.file_mode = "a" if resume else "w"

        # main log
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        dir_name = f"{timestamp}_{tag}" if tag else f"{timestamp}"
        log_path = log_dir if log_dir else os.path.abspath(os.path.join(__file__, "../"))
        self.log_dir = os.path.join(log_path, dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "output.log")

        # backup important files (e.g. config.yaml)
        self.backup_dir = os.path.join(self.log_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)
        for file in backup_list:
            copyfile(
                os.path.abspath(file),
                os.path.join(self.backup_dir, os.path.basename(file)),
            )

        # storing results (network output etc.)
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)

        # storing checkpoints
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        sigterm_ckpt_name = f"sigterm_ckpt_{tag}.pth" if tag else f"sigterm_ckpt.pth"
        self.sigterm_ckpt_file = os.path.join(self.log_dir, sigterm_ckpt_name)

        # tensorboard
        self.tb_dir = os.path.join(self.log_dir, "tensorboard")
        os.makedirs(self.tb_dir, exist_ok=True)

        # create logger
        self.logger_name = f"Logger_{timestamp}_{tag}" if tag else f"Logger_{timestamp}"
        self.logger = None
        if _init_logger:
            self.logger = self.create()

    def create(self) -> logging.Logger:
        """
        Create the log handler

        Returns:
            logger: logging.Logger
                the log handler
        """
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)
        formatter = logging.Formatter(
            "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self.log_file is not None:
            file_handler = logging.FileHandler(filename=self.log_file, mode=self.file_mode)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger

    def INFO(self, s: str) -> None:
        self.logger.info(s)

    def DEBUG(self, s: str) -> None:
        self.logger.debug(s)

    def WARN(self, s: str) -> None:
        self.logger.warning(s)

    def ERROR(self, s: str) -> None:
        self.logger.error(s)
