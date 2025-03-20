import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator

from babisteps import logger


class BaseGenerator(BaseModel, ABC):
    verbosity: Union[int, str] = Field(default=logging.INFO)
    logger: Optional[Any] = None
    log_file: Optional[Path] = None
    original_inputs: Optional[dict] = None
    name: Optional[str] = None

    @model_validator(mode="before")
    def save_inputs_dict(cls, values):
        values["original_inputs"] = deepcopy(values)
        return values

    @model_validator(mode="after")
    def fill_logger(self):
        if not self.logger:
            self.logger = logger.get_logger(
                self.__class__.__name__ if self.name is None else
                self.__class__.__name__ + "-" + self.name,
                level=self.verbosity,
                log_file=self.log_file)
        return self

    def recreate(self):
        """Recreates the instance with the original input values."""
        self.logger.info("Recreating instance with original inputs.",
                         original_inputs=self.original_inputs)
        if self.original_inputs is None:
            raise ValueError("Original inputs not available.")
        # Use self.__class__ so that the child class is recreated
        return self.__class__(**self.original_inputs)

    @abstractmethod
    def generate(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass

    @abstractmethod
    def get_json(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass

    @abstractmethod
    def get_txt(self, **kwargs):
        """Abstract method to be implemented in subclasses."""
        pass
