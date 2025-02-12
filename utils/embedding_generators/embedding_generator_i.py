from abc import ABC, abstractmethod
from ast import TypeVar
from typing import Any

import numpy as np
from torch import Tensor

T = TypeVar("T")


class EmbeddingGeneratorI[T](ABC):
    @abstractmethod
    def generate(self, data: T) -> tuple[list[Any], Tensor]:
        pass
