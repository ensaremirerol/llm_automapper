from abc import ABC, abstractmethod
from ast import TypeVar

T = TypeVar("T")


class TextExtractorI[T](ABC):
    @abstractmethod
    def extract(self, data: T) -> str: ...
