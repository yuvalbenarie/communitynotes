import abc


class TSVWriterBase(abc.ABC):
    """
    store a TSV file to local disk.
    """

    def __init__(self, path: str):
        """
        Initialize the TSVWriter instance.

        Args:
            path (str): location of file on disk.
        """
        if path is None:
            raise ValueError("Path must be provided.")
        self._path = path

    @abc.abstractmethod
    def write(self, **kwargs) -> None:
        """Perform the write-to-disk operation."""
        pass
