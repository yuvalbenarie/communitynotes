import abc
import logging
import typing

import pandas as pd

_logger = logging.getLogger(__name__)


class TSVReaderBase(abc.ABC):
  def __init__(self, path: str):
    self._path = path

  def read(self) -> pd.DataFrame:
    """
    Reads a tsv file into a pandas DataFrame, and validates the columns read are expected.
    Returns: The data read from the provided file path, as a DataFrame.
    """
    try:
      # First assume no header row in the file, and try to add the headers ourselves by providing `names`.
      results = pd.read_csv(
        filepath_or_buffer=self._path,
        sep=self._separator,
        dtype=self._datatype_mapping,
        names=self._columns,
      )
    except ValueError:
      _logger.info("The file %s might contain a header row. Re-reading without forced column names.", self._path)
      # If failed, assume there is a header row and verify the names after the file is read.
      results = pd.read_csv(
        filepath_or_buffer=self._path,
        sep="\t",
        dtype=self._datatype_mapping,
      )
      self._verify_read_columns(results)

    return results

  def _verify_read_columns(self, results: pd.DataFrame) -> None:
    """
    Verify that the read file has exactly the columns we are expecting.
    Args:
        results (DataFrame): The DataFrame returned by the read method.
    Raises:
        ValueError: if there are any extra or missing columns in the DataFrame.
    """
    actual_columns = set(results.columns.values)
    expected_columns = set(self._columns)
    if actual_columns != expected_columns:
      raise ValueError(
        f"Columns don't match for {self._path}:\n"
        f"{actual_columns - expected_columns} are extra columns,\n"
        f"{expected_columns - actual_columns} are missing.",
      )

  @property
  def _separator(self) -> str:
    return "\t"

  @property
  @abc.abstractmethod
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    pass

  @property
  def _columns(self) -> typing.Tuple[str, ...]:
    return tuple(self._datatype_mapping.keys())
