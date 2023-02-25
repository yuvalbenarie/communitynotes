import pandas as pd

from tsv_writers import tsv_writer_base


class DataFrameTSVWriter(tsv_writer_base.TSVWriterBase):
    """
    Format a DataFrame as a TSV and store to local disk.
    """

    def write(self, df: pd.DataFrame) -> None:
        """Perform the formatting and write-to-disk operations.

        Args:
          df: pd.DataFrame to write to disk.

        Returns:
          None, because path is always None.
        """

        # Note that:
        #   index is set to False so the index column will not be written to disk.
        #   header is set to True so the first line of the output will contain row names.
        result = df.to_csv(
            path_or_buf=self._path,
            index=False,
            header=True,
            sep="\t",
        )
        if result is not None:
            raise RuntimeError(f"Expected TSV to be written to {self._path}, but it was written to memory.")
