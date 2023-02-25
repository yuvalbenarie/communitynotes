import typing

import constants

from tsv_readers import tsv_reader_base


class RatingsTSVReader(tsv_reader_base.TSVReaderBase):
  @property
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    return constants.ratingTSVTypeMapping
