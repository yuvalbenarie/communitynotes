import logging

import numpy as np
import pandas as pd

import constants as c, note_status_history
import notes_filters
import ratings_filters
import tsv_readers

_logger = logging.getLogger(__name__)


class DataLoader:
  """
  Loader class for all Birdwatch notes and ratings related data from TSV files.
  The `load` method both reads the files and pre-processes the data the contain.
  """

  # Instance members that are initialized outside the __init__ method.
  _notes: pd.DataFrame
  _notes_status_history: pd.DataFrame
  _ratings: pd.DataFrame
  _user_enrollments: pd.DataFrame

  def __init__(
    self,
    notesPath: str,
    ratingsPath: str,
    noteStatusHistoryPath: str,
    userEnrollmentPath: str,
  ):
    """
    Initialize the DataLoader instance with paths to all the required tsv files.

    Args:
        notesPath (str): file path
        ratingsPath (str): file path
        noteStatusHistoryPath (str): file path
        userEnrollmentPath (str): file path
    """
    self._notesPath = notesPath
    self._ratingsPath = ratingsPath
    self._noteStatusHistoryPath = noteStatusHistoryPath
    self._userEnrollmentPath = userEnrollmentPath

  def load(
    self,
    should_filter_not_misleading_notes: bool = True,
    should_log: bool = True,
  ) -> None:
    """All-in-one function for reading Birdwatch notes and ratings from TSV files.
    It does both reading and pre-processing.

    Args:
        should_filter_not_misleading_notes (bool, optional): Throw out not-misleading notes if True. Defaults to True.
        should_log (bool, optional): Print out debug output. Defaults to True.
    """

    self._load_user_enrollments()
    self._load_notes()

    # Dependent on notes, so must come after they are loaded.
    self._load_notes_status_history()

    # Dependent on notes status history, so must come after they are loaded.
    self._load_ratings(
      should_filter_not_misleading_notes=should_filter_not_misleading_notes,
      should_log=should_log,
    )

    if logging:
      self._log_results()

  def _load_user_enrollments(self) -> None:
    self._user_enrollments = tsv_readers.UserEnrollmentsTSVReader(self._userEnrollmentPath).read()

  def _load_notes(self) -> None:
    notes = tsv_readers.NotesTSVReader(path=self._notesPath).read()
    notes = notes_filters.FilterDuplicateNotes.filter(notes=notes)
    notes[c.tweetIdKey] = notes[c.tweetIdKey].astype(np.str)
    self._notes = notes

  def _load_notes_status_history(self) -> None:
    notes_status_history = tsv_readers.NotesStatusHistoryTSVReader(self._noteStatusHistoryPath).read()
    self._notes_status_history = note_status_history.merge_note_info(notes_status_history, self.notes)

  def _load_ratings(
    self,
    should_filter_not_misleading_notes: bool,
    should_log: bool,
  ) -> None:
    ratings = tsv_readers.RatingsTSVReader(path=self._ratingsPath).read()
    ratings = ratings_filters.FilterDuplicateRatings.filter(ratings=ratings)

    # Populate helpfulNumKey, a unified column that merges the helpfulness answers from
    # the V1 and V2 rating forms together, as described in
    # https://twitter.github.io/communitynotes/ranking-notes/#helpful-rating-mapping.
    ratings.loc[:, c.helpfulNumKey] = np.nan
    ratings.loc[ratings[c.helpfulKey] == 1, c.helpfulNumKey] = 1
    ratings.loc[ratings[c.notHelpfulKey] == 1, c.helpfulNumKey] = 0
    ratings.loc[ratings[c.helpfulnessLevelKey] == c.notHelpfulValueTsv, c.helpfulNumKey] = 0
    ratings.loc[ratings[c.helpfulnessLevelKey] == c.somewhatHelpfulValueTsv, c.helpfulNumKey] = 0.5
    ratings.loc[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv, c.helpfulNumKey] = 1
    ratings = ratings.loc[~pd.isna(ratings[c.helpfulNumKey])]

    if should_filter_not_misleading_notes:
      ratings = ratings_filters.FilterRatingsOfMisleadingNotes.filter(
        ratings=ratings,
        noteStatusHistory=self._notes_status_history,
        should_log=should_log,
      )

    self._ratings = ratings

  def _log_results(self) -> None:
    _logger.info(
      "Timestamp of latest rating in data: %s",
      pd.to_datetime(self.ratings[c.createdAtMillisKey], unit="ms").max(),
    )
    _logger.info(
      "Timestamp of latest note in data: %s",
      pd.to_datetime(self.notes[c.createdAtMillisKey], unit="ms").max(),
    )
    _logger.info(
      "Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d",
      len(self.ratings),
      len(np.unique(self.ratings[c.noteIdKey])),
      len(np.unique(self.ratings[c.raterParticipantIdKey])),
    )

  @property
  def user_enrollments(self) -> pd.DataFrame:
    return self._user_enrollments

  @property
  def notes(self) -> pd.DataFrame:
    return self._notes

  @property
  def notes_status_history(self) -> pd.DataFrame:
    return self._notes_status_history
  @property
  def ratings(self) -> pd.DataFrame:
    return self._ratings
