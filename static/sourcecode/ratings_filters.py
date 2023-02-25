import logging

import numpy as np
import pandas as pd

import constants

_logger = logging.getLogger(__name__)


class FilterDuplicateRatings:
    @classmethod
    def filter(cls, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicate ratings, then assert that there is exactly one rating per noteId per raterId.
        Args:
            ratings (pd.DataFrame) with possible duplicated ratings
        Returns:
            pd.DataFrame: ratings, with one record per userId, noteId.
        """
        # Construct a new DataFrame to avoid SettingWithCopyWarning
        ratings = pd.DataFrame(ratings.drop_duplicates())

        num_ratings = len(ratings)
        num_unique_rater_id_note_id_pairs = len(
            ratings.groupby([constants.raterParticipantIdKey, constants.noteIdKey]).head(1),
        )
        if num_ratings != num_unique_rater_id_note_id_pairs:
            raise ValueError(
                f"Only {num_unique_rater_id_note_id_pairs} unique raterId,noteId pairs but {num_ratings} ratings",
            )
        return ratings


class FilterRatingsOfMisleadingNotes:
    @classmethod
    def filter(
            cls,
            ratings: pd.DataFrame,
            noteStatusHistory: pd.DataFrame,
            should_log: bool = True,
    ) -> pd.DataFrame:
        """
        This function filters ratings, based on which notes they rate.

        Filter out ratings of notes that say the Tweet isn't misleading.
        Also filter out ratings of deleted notes, unless they were deleted after
        c.deletedNotesTombstoneLaunchTime, and appear in noteStatusHistory.

        Args:
          ratings (pd.DataFrame): _description_
          noteStatusHistory (pd.DataFrame): _description_
          should_log (bool, optional): _description_. Defaults to True.

        Returns:
          pd.DataFrame: filtered ratings
        """
        ratings = ratings.merge(
            noteStatusHistory[[constants.noteIdKey, constants.createdAtMillisKey, constants.classificationKey]],
            on=constants.noteIdKey,
            how="left",
            suffixes=("", "_nsh"),
        )

        deletedNoteKey = "deletedNote"
        notDeletedMisleadingKey = "notDeletedMisleading"
        deletedButInNSHKey = "deletedButInNSH"
        createdAtMillisNSHKey = constants.createdAtMillisKey + "_nsh"

        ratings[deletedNoteKey] = pd.isna(ratings[constants.classificationKey])
        ratings[notDeletedMisleadingKey] = np.invert(ratings[deletedNoteKey]) & (
                ratings[constants.classificationKey] == constants.notesSaysTweetIsMisleadingKey
        )
        ratings[deletedButInNSHKey] = ratings[deletedNoteKey] & np.invert(
            pd.isna(ratings[createdAtMillisNSHKey])
        )

        deletedNotInNSH = (ratings[deletedNoteKey]) & pd.isna(ratings[createdAtMillisNSHKey])
        notDeletedNotMisleadingOldUI = (
                                               ratings[
                                                   constants.classificationKey] == constants.noteSaysTweetIsNotMisleadingKey
                                       ) & (ratings[createdAtMillisNSHKey] <= constants.notMisleadingUILaunchTime)
        notDeletedNotMisleadingNewUI = (
                                               ratings[
                                                   constants.classificationKey] == constants.noteSaysTweetIsNotMisleadingKey
                                       ) & (ratings[createdAtMillisNSHKey] > constants.notMisleadingUILaunchTime)

        if should_log:
            print(
                f"Preprocess Data: Filter misleading notes, starting with {len(ratings)} ratings on {len(np.unique(ratings[constants.noteIdKey]))} notes"
            )
            print(
                f"  Keeping {ratings[notDeletedMisleadingKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[notDeletedMisleadingKey], constants.noteIdKey]))} misleading notes"
            )
            print(
                f"  Keeping {ratings[deletedButInNSHKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[deletedButInNSHKey], constants.noteIdKey]))} deleted notes that were previously scored (in note status history)"
            )
            print(
                f"  Removing {notDeletedNotMisleadingOldUI.sum()} ratings on {len(np.unique(ratings.loc[notDeletedNotMisleadingOldUI, constants.noteIdKey]))} older notes that aren't deleted, but are not-misleading."
            )
            print(
                f"  Removing {deletedNotInNSH.sum()} ratings on {len(np.unique(ratings.loc[deletedNotInNSH, constants.noteIdKey]))} notes that were deleted and not in note status history (e.g. old)."
            )

        ratings = ratings[
            ratings[notDeletedMisleadingKey] | ratings[deletedButInNSHKey] | notDeletedNotMisleadingNewUI
            ]
        ratings = ratings.drop(
            columns=[
                createdAtMillisNSHKey,
                constants.classificationKey,
                deletedNoteKey,
                notDeletedMisleadingKey,
                deletedButInNSHKey,
            ]
        )
        return ratings


class FilterRatingsForTraining:
    @classmethod
    def filter(cls, ratings: pd.DataFrame, should_log: bool = True) -> pd.DataFrame:
        """
        Apply min number of ratings for raters & notes. Instead of iterating these filters
        until convergence, simply stop after going back and force once.

        Args:
            ratings (pd.DataFrame): unfiltered ratings
            should_log (bool, optional): debug output. Defaults to True.

        Returns:
            pd.DataFrame: filtered ratings
        """

        if should_log:
            _logger.info("Filtering notes and ratings with too few ratings.")

        ratings = cls._filter_ratings_on_notes_with_minimal_num_ratings(
            ratings=ratings,
            should_log=should_log,
        )

        ratings = cls._filter_ratings_from_raters_with_minimal_num_ratings(
            ratings=ratings,
            should_log=should_log,
        )

        ratings = cls._filter_ratings_on_notes_with_minimal_num_ratings(
            ratings=ratings,
            should_log=should_log,
        )

        return ratings

    @classmethod
    def _filter_ratings_on_notes_with_minimal_num_ratings(
            cls,
            ratings: pd.DataFrame,
            should_log: bool,
    ) -> pd.DataFrame:
        ratings_grouped_by_note_id = ratings.groupby(constants.noteIdKey).size().reset_index()
        notes_with_min_num_ratings = ratings_grouped_by_note_id[
            ratings_grouped_by_note_id[0] >= constants.minNumRatersPerNote
        ]

        filtered_ratings = ratings.merge(
            notes_with_min_num_ratings[[constants.noteIdKey]],
            on=constants.noteIdKey,
        )

        if should_log:
            _logger.info(
                "After Filtering Notes w/less than %d Ratings, Num Ratings: %d, "
                "Num Unique Notes Rated: %d, Num Unique Raters: %d",
                constants.minNumRatersPerNote,
                len(filtered_ratings),
                len(np.unique(filtered_ratings[constants.noteIdKey])),
                len(np.unique(filtered_ratings[constants.raterParticipantIdKey])),
            )
        return filtered_ratings

    @classmethod
    def _filter_ratings_from_raters_with_minimal_num_ratings(
            cls,
            ratings: pd.DataFrame,
            should_log: bool,
    ) -> pd.DataFrame:
        ratings_grouped_by_rater_id = ratings.groupby(constants.raterParticipantIdKey).size().reset_index()
        raters_with_min_num_ratings = ratings_grouped_by_rater_id[
            ratings_grouped_by_rater_id[0] >= constants.minNumRatingsPerRater
        ]

        filtered_ratings = ratings.merge(
            raters_with_min_num_ratings[[constants.raterParticipantIdKey]],
            on=constants.raterParticipantIdKey,
        )
        if should_log:
            _logger.info(
                "After Filtering Raters w/less than %s Notes, Num Ratings: %d, "
                "Num Unique Notes Rated: %d, Num Unique Raters: %d",
                constants.minNumRatingsPerRater,
                len(filtered_ratings),
                len(np.unique(filtered_ratings[constants.noteIdKey])),
                len(np.unique(filtered_ratings[constants.raterParticipantIdKey])),
            )
        return filtered_ratings
