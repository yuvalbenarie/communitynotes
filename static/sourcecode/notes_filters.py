import numpy as np
import pandas as pd

import constants


class FilterDuplicateNotes:
    @classmethod
    def filter(cls, notes: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate notes, then assert that there is only one copy of each noteId.
        Args:
            notes (pd.DataFrame): with possible duplicate notes
        Returns:
            notes (pd.DataFrame) with one record per noteId
        """
        # Construct a new DataFrame to avoid SettingWithCopyWarning
        notes = pd.DataFrame(notes.drop_duplicates())

        num_notes = len(notes)
        num_unique_notes = len(np.unique(notes[constants.noteIdKey]))
        if num_notes != num_unique_notes:
            raise ValueError(f"Found only {num_unique_notes} unique noteIds out of {num_notes} notes.")
        return notes
