import argparse
import logging

import algorithm
import constants
import process_data
import tsv_writers

_logger = logging.getLogger(__name__)

"""
Example Usage:
  python main.py \
    --enrollment userEnrollment-00000.tsv \
    --notes_path notes-00000.tsv  \
    --ratings_path ratings-00000.tsv \
    --note_status_history_path noteStatusHistory-00000.tsv
"""


def setup_logging() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def get_args():
  """Parse command line arguments for running on command line.

  Returns:
      args: the parsed arguments
  """
  parser = argparse.ArgumentParser(description="Birdwatch Algorithm.")
  parser.add_argument(
    "-e", "--enrollment", default=constants.enrollmentInputPath, help="user enrollment dataset"
  )
  parser.add_argument("-n", "--notes_path", default=constants.notesInputPath, help="note dataset")
  parser.add_argument("-r", "--ratings_path", default=constants.ratingsInputPath, help="rating dataset")
  parser.add_argument(
    "-s",
    "--note_status_history_path",
    default=constants.noteStatusHistoryInputPath,
    help="note status history dataset",
  )
  parser.add_argument("-o", "--output_path", default=constants.scoredNotesOutputPath, help="output path")
  return parser.parse_args()


def run_scoring():
  """
  Run the complete Birdwatch algorithm, including parsing command line args,
  reading data and writing scored output; mean to be invoked from main.
  """
  args = get_args()
  data_loader = process_data.DataLoader(
    notesPath=args.notes_path,
    ratingsPath=args.ratings_path,
    noteStatusHistoryPath=args.note_status_history_path,
    userEnrollmentPath=args.enrollment,
  )
  data_loader.load()
  scoredNotes, _, _, _ = algorithm.run_algorithm(
    ratings=data_loader.ratings,
    noteStatusHistory=data_loader.notes_status_history,
    userEnrollment=data_loader.user_enrollments,
  )
  tsv_writers.DataFrameTSVWriter(path=constants.scoredNotesOutputPath).write(df=scoredNotes)
  _logger.info("Finished.")


if __name__ == "__main__":
  setup_logging()
  run_scoring()
