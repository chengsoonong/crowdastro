"""Generates training data."""

import contextlib
import sqlite3

def generate(db_path, consensus_table, model_path, features_path, output_path):
    """Generates training data.

    db_path: Path to consensus SQLite database.
    consensus_table: Name of the consensus table in the database.
    model_path: Path to CNN model JSON.
    features_path: Path to CNN model weights HDF5 file.
    output_path: Path to output HDF5 file. Training data will be output here.
    """
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
