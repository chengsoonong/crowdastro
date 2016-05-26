"""Tests for data.py.

Matthew Alger
The Australian National University
2016
"""

#/usr/bin/env python3

import unittest

from . import data

N_SUBJECTS = 177461
N_SUBJECTS_ATLAS_CDFS = 2460
N_SUBJECTS_ATLAS_ELAIS_S1 = 0  # There are no ELAIS-S1 Zooniverse subjects.


class TestGetAllSubjects(unittest.TestCase):

    def test_default(self):
        """Returns correct number of subjects with no arguments."""
        # Testing based on returned count is somewhat fragile but I can't think
        # of a better way.
        self.assertEqual(data.get_all_subjects().count(), N_SUBJECTS)

    def test_atlas(self):
        """Returns correct number of subjects in ATLAS survey."""
        self.assertEqual(data.get_all_subjects(survey='atlas').count(),
                         N_SUBJECTS_ATLAS_CDFS + N_SUBJECTS_ATLAS_ELAIS_S1)

    def test_atlas_cdfs(self):
        """Returns correct number of subjects in ATLAS survey, CDFS field."""
        self.assertEqual(
                data.get_all_subjects(survey='atlas', field='cdfs').count(),
                N_SUBJECTS_ATLAS_CDFS)

    def test_atlas_elais_s1(self):
        """Returns correct number of subjects in ATLAS survey, ELAIS-S1 field.
        """
        self.assertEqual(
                data.get_all_subjects(survey='atlas', field='elais-s1').count(),
                N_SUBJECTS_ATLAS_ELAIS_S1)


if __name__ == '__main__':
    unittest.main()
