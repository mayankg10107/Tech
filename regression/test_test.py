"""
Unit testing for testing File
"""
import os
import unittest
from test import testing


class TestTest(unittest.TestCase):
    """
    Unit test class
    """

    def test_train(self):
        # Get current directory, parent directory
        curr_path = os.getcwd()
        par_path = os.path.abspath(os.path.join(curr_path, os.pardir))

        # Test file path
        path = par_path + "\\Tech\\\\data\\unittest\\regression_test.csv"
        mean_sq_error = testing(path)
        self.assertTrue(isinstance(mean_sq_error, float))

        path = par_path + "\\Tech\\\\data\\unittest\\reg_test.csv"
        error = "File not found or Error reading .csv file"
        self.assertEqual(testing(path), error)

        path = par_path + "\\Tech\\\\data\\unittest\\regression_train_height_delete.csv"
        error = "Please check the contents of the file"
        self.assertEqual(testing(path), error)


if __name__ == "__main__":
    unittest.main()
