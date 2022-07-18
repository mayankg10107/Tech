"""
Unit testing for training File
"""
import os
import unittest
from train import training


class TestTrain(unittest.TestCase):
    """
    Unit test class
    """

    def test_train(self):
        # Get current directory, parent directory
        curr_path = os.getcwd()
        par_path = os.path.abspath(os.path.join(curr_path, os.pardir))

        # Test file path
        path = par_path + "\\Tech\\\\data\\unittest\\regression_train.csv"
        _, _, mean_sq_error = training(path)
        self.assertTrue(isinstance(mean_sq_error, float))

        path = par_path + "\\Tech\\\\data\\unittest\\reg_train.csv"
        error = "File not found or Error reading .csv file"
        self.assertEqual(training(path), error)

        path = par_path + "\\Tech\\\\data\\unittest\\regression_train_height_delete.csv"
        error = "Please check the contents of the file"
        self.assertEqual(training(path), error)

        path = par_path + "\\Tech\\\\data\\unittest\\regression_train_str_value.csv"
        error = "Error fitting the model"
        self.assertEqual(training(path), error)


if __name__ == "__main__":
    unittest.main()
