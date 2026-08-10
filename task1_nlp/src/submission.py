"""The function that writes the submission file of Task 1.

The brief tells you to write the predictions with the function from the Colab
worksheet. If you use a different function, you lose marks. The function
save_as_csv below is that function. This module keeps the function and its
docstring without a change from
worksheets/aml_task1_nlp_worksheet.ipynb. The function writes no header row.
It uses the format of np.savetxt. It keeps the assert for the 1434 rows.

The report gives these results of the unchanged function:

  * The function writes each value in scientific notation. It writes 0 as
    0.000000000000000000e+00.
  * Thus the function writes the dummy label -1 for spam as
    -1.000000000000000000e+00. The brief tells you to give a spam row a label
    that is not 0 and not 1. This value obeys that rule.
  * The rows keep the order of the test set. The code does not change the
    order.
"""
from __future__ import annotations

import numpy as np


def save_as_csv(pred_labels, location='.'):
    """
    Save the labels out as a .csv file
    :pred_labels: numpy array of shape (no_test_labels,) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert pred_labels.shape[0] == 1434, 'wrong number of labels, should be 1434 test labels'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',')
