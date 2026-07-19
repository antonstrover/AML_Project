"""Task 1 submission writer.

The brief states that predictions **must** be exported with the function
supplied in the Colab worksheet, "to avoid losing marks". ``save_as_csv`` below
is that function, copied verbatim (docstring included) from
``worksheets/aml_task1_nlp_worksheet.ipynb`` -- no header row, ``np.savetxt``
formatting, and the 1434-row assertion retained.

Consequences of using it verbatim, which the report states explicitly:
  * values are written in scientific notation (``0`` -> ``0.000000000000000000e+00``);
  * the spam dummy label ``-1`` therefore appears as ``-1.000000000000000000e+00``,
    which satisfies the brief's requirement that spam rows carry a label that is
    neither 0 nor 1;
  * row order is the test-set order and is never permuted.
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
