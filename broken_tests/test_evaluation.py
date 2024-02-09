## look over wanrings if evaluation file is not dismissed

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.evaluation import *

import pytest
import numpy as np

@pytest.fixture
def example_ann_file():
    # Replace this with appropriate data for your tests
    ann_file = {(1, 'author1', 'title1'): True, (2, 'author2', 'title2'): False}
    return ann_file

@pytest.fixture
def example_prediction():
    # Replace this with appropriate data for your tests
    prediction = {(1, 'author1', 'title1'): True, (2, 'author2', 'title2'): True}
    return prediction

def test_evaluation_A(example_ann_file, example_prediction):
    precision, recall, f_measure = evaluation_A(example_ann_file, example_prediction)

    # Replace these assertions with your expected values
    assert precision == 0.5
    assert recall == 1.0
    assert f_measure == 0.6666666666666666

def test_evaluation_B(example_ann_file, example_prediction):
    precision, recall, fscore = evaluation_B(example_ann_file, example_prediction)

    # Replace these assertions with your expected values
    assert precision == 0.5
    assert recall == 1.0
    assert fscore == 0.6666666666666666

def test_evaluation_B_empty_data():
    # If the input data is empty, the result should be (0.0, 0.0, 0.0)
    ann_file = {}
    prediction = {}
    
    precision, recall, fscore = evaluation_B(ann_file, prediction)

    assert precision == 0.0
    assert recall == 0.0
    assert fscore == 0.0