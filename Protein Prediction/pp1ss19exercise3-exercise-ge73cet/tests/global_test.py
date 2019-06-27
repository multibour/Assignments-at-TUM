import pytest
import numpy as np
import os
import json
import global_alignment
from .matrices import MATRICES

@pytest.fixture(scope="module")
def json_data():
    relative_path = os.path.dirname(__file__)
    with open(os.path.join(relative_path, 'global_test.json')) as json_file:
        json_data = json.load(json_file)
    return json_data


@pytest.fixture(scope="module")
def small_ga(json_data):
    small_ga = global_alignment.GlobalAlignment(
        *json_data['small']['strings'],
        json_data['small']['gap_penalty'],
        MATRICES[json_data['small']['matrix']]
    )

    return small_ga


@pytest.fixture(scope="module")
def large_ga(json_data):
    large_ga = global_alignment.GlobalAlignment(
        *json_data['large']['strings'],
        json_data['large']['gap_penalty'],
        MATRICES[json_data['large']['matrix']]
    )

    return large_ga


@pytest.fixture(scope="module")
def matching_ga(json_data):
    matching_ga = global_alignment.GlobalAlignment(
        *json_data['matching']['strings'],
        json_data['matching']['gap_penalty'],
        MATRICES[json_data['matching']['matrix']]
    )

    return matching_ga


@pytest.fixture(scope="module")
def mismatching_ga(json_data):
    mismatching_ga = global_alignment.GlobalAlignment(
        *json_data['mismatching']['strings'],
        json_data['mismatching']['gap_penalty'],
        MATRICES[json_data['mismatching']['matrix']]
    )

    return mismatching_ga


@pytest.fixture(scope="module")
def indel_ga(json_data):
    indel_ga = global_alignment.GlobalAlignment(
        *json_data['indel']['strings'],
        json_data['indel']['gap_penalty'],
        MATRICES[json_data['indel']['matrix']]
    )

    return indel_ga


@pytest.fixture(scope="module")
def all_changes_ga(json_data):
    all_changes_ga = global_alignment.GlobalAlignment(
        *json_data['all_changes']['strings'],
        json_data['all_changes']['gap_penalty'],
        MATRICES[json_data['all_changes']['matrix']]
    )

    return all_changes_ga


def test_get_best_score_on_small_strings(json_data, small_ga):
    best_score_match = small_ga.get_best_score() == json_data['small']['best_score']
    assert best_score_match, 'Wrong best score returned'


def test_get_best_score_on_large_strings(json_data, large_ga):
    best_score_match = large_ga.get_best_score() == json_data['large']['best_score']

    assert best_score_match, 'Wrong best score returned'


def test_get_number_of_alignments_on_small_strings(json_data, small_ga):
    number_of_alignments_match = small_ga.get_number_of_alignments() == json_data['small']['number_of_alignments']

    assert number_of_alignments_match, 'Wrong number of alignments returned'


def test_get_number_of_alignments_on_large_strings(json_data, large_ga):
    number_of_alignments_match = large_ga.get_number_of_alignments() == json_data['large']['number_of_alignments']

    assert number_of_alignments_match, 'Wrong number of alignments returned'


def test_get_alignments_on_small_strings(json_data, small_ga):
    alignments_match = set(small_ga.get_alignments()) == {tuple(x) for x in json_data['small']['alignments']}
    assert alignments_match, 'Wrong alignments returned'


def test_get_alignments_on_large_strings(json_data, large_ga):
    alignments_match = set(large_ga.get_alignments()) == {tuple(x) for x in json_data['large']['alignments']}
    assert alignments_match, 'Wrong alignments returned'


def test_score_matrix_on_matching_strings(json_data, matching_ga):
    score_matrix_match = np.array_equal(matching_ga.get_score_matrix(),
                          np.array(json_data['matching']['score_matrix']))
    assert score_matrix_match, 'Wrong score matrix'


def test_score_matrix_on_mismatching_strings(json_data, mismatching_ga):
    score_matrix_match = np.array_equal(mismatching_ga.get_score_matrix(),
                          np.array(json_data['mismatching']['score_matrix']))
    assert score_matrix_match, 'Wrong score matrix'


def test_score_matrix_on_indel_strings(json_data, indel_ga):
    score_matrix_match = np.array_equal(indel_ga.get_score_matrix(),
                          np.array(json_data['indel']['score_matrix']))
    assert score_matrix_match, 'Wrong score matrix'


def test_score_matrix_on_all_changes_strings(json_data, all_changes_ga):
    score_matrix_match = np.array_equal(all_changes_ga.get_score_matrix(),
                          np.array(json_data['all_changes']['score_matrix']))
    assert score_matrix_match, 'Wrong score matrix'


