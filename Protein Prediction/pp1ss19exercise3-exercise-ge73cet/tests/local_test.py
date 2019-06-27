import pytest
import local_alignment
import os
import json
from .matrices import MATRICES

@pytest.fixture(scope="module")
def json_data():
    relative_path = os.path.dirname(__file__)
    with open(os.path.join(relative_path, 'local_test.json')) as json_file:
        json_data = json.load(json_file)
    return json_data


@pytest.fixture(scope="module")
def null_la(json_data):
    null_la = local_alignment.LocalAlignment(
        *json_data['null']['strings'],
        json_data['null']['gap_penalty'],
        MATRICES[json_data['null']['matrix']]
    )

    return null_la

@pytest.fixture(scope="module")
def short_la(json_data):
    short_la = local_alignment.LocalAlignment(
        *json_data['short']['strings'],
        json_data['short']['gap_penalty'],
        MATRICES[json_data['short']['matrix']]
    )

    return short_la


def test_has_alignments(null_la, short_la):
    has_aligments_match = [
        not null_la.has_alignment(),
        short_la.has_alignment()
    ]
    assert all(has_aligments_match), 'Wrong value returned from has_alignments()'


def test_get_alignment_on_small_strings(json_data, short_la):
    alignment_match = short_la.get_alignment() == tuple(json_data['short']['alignment'])
    assert alignment_match, 'Wrong alignment returned'


def test_get_alignment_on_null_strings(json_data, null_la):
    alignment_match = null_la.get_alignment() == tuple(json_data['null']['alignment'])
    assert alignment_match, 'Wrong alignment returned'


def test_is_residue_aligned_on_first_string(json_data, short_la):
    [
        [string_number1, residue_number1, result1],
        [string_number2, residue_number2, result2]
    ] = json_data["short"]["residue_aligned_on_first"]

    is_residue_aligned_match = [
        short_la.is_residue_aligned(string_number1, residue_number1) is result1,
        short_la.is_residue_aligned(string_number2, residue_number2) is result2
    ]
    assert all(is_residue_aligned_match), 'Wrong value returned from is_residue_aligned'


def test_is_residue_aligned_on_second_string(json_data, short_la):
    [
        [string_number1, residue_number1, result1],
        [string_number2, residue_number2, result2]
    ] = json_data["short"]["residue_aligned_on_second"]

    is_residue_aligned_match = [
        short_la.is_residue_aligned(string_number1, residue_number1) is result1,
        short_la.is_residue_aligned(string_number2, residue_number2) is result2
    ]
    assert all(is_residue_aligned_match), 'Wrong value returned from is_residue_aligned'

