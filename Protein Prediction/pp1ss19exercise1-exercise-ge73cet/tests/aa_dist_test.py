import pytest
import aa_dist
from math import isclose
import os 

absolute_frequencies = {
    'A': 71,
    'R': 62,
    'N': 46,
    'D': 65,
    'C': 18,
    'E': 64,
    'Q': 31,
    'G': 80,
    'H': 53,
    'I': 66,
    'L': 107,
    'K': 85,
    'M': 35,
    'F': 54,
    'P': 44,
    'S': 94,
    'T': 60,
    'W': 13,
    'Y': 45,
    'V': 66,
}

number_of_sequences = 5
average_length = 231.8
total_length = 1159


@pytest.fixture(scope='session')
def setup():
    relative_path = os.path.dirname(__file__)
    dist =  aa_dist.AADist(os.path.join(relative_path,"tests.fasta"))
    return dist

def test_counts(setup):
    dist = setup
    numbers_match = dist.get_counts() == number_of_sequences
    assert numbers_match, "Wrong number of sequences."


def test_average_length(setup):
    dist = setup
    average_len_match = dist.get_average_length() == average_length
    assert average_len_match, "Wrong average length."


def test_abs_freq(setup):
    dist = setup
    abs_freq_dict = dist.get_abs_frequencies()
    abs_freqs_match = [
        absolute_frequencies[key] == abs_freq_dict[key] for key in absolute_frequencies
    ]
    correct_test = all(abs_freqs_match)
    assert correct_test, "Some absolute frequencies are wrong."


def test_pruning(setup):
    dist = setup
    abs_freq_dict = dist.get_abs_frequencies()
    pruning_check = "*" not in abs_freq_dict
    assert pruning_check, "Found a symbol that should have been pruned."


def test_av_freq(setup):
    dist = setup
    av_freq_dict = dist.get_av_frequencies()
    av_freqs_match = [
        isclose(
            av_freq_dict[key], absolute_frequencies[key] / total_length, rel_tol=1e-4
        )
        for key in absolute_frequencies
    ]
    correct_test = all(av_freqs_match)
    assert correct_test, "Some average frequencies are wrong."
