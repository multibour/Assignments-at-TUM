import json
import pytest
import orffinder

from pathlib import Path


# Global ORF list from student code
orf_list_student = None

# Global number of ORFs found
num_orfs_student = 0

# Global ORF indices
ORF_START_IDX = 0
ORF_STOP_IDX = 1
ORF_COMPL_IDX = 3
ORF_AA_SEQ_IDX = 2


@pytest.fixture(scope="module")
def json_data():
    test_json = 'orffinder_test_2.json'
    relative_path = Path(__file__).parent

    with Path(relative_path, test_json).open('r') as json_file:
        json_data = json.load(json_file)

    return json_data


@pytest.fixture(scope="module")
def genome(json_data):
    return json_data['genome']


@pytest.fixture(scope="module")
def invalid_genome(json_data):
    return json_data['invalid_genome']


@pytest.fixture(scope="module")
def orf_list_master(json_data):
    orf_list_master = [tuple(e) for e in json_data['orf_list']]
    orf_list_master = set(orf_list_master)

    return orf_list_master


def test_orffinder_get_orfs(genome):
    try:
        # Get whatever get_orfs() returns
        global orf_list_student
        orf_list_student = orffinder.get_orfs(genome)
    except Exception:
        assert False, 'Error while testing get_orfs().'
    # Check if get_orfs() did return a list
    is_list = isinstance(orf_list_student, list)
    assert is_list, 'The get_orfs() function did not return a list.'
    # Convert list to set
    orf_list_student = set(orf_list_student)
    # Get number of ORFs
    global num_orfs_student
    num_orfs_student = len(orf_list_student)


def test_orffinder_raise_error(invalid_genome):
    raised_error = False
    # Check if get_orfs() raises a TypeError when getting invalid input
    try:
        orffinder.get_orfs(invalid_genome)
    except TypeError:
        raised_error = True
    except Exception:
        pass

    assert raised_error, 'Failed to raise TypeError for invalid genome.'


def test_orffinder_valid_orfs(genome):
    genome_length = len(genome)
    # List needs to contain at least 3 ORFs
    print(orf_list_student)
    is_min_num_orfs = num_orfs_student >= 3
    assert is_min_num_orfs, 'You need to return at least 3 ORFs.'
    # For every ORF...
    for orf in orf_list_student:
        # Test start/stop indices
        orf_start, orf_stop = orf[ORF_START_IDX], orf[ORF_STOP_IDX]
        # Check if indices are positive and within bounds
        is_in_bounds_1 = (0 <= orf_start < genome_length)
        is_in_bounds_2 = (0 <= orf_stop < genome_length)
        assert is_in_bounds_1 and is_in_bounds_2, 'ORF indices out of bounds.'
        # Test translated ORF sequence
        orf_aa_seq = orf[ORF_AA_SEQ_IDX]
        # Check if translated ORF sequence is a string
        is_string = isinstance(orf_aa_seq, str)
        assert is_string, 'Translated ORF sequence is not a string.'
        # Check if translated ORF sequence looks like DNA
        is_only_acgt = set(orf_aa_seq.upper()) == set('ACGT')
        assert not is_only_acgt, 'Translated ORF sequence looks like DNA.'
        # Check if translated ORF sequence is long enough
        is_min_length = len(orf_aa_seq) >= 34
        assert is_min_length, 'Translated ORF sequence is too short.'
        # Test if ORF is valid
        is_valid_orf = validate_orf(orf, genome)
        assert is_valid_orf, 'Invalid ORF.'


def test_orffinder_subset(orf_list_master):
    # List needs to contain at least 3 ORFs
    is_min_num_orfs = num_orfs_student >= 3
    assert is_min_num_orfs, 'You need to return at least 3 ORFs.'
    # For every ORF...
    print(len(orf_list_student))
    for orf in orf_list_student:
        print(orf)
    for orf in orf_list_student:
        # Check if ORF is in master solution
        is_in_master = orf in orf_list_master
        assert is_in_master, 'One or more ORFs are not part of the solution. {}'.format(orf)


def test_orffinder_complete_set(orf_list_master):
    # Check if test_orffinder_subset() was passed
    test_orffinder_subset(orf_list_master)
    # Both lists of ORFs should have same length
    is_correct_number = num_orfs_student == len(orf_list_master)
    assert is_correct_number, 'Incorrect number of ORFs.'


# Checks only if an ORF is valid; True by default for offline tests only!
def validate_orf(orf, genome):
    return True
