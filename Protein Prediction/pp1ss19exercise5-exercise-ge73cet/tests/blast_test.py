import json
import pytest
import numpy as np

from blast import BlastDb, Blast
from pathlib import Path


@pytest.fixture(scope="module")
def json_data():
    test_json = 'blast_test.json'
    relative_path = Path(__file__).parent

    with Path(relative_path, test_json).open('r') as json_file:
        json_data = json.load(json_file)

    return json_data


@pytest.fixture(scope="module")
def db_sequences(json_data):
    return json_data['db_sequences']


@pytest.fixture(scope="module")
def db_stats(json_data):
    return tuple(json_data['db_stats'])


@pytest.fixture(scope="module")
def db_seqs_for_word(json_data):
    return json_data['db_seqs_for_word']


@pytest.fixture(scope="module")
def sub_matrix(json_data):
    return np.array(json_data['sub_matrix'], dtype=np.int64)


@pytest.fixture(scope="module")
def query_seq(json_data):
    return json_data['query_seq']


@pytest.fixture(scope="module")
def query_pssm(json_data):
    return np.array(json_data['query_pssm'], dtype=np.int64)


@pytest.fixture(scope="module")
def blast_words(json_data):
    return json_data['blast_words']


@pytest.fixture(scope="module")
def blast_words_pssm(json_data):
    return json_data['blast_words_pssm']


def table_list_tuple(data):
    for key, value in data.items():
        data[key] = [tuple(x) for x in value]
    return data


@pytest.fixture(scope="module")
def blast_hsp_one_hit_1(json_data):
    return table_list_tuple(json_data['blast_hsp_one_hit_1'])


@pytest.fixture(scope="module")
def blast_hsp_one_hit_2(json_data):
    return table_list_tuple(json_data['blast_hsp_one_hit_2'])


@pytest.fixture(scope="module")
def blast_hsp_two_hit_1(json_data):
    return table_list_tuple(json_data['blast_hsp_two_hit_1'])


@pytest.fixture(scope="module")
def blast_hsp_two_hit_2(json_data):
    return table_list_tuple(json_data['blast_hsp_two_hit_2'])


@pytest.fixture(scope="module")
def blast_hsp_one_hit_pssm_1(json_data):
    return table_list_tuple(json_data['blast_hsp_one_hit_pssm_1'])


@pytest.fixture(scope="module")
def blast_hsp_two_hit_pssm_1(json_data):
    return table_list_tuple(json_data['blast_hsp_two_hit_pssm_1'])


blast = None
blast_db = None


def check_init():
    assert blast is not None, 'Blast initialization failed.'
    assert blast_db is not None, 'BlastDB initialization failed.'


def test_blast_db_get_db_stats(db_sequences, db_stats):
    global blast_db

    try:
        blast_db = BlastDb()
    except Exception:
        blast_db = None
        assert False, 'Error while creating BlastDB.'

    assert blast_db is not None, 'BlastDB initialization failed.'

    try:
        for s in db_sequences:
            blast_db.add_sequence(s)
    except Exception:
        assert False, 'Error in BlastDB.add_sequence().'

    try:
        stats = blast_db.get_db_stats()
    except Exception:
        assert False, 'Error in BlastDB.get_db_stats().'

    passed = (db_stats == stats)

    assert passed, 'Incorrect BlastDB statistics.'


def test_blast_db_get_sequences(db_seqs_for_word):
    assert blast_db is not None, 'BlastDB initialization failed.'

    try:
        seqs_for_word = blast_db.get_sequences('DEF')
    except Exception:
        assert False, 'Error in BlastDB.get_sequences().'

    passed_1 = (len(db_seqs_for_word) == len(seqs_for_word))
    passed_2 = (set(db_seqs_for_word) == set(seqs_for_word))

    passed = (passed_1 and passed_2)

    assert passed, 'Incorrect sequences returned.'


def test_blast_get_words(sub_matrix, query_seq, blast_words):
    global blast

    try:
        blast = Blast(sub_matrix)
    except Exception:
        blast = None
        assert False, 'Error while creating Blast.'

    assert blast is not None, 'Blast initialization failed.'

    try:
        words = blast.get_words(sequence=query_seq, T=13)
    except Exception:
        assert False, 'Error in Blast.get_words().'

    passed_1 = (len(blast_words) == len(words))
    passed_2 = (set(blast_words) == set(words))

    passed = (passed_1 and passed_2)

    assert passed, 'Incorrect words returned.'


def test_blast_get_words_with_pssm(query_pssm, blast_words_pssm):
    assert blast is not None, 'Blast initialization failed.'

    try:
        words = blast.get_words(pssm=query_pssm, T=11)
    except Exception:
        assert False, 'Error in Blast.get_words().'

    passed_1 = (len(blast_words_pssm) == len(words))
    passed_2 = (set(blast_words_pssm) == set(words))

    passed = (passed_1 and passed_2)

    assert passed, 'Incorrect words returned.'


def compare_blast_results(blast_results, results):
    passed_1 = (len(blast_results) == len(results))
    passed_2 = (set(blast_results) == set(results))

    passed = (passed_1 and passed_2)

    assert passed, 'Incorrect target sequences returned.'

    for target, hsp_list in results.items():
        blast_hsp_list = blast_results[target]

        passed_1 = (len(blast_hsp_list) == len(hsp_list))
        passed_2 = (set(blast_hsp_list) == set(hsp_list))

        passed = (passed_1 and passed_2)

        assert passed, 'Incorrect HSPs returned.'


def test_blast_search_one_hit_1(query_seq, blast_hsp_one_hit_1):
    check_init()

    try:
        results = blast.search_one_hit(blast_db,
                                       query=query_seq,
                                       T=13,
                                       X=5,
                                       S=30)
    except Exception:
        assert False, 'Error in Blast.search_one_hit()'

    compare_blast_results(blast_hsp_one_hit_1, results)
    pass


def test_blast_search_one_hit_2(query_seq, blast_hsp_one_hit_2):
    check_init()

    try:
        results = blast.search_one_hit(blast_db,
                                       query=query_seq,
                                       T=13,
                                       X=7,
                                       S=40)
    except Exception:
        assert False, 'Error in Blast.search_one_hit()'

    compare_blast_results(blast_hsp_one_hit_2, results)


def test_blast_search_two_hit_1(query_seq, blast_hsp_two_hit_1):
    check_init()

    try:
        results = blast.search_two_hit(blast_db,
                                       query=query_seq,
                                       T=11,
                                       X=5,
                                       S=30,
                                       A=40)
    except Exception:
        assert False, 'Error in Blast.search_two_hit()'

    r = sorted(((k, v) for k, v in results.items()), key=lambda e: e[0])
    o = sorted(((k, v) for k, v in blast_hsp_two_hit_1.items()), key=lambda e: e[0])

    t = list()
    c = list()
    extras = list()
    for k, v in r:
        if k not in blast_hsp_two_hit_1:
            extras.append((k, v))
            continue
        t.append((k, results[k], blast_hsp_two_hit_1[k]))
        if sorted(results[k]) != sorted(blast_hsp_two_hit_1[k]):
            c.append((k, results[k], blast_hsp_two_hit_1[k]))
    del r, o

    missed = set(blast_hsp_two_hit_1.keys()) - set(results.keys())

    compare_blast_results(blast_hsp_two_hit_1, results)


def test_blast_search_two_hit_2(query_seq, blast_hsp_two_hit_2):
    check_init()

    try:
        results = blast.search_two_hit(blast_db,
                                       query=query_seq,
                                       T=11,
                                       X=7,
                                       S=40,
                                       A=40)
    except Exception:
        assert False, 'Error in Blast.search_two_hit()'

    compare_blast_results(blast_hsp_two_hit_2, results)


def test_blast_search_one_hit_with_pssm_1(query_pssm, blast_hsp_one_hit_pssm_1):
    check_init()

    try:
        results = blast.search_one_hit(blast_db,
                                       pssm=query_pssm,
                                       T=13,
                                       X=5,
                                       S=30)
    except Exception:
        assert False, 'Error in Blast.search_one_hit()'

    compare_blast_results(blast_hsp_one_hit_pssm_1, results)


def test_blast_search_two_hit_with_pssm_1(query_pssm, blast_hsp_two_hit_pssm_1):
    check_init()

    try:
        results = blast.search_two_hit(blast_db,
                                       pssm=query_pssm,
                                       T=11,
                                       X=5,
                                       S=30,
                                       A=40)
    except Exception:
        assert False, 'Error in Blast.search_two_hit()'

    compare_blast_results(blast_hsp_two_hit_pssm_1, results)
