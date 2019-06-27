import pytest
import os
import json
import numpy as np
import exe2_pdb


@pytest.fixture(scope="module")
def relative_path():
    return os.path.dirname(__file__)


@pytest.fixture(scope="module")
def json_data(relative_path):
    with open(os.path.join(relative_path, 'exe2_pdb_test.json')) as json_file:
        json_data = json.load(json_file)
    return json_data


@pytest.fixture(scope="module")
def pdb_parser(relative_path, json_data):
    return exe2_pdb.PDB_Parser(
        os.path.join(relative_path, json_data['filename'])
    )


@pytest.fixture(scope="module")
def bfactors_1(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["bfactors_filename_1"]))


@pytest.fixture(scope="module")
def bfactors_2(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["bfactors_filename_2"]))


@pytest.fixture(scope="module")
def bfactors_3(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["bfactors_filename_3"]))


@pytest.fixture(scope="module")
def bfactors_4(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["bfactors_filename_4"]))


@pytest.fixture(scope="module")
def bfactors_5(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["bfactors_filename_5"]))


@pytest.fixture(scope="module")
def contact_map_1(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["contact_map_filename_1"]))


@pytest.fixture(scope="module")
def contact_map_2(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["contact_map_filename_2"]))

@pytest.fixture(scope="module")
def contact_map_3(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["contact_map_filename_3"]))


@pytest.fixture(scope="module")
def contact_map_4(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["contact_map_filename_4"]))


@pytest.fixture(scope="module")
def contact_map_5(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["contact_map_filename_5"]))


########## Tests for PDB ################
                        
# 2.8 Check whether the student provided the correct number of chains in a given PDB structure.    
def test_pdb_nChains(json_data, pdb_parser):
    student_answer  = pdb_parser.get_number_of_chains()
    assert student_answer, 'Number of chains in PDB structure missing.'
    
    correct_answer  = student_answer == json_data['number_of_chains']
    assert correct_answer,  ( 'Wrong number of chains provided. ' +
        'General structure of Bio.PDB: Structure -> Model -> Chains -> Residues -> Atoms.'
    )
    
# 2.9 Check whether the student provided the correct sequence for a given 
# chain in a given PDB structure
def test_pdb_sequence(json_data, pdb_parser):
    student_answer  = pdb_parser.get_sequence(json_data['sequence_chain'])
    assert student_answer, 'Sequence for chain in PDB structure is missing.'
    
    correct_answer  = student_answer ==  json_data['sequence']
    assert correct_answer, ( 
        'Wrong sequence for chain in PDB structure provided. Give the sequence ' +
        'as a string of one-letter amino acid codes!'
    )
        
# 2.10: Check whether the student provided the correct number of water molecules in a chain
# for a given PDB structure.
def test_number_of_water_molecules(json_data, pdb_parser):
    student_answer  = pdb_parser.get_number_of_water_molecules(json_data['water_molecules_chain'])
    assert student_answer, 'Number of water molecules for a given chain is missing.'
    
    correct_answer  = student_answer == json_data['number_of_water_molecules']
    assert correct_answer, ( 
        'Wrong number of water molecules provided. Waters are part of Bio.PDBs chains ' +
        ' and the residue name (resname) HOH.'
    )



# 2.11.1 Check whether the student provided the correct C-alpha distance for two
# residues within the same chain of a given PDB structure.
def test_ca_distance_same_chains(json_data, pdb_parser):
    student_answer  = pdb_parser.get_ca_distance(
        json_data["ca_distance_same_chains"]["chain_1"],
        json_data["ca_distance_same_chains"]["id_1"],
        json_data["ca_distance_same_chains"]["chain_2"],
        json_data["ca_distance_same_chains"]["id_2"]
    )
    assert student_answer, 'No C-alpha distance provided.'
    
    correct_answer  = student_answer == json_data["ca_distance_same_chains"]["result"]
    assert correct_answer, (
        'Wrong C-alpha distance provided. Remember rounding the distance via int().'
        )
    

# 2.11.2 Check whether the student provided the correct C-alpha distance for two residues
# between two different chains of a given PDB structure.
def test_ca_distance_different_chains(json_data, pdb_parser):
    student_answer  = pdb_parser.get_ca_distance(
        json_data["ca_distance_diff_chains"]["chain_1"],
        json_data["ca_distance_diff_chains"]["id_1"],
        json_data["ca_distance_diff_chains"]["chain_2"],
        json_data["ca_distance_diff_chains"]["id_2"]
    )
    assert student_answer, 'No C-alpha distance provided.'
    
    correct_answer = student_answer == json_data["ca_distance_diff_chains"]["result"]
    assert correct_answer,  ( 'Wrong C-alpha distance provided. ' + 
        'Take into account that two DIFFERENT chains are now considered. '+
        'Again, the resulting distance has to be rounded via int().'
        )


# 2.12.1 Check whether the student provided the correct numpy array for the normalized
# (Standard score) B-factors as np.int.
def test_bfactors_1(json_data, bfactors_1, pdb_parser):
    student_answer  = pdb_parser.get_bfactors(json_data['bfactors_chain_1'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'B-Factors are missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as np.int.'
    
    same_size = (student_answer.shape == bfactors_1.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size!'
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (bfactors_1 == student_answer) | (np.isnan(bfactors_1) & np.isnan(student_answer))
    )
    assert correct_answer, (
        'Wrong B-Factors provided. Remember providing the result as numpy.array. ' +
        'Additionally, the result has to be normalized to Standard scores (zero mean, unit variance). ' +
        'Finally, the normalized B-factors have to be rounded to int via numpy.astype( np.int ).'
    )

# 2.12.2 Check whether the student provided the correct numpy array for the normalized
# (Standard score) B-factors as np.int.

def test_bfactors_2(json_data, bfactors_2, pdb_parser):
    student_answer  = pdb_parser.get_bfactors(json_data['bfactors_chain_2'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'B-Factors are missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as np.int.'
    
    same_size = (student_answer.shape == bfactors_2.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size!'
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (bfactors_2 == student_answer) | (np.isnan(bfactors_2) & np.isnan(student_answer))
    )
    assert correct_answer, (
        'Wrong B-Factors provided. Remember providing the result as numpy.array. ' +
        'Additionally, the result has to be normalized to Standard scores (zero mean, unit variance). ' +
        'Finally, the normalized B-factors have to be rounded to int via numpy.astype( np.int ).'
    )


# 2.12.3 Check whether the student provided the correct numpy array for the normalized
# (Standard score) B-factors as np.int.

def test_bfactors_3(json_data, bfactors_3, pdb_parser):
    student_answer  = pdb_parser.get_bfactors(json_data['bfactors_chain_3'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'B-Factors are missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as np.int.'
    
    same_size = (student_answer.shape == bfactors_3.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size!'
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (bfactors_3 == student_answer) | (np.isnan(bfactors_3) & np.isnan(student_answer))
    )
    assert correct_answer, (
        'Wrong B-Factors provided. Remember providing the result as numpy.array. ' +
        'Additionally, the result has to be normalized to Standard scores (zero mean, unit variance). ' +
        'Finally, the normalized B-factors have to be rounded to int via numpy.astype( np.int ).'
    )


# 2.12.4 Check whether the student provided the correct numpy array for the normalized
# (Standard score) B-factors as np.int.

def test_bfactors_4(json_data, bfactors_4, pdb_parser):
    student_answer  = pdb_parser.get_bfactors(json_data['bfactors_chain_4'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'B-Factors are missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as np.int.'
    
    same_size = (student_answer.shape == bfactors_4.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size!'
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (bfactors_4 == student_answer) | (np.isnan(bfactors_4) & np.isnan(student_answer))
    )
    assert correct_answer, (
        'Wrong B-Factors provided. Remember providing the result as numpy.array. ' +
        'Additionally, the result has to be normalized to Standard scores (zero mean, unit variance). ' +
        'Finally, the normalized B-factors have to be rounded to int via numpy.astype( np.int ).'
    )

# 2.12.5 Check whether the student provided the correct numpy array for the normalized
# (Standard score) B-factors as np.int.
def test_bfactors_5(json_data, bfactors_5, pdb_parser):
    student_answer  = pdb_parser.get_bfactors(json_data['bfactors_chain_5'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'B-Factors are missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as np.int.'
    
    same_size = (student_answer.shape == bfactors_5.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size!'
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (bfactors_5 == student_answer) | (np.isnan(bfactors_5) & np.isnan(student_answer))
    )
    assert correct_answer, (
        'Wrong B-Factors provided. Remember providing the result as numpy.array. ' +
        'Additionally, the result has to be normalized to Standard scores (zero mean, unit variance). ' +
        'Finally, the normalized B-factors have to be rounded to int via numpy.astype( np.int ).'
   )


# 2.13.1 Check whether the student provided the correct contact map for a given chain
# in a PDB structure. Result has to be a numpy array of type np.int
def test_contact_map_1(json_data, contact_map_1, pdb_parser):
    import numpy as np
    student_answer  = pdb_parser.get_contact_map(json_data['contact_map_chain_1'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'Contact map is missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as numpy array with dtype: np.int.'
    
    same_size = (student_answer.shape == contact_map_1.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size! {} {}'.format(student_answer.shape, contact_map_1.shape)
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (contact_map_1 == student_answer) | (np.isnan(contact_map_1) & np.isnan(student_answer))
    )
    assert correct_answer, ('Contact map is not correct. ' +
        'Remember to provide the result as numpy.array, rounded with numpy.astype( np.int ).'
    )


# 2.13.2 Check whether the student provided the correct contact map for a given chain
# in a PDB structure. Result has to be a numpy array of type np.int
def test_contact_map_2(json_data, contact_map_2, pdb_parser):
    import numpy as np
    student_answer  = pdb_parser.get_contact_map(json_data['contact_map_chain_2'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'Contact map is missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as numpy array with dtype: np.int.'
    
    same_size = (student_answer.shape == contact_map_2.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size! {} {}'.format(student_answer.shape, contact_map_2.shape)
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (contact_map_2 == student_answer) | (np.isnan(contact_map_2) & np.isnan(student_answer))
    )
    assert correct_answer, ('Contact map is not correct. ' +
        'Remember to provide the result as numpy.array, rounded with numpy.astype( np.int ).'
    )


# 2.13.3 Check whether the student provided the correct contact map for a given chain
# in a PDB structure. Result has to be a numpy array of type np.int
def test_contact_map_3(json_data, contact_map_3, pdb_parser):
    import numpy as np
    student_answer  = pdb_parser.get_contact_map(json_data['contact_map_chain_3'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'Contact map is missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as numpy array with dtype: np.int.'
    
    same_size = (student_answer.shape == contact_map_3.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size! {} {}'.format(student_answer.shape, contact_map_3.shape)
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (contact_map_3 == student_answer) | (np.isnan(contact_map_3) & np.isnan(student_answer))
    )
    assert correct_answer, ('Contact map is not correct. ' +
        'Remember to provide the result as numpy.array, rounded with numpy.astype( np.int ).'
    )



# 2.13.4 Check whether the student provided the correct contact map for a given chain
# in a PDB structure. Result has to be a numpy array of type np.int
def test_contact_map_4(json_data, contact_map_4, pdb_parser):
    import numpy as np
    student_answer  = pdb_parser.get_contact_map(json_data['contact_map_chain_4'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'Contact map is missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as numpy array with dtype: np.int.'
    
    same_size = (student_answer.shape == contact_map_4.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size! {} {}'.format(student_answer.shape, contact_map_4.shape)
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (contact_map_4 == student_answer) | (np.isnan(contact_map_4) & np.isnan(student_answer))
    )
    assert correct_answer, ('Contact map is not correct. ' +
        'Remember to provide the result as numpy.array, rounded with numpy.astype( np.int ).'
    )
    

# 2.13.5 Check whether the student provided the correct contact map for a given chain
# in a PDB structure. Result has to be a numpy array of type np.int
def test_contact_map_5(json_data, contact_map_5, pdb_parser):
    import numpy as np
    student_answer  = pdb_parser.get_contact_map(json_data['contact_map_chain_5'])
    
    student_answer_exists = hasattr(student_answer, 'shape')
    assert student_answer_exists, 'Contact map is missing for a given chain in a PDB structure'
    
    student_answer_dtype  = np.issubdtype(student_answer.dtype, np.int64)
    assert student_answer_dtype, 'The answer must be provided as numpy array with dtype: np.int.'
    
    same_size = (student_answer.shape == contact_map_5.shape)
    assert same_size, 'The shape of your numpy array does not fit the expected size! {} {}'.format(student_answer.shape, contact_map_5.shape)
    
    # consider nan values when comparing student's answer to solution
    correct_answer  = np.all(
        (contact_map_5 == student_answer) | (np.isnan(contact_map_5) & np.isnan(student_answer))
    )
    assert correct_answer, ('Contact map is not correct. ' +
        'Remember to provide the result as numpy.array, rounded with numpy.astype( np.int ).'
    )
                        

def main():
    return None
    
if __name__ == '__main__':
    main()