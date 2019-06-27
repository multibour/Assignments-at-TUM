import pytest
import os
import json
import exe2_swissprot

    
@pytest.fixture(scope="module")
def json_data():
    relative_path = os.path.dirname(__file__)
    with open(os.path.join(relative_path, 'exe2_swissprot_test.json')) as json_file:
        json_data = json.load(json_file)
    return json_data


@pytest.fixture(scope="module")
def swissprot_parser(json_data):
    relative_path = os.path.dirname(__file__)
    return exe2_swissprot.SwissProt_Parser(
        os.path.join(relative_path, json_data['filename'])
    )


########## Tests for SwissProt ###############

# 3.2 Check whether the student retrieved the correct SP identifier
def test_sp_identifier(json_data, swissprot_parser):
    student_answer  = swissprot_parser.get_sp_identifier()
    assert student_answer, 'No SwissProt identifier provided.'
    
    correct_answer  = student_answer == json_data['identifier']
    assert correct_answer, 'Wrong SwissProt identifier provided.'
    
# 3.3 Check whether the student provided the correct sequence based on a SP XML file
def test_sp_sequence_length(json_data, swissprot_parser):
    student_answer  = swissprot_parser.get_sp_sequence_length()
    assert student_answer, 'No Sequence length provided.'
    
    
    correct_answer  = student_answer == json_data['sequence_length']
    assert correct_answer, 'Wrong sequence length provided.'
    
# 3.4 Check whether the student provided the correct organism based on a SP XML file
def test_sp_organism(json_data, swissprot_parser):
    student_answer  = swissprot_parser.get_organism()
    assert student_answer, 'No organism provided.'
    
    correct_answer  = student_answer == json_data['organism']
    assert correct_answer, 'Wrong organism provided. Check the field [organism].' 
    
# 3.5 Check whether the student provided the correct list of localizations based on a SP XML file
def test_sp_localization(json_data, swissprot_parser):
    student_answer  = swissprot_parser.get_localization()
    assert student_answer, 'No localization provided.'
    
    correct_length  = len(student_answer ) == len(json_data['localization'])
    correct_answer  = (len(set(student_answer) ^ set(json_data['localization'])) == 0)
    assert correct_length, 'Wrong number of localizations provided.'
    assert correct_answer, (
        'Wrong localization provided. Check the field [comment_subcellularlocation_location].'
        )
        
# 3.6 Check whether the student provided the correct list of PDB identifiers based on a SP XML File
def test_sp_pdb_support(json_data, swissprot_parser):
    student_answer  = swissprot_parser.get_pdb_support()
    assert student_answer, 'No PDB identifiers provided.'
    
    # PDB IDs shouldn't contain ':' ...
    correct_identifier = (''.join(student_answer).count(':') == 0) 
    assert correct_identifier, 'You did not provide a correct PDB identifier.'
    
    # Check the number of identifiers
    correct_length     =   len(student_answer ) == len(json_data['pdb_support'])
    assert correct_length, 'Wrong number of PDB IDs provided. Check the field [dbxrefs].' 
    
    # Finally, check the identifiers for equality
    correct_answer     = (len(set(student_answer) ^ set(json_data['pdb_support'])) == 0)
    assert correct_answer, 'Wrong PDB identifiers provided. Check the field [dbxrefs].'