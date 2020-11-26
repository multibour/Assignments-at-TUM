# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:27:33 2018

@author: Michael
"""

import pytest
import json
import pickle

import os
import numpy as np


from tests import basic_security, silence_printing
from ann import MLP_ANN as ann_student


############ HELPER FUNCTIONS ##################
@pytest.fixture(scope="module")
def relative_path():
    return os.path.dirname(__file__)

@pytest.fixture(scope="module")
def json_data(relative_path):
    with open(os.path.join(relative_path, 'test.json')) as json_file:
        json_data = json.load(json_file)
    return json_data

@pytest.fixture(scope="module")
def seed(json_data):
    return json_data['parameters']['seed']

@pytest.fixture(scope="module")
def raw_x(relative_path, json_data):
    with open( os.path.join(relative_path, json_data["raw_x"]), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture(scope="module")
def raw_y(relative_path, json_data):
    with open( os.path.join(relative_path, json_data["raw_y"]), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture(scope="module")
def one_hot_lookup(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["one_hot_lookup"]))

@pytest.fixture(scope="module")
def one_hot_inputs(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["one_hot_inputs"]))

@pytest.fixture(scope="module")
def sliding_window_inputs(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["sliding_window_inputs"]))

@pytest.fixture(scope="module")
def y_classes(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["y_classes"]))

@pytest.fixture(scope="module")
def y_mask(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["y_mask"]))

@pytest.fixture(scope="module")
def filtered_x(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["filtered_x"]))

@pytest.fixture(scope="module")
def filtered_y(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["filtered_y"]))

@pytest.fixture(scope="module")
def train_test_splits(relative_path, json_data):
    with open( os.path.join(relative_path, json_data["train_test_splits"]), 'rb') as f:
        data = pickle.load(f)
    return data

@pytest.fixture(scope="module")
def n_samples(json_data):
    return json_data["n_samples"]

@pytest.fixture(scope="module")
def n_masked(json_data):
    return json_data["n_masked"]

@pytest.fixture(scope="module")
def n_pos(json_data):
    return json_data["n_pos"]

@pytest.fixture(scope="module")
def n_neg(json_data):
    return json_data["n_neg"]

@pytest.fixture(scope="module")
def softmax_test_size(json_data):
    return json_data["softmax_test"]["softmax_test_set_size"]

@pytest.fixture(scope="module")
def softmax_result(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["softmax_test"]["softmax_result"]))

@pytest.fixture(scope="module")
def n_ce_samples(json_data):
    return json_data["cross_entropy"]["n_samples"]

@pytest.fixture(scope="module")
def n_ce_classes(json_data):
    return json_data["cross_entropy"]["n_classes"]

@pytest.fixture(scope="module")
def cross_entropy_result(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["cross_entropy"]["cross_entropy_result"]))

@pytest.fixture(scope="module")
def delta_ce_result(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["cross_entropy"]["delta_ce_result"]))

@pytest.fixture(scope="module")
def window_size(json_data):
    return (2 * json_data["parameters"]["half_window"]) + 1

@pytest.fixture(scope="module")
def alphabet_size(json_data):
    return json_data["parameters"]["alphabet_size"]

@pytest.fixture(scope="module")
def predictions(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["predictions"]))

@pytest.fixture(scope="module")
def trained_weights(relative_path, json_data):
    return np.load(os.path.join(relative_path, json_data["trained_weights"]))

@pytest.fixture(scope="module")
def performance(relative_path, json_data):
    with open( os.path.join(relative_path, json_data["performance"]), 'rb') as f:
        data = pickle.load(f)
    return data


############ INIT STUDENT PERCEPTRON ######################
@pytest.fixture(scope="module")
def student_ann(relative_path, json_data):
    # learning rate, number of epochs and random seed for single perceptrons
    LEARNING_RATE = json_data['parameters']['learning_rate']
    NEPOCHS       = json_data['parameters']['nepochs']
    SEED          = json_data['parameters']['seed']
    HALF_WINDOW   = json_data['parameters']['half_window']
    NHIDDEN       = json_data['parameters']['nhidden']
    TEST_PERC     = json_data['parameters']['test_percentage']

    return ann_student( nEpochs=NEPOCHS, learning_rate=LEARNING_RATE, 
                        half_window_size=HALF_WINDOW, n_hidden=NHIDDEN, 
                        test_percentage=TEST_PERC, seed=SEED  )





############################ UTILITY FUNCTIONS ################################

def _cmp_dict_keys( dict1, dict2 ):
    return dict1.keys() == dict2.keys()

def _cmp_dict_vals( dict1, dict2 ):
    return dict1 == dict2

def _cmp_dict_vals_numpy( dict1, dict2 ):
    return all( np.array_equal( value, dict2[ key ] ) 
                                  for key, value in dict1.items() )
    
def _cmp_numpy( np1, np2 ):
    # single element vs empty array would evaluate to True
    if np1.size == 0 or np2.size == 0: 
        return False
    try:
        return np.allclose( np1, np2 )
    except ValueError: # shape mismatch, no broadcasting possible
        return False
    except TypeError: # no numpy array provided
        return False
    
    
    
############################ TEST PREPROCESSING ###############################
        
def test_raw_data( student_ann, raw_x, raw_y, basic_security ):
    try:
        stud_in_raw, stud_out_raw = student_ann._get_raw_data()
    except TypeError: # if only one variable was returned instead of a tuple
        assert False, 'You returned the wrong number of attributes.'
        
    test_none = stud_in_raw is not None and stud_out_raw is not None
    assert test_none, ( 
            'You returned None while reading in raw inputs & outputs.' )
    
    test_in_keys  = _cmp_dict_keys( raw_x, stud_in_raw  )
    test_out_keys = _cmp_dict_keys( raw_y, stud_out_raw )
    assert test_in_keys, (
            'Your raw input dictionary did not contain the correct protein ' + 
            'identifiers as keys. Double check the desc. in the template.' )
    assert test_out_keys, (
            'Your raw output dictionary did not contain the correct protein ' + 
            'identifiers as keys. Double check the desc. in the template.' )

    test_in_vals  = _cmp_dict_vals( raw_x,  stud_in_raw  )
    test_out_vals = _cmp_dict_vals( raw_y, stud_out_raw )
    assert test_in_vals, (
            'Your raw input dictionary did not contain the correct protein ' + 
            'sequences as values.' )
    assert test_out_vals, (
            'Your raw output dictionary did not contain the correct labels ' + 
            'as values.' )

    

def test_one_hot_encoding( student_ann, one_hot_lookup, basic_security ):

    stud_one_hot = student_ann._get_one_hot_encoding()
    
    test_none = stud_one_hot is not None
    assert test_none, ( 
            'You returned None while creating the one-hot-encoding for the inputs.' )
    
    test_one_hot_keys = _cmp_dict_keys( one_hot_lookup, stud_one_hot )
    assert test_one_hot_keys, (
            'Your one-hot-encoding dictionary does not contain the ALPHABET as keys. '
            'Check the static class variables for further information on ALPHABET.' )
    
    test_one_hot_vals = _cmp_dict_vals_numpy( one_hot_lookup, stud_one_hot )
    assert test_one_hot_vals, (
            'Your one-hot-encoding dictionary does not contain the correct ' + 
            'one-hot-encoding as values. Double check the description in the template.' )
    
    
def test_input_encoding( student_ann, one_hot_inputs, basic_security ):

    stud_in_one_hot = student_ann._get_one_hot_inputs()
    
    test_none = stud_in_one_hot is not None
    assert test_none, ( 
            'You returned None while encoding the input sequence as one-hot-vectors. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test.' )
    
    test_one_hot_inputs = _cmp_dict_keys( one_hot_inputs, stud_in_one_hot )
    assert test_one_hot_inputs, (
            'Your dictionary summarizing the one-hot-encoded inputs did not contain ' +
            'the correct protein identifiers as keys. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test. ' + 
            'Also check the example identifier in the description of the function.' )
    
    test_one_hot_vals = _cmp_dict_vals_numpy( one_hot_inputs, stud_in_one_hot )
    assert test_one_hot_vals, (
            'Your dictionary summarizing the one-hot-encoded inputs did not contain ' +
            'the correct one-hot-encodings as values. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test. ' + 
            'Also check whether the shape of your returned array fits the ' + 
            'expected shape described in the template.' )
    
    
def test_sliding_window_view( student_ann, sliding_window_inputs, basic_security ):
    
    stud_in_window = student_ann._get_sliding_window_view()
    
    test_none = stud_in_window is not None
    assert test_none, ( 
            'You returned None while creating the sliding window view for the ' +
            'one-hot-encoded input sequences. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test.' )
    
    test_window_inputs = _cmp_dict_keys( sliding_window_inputs, stud_in_window )
    assert test_window_inputs, (
            'Your dictionary summarizing the sliding-window view on the one-hot-encoded ' +  
            'inputs did not contain the correct protein identifiers as keys. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test. ' + 
            'Also check the example identifier in the template.' )
    
    test_one_hot_vals = _cmp_dict_vals_numpy( sliding_window_inputs, stud_in_window )
    assert test_one_hot_vals, (
            'Your dictionary summarizing the sliding-window view on the one-hot-encoded ' + 
            'inputs did not contain the correct values. ' + 
            'Double check whether your one-hot-encoding dictionary passes all test. ' + 
            'Also check whether the shape of your returned array fits the ' + 
            'expected shape described in the template.' )

    
    
def test_output_encoding( student_ann, y_classes, y_mask, basic_security ):
    try:
        stud_out, stud_mask = student_ann._get_integer_outputs()
    except TypeError: # if only one variable was returned instead of a tuple
        assert False, 'You returned the wrong number of attributes.'
        
    test_none = stud_out is not None and stud_mask is not None
    assert test_none, (
            'You returned None for the encoded outputs or the mask.' )
    
    test_output_keys = _cmp_dict_keys( y_classes, stud_out )
    assert test_output_keys, (
            'Your dictionary summarizing the encoded outputs did not contain the ' +
            'correct protein identifiers as keys. Double check the example ' +
            'identifier in the template. ' )
    
    test_output_vals = _cmp_dict_vals_numpy( y_classes, stud_out )
    assert test_output_vals, (
            'Your dictionary summarizing the encoded outputs did not contain the ' +
            'correct output encoding (values). Double check whether the expected ' +
            'shape fits the shape of your numpy array.' )
    
    test_mask_keys = _cmp_dict_keys( y_mask, stud_mask )
    assert test_mask_keys, (
            'Your dictionary summarizing the unresolved or masked out residues ' + 
            'did not contain the correct protein identifiers as keys. ' + 
            'Double check the example identifier in the template. ' )
    
    test_output_vals = _cmp_dict_vals_numpy( y_mask, stud_mask )
    assert test_output_vals, (
            'Your dictionary summarizing the unresolved or masked out residues ' +
            'did not contain the correct output encoding (values). Double check ' + 
            'whether the expected shape fits the shape of your numpy array.' )
    

def test_remove_unresolved( student_ann, filtered_x, filtered_y, basic_security ):
    try:
        stud_in, stud_out = student_ann._remove_unresolved()
    except TypeError: # if only one variable was returned instead of a tuple
        assert False, 'You returned the wrong number of attributes.'
        
    test_none = stud_in is not None and stud_out is not None
    assert test_none, (
            'You returned None instead of the modified inputs and outputs.' )
    
    test_input_keys = _cmp_dict_keys( filtered_x, stud_in )
    assert test_input_keys, (
            'Your dictionary summarizing the modified inputs did not contain ' +
            'the correct protein identifiers as keys. '                         +
            'Double check the description in the template' )
    
    test_input_vals = _cmp_dict_vals_numpy( filtered_x, stud_in )
    assert test_input_vals, (
            'Your dictionary summarizing the modified inputs did not contain the ' +
            'correct output encoding (values). Double check whether the expected '  +
            'shape fits the shape of your numpy array. Also double check whether '  +
            'you have removed all unresolved residues (marked as U in the labels).' )
    
    test_output_keys = _cmp_dict_keys( filtered_y, stud_out )
    assert test_output_keys, (
            'Your dictionary summarizing the modified outputs did not contain ' + 
            'the correct protein identifiers as keys. ' + 
            'Double check the example identifier in the template. ' )
    
    test_output_vals = _cmp_dict_vals_numpy( filtered_y, stud_out )
    assert test_output_vals, (
            'Your dictionary summarizing the modified outputs ' +
            'did not contain the correct output encoding (values). Double check ' + 
            'whether you have removed all unresolved residues ' + 
            '(marked as U in the labels).' )
    
    
    
def test_splits( student_ann, train_test_splits, basic_security ):

    stud_splits= student_ann._get_train_test_split()
    split_names = [ 'train_inputs', 'train_outputs' ,'test_inputs' ,'test_outputs' ]
    
    test_none = stud_splits is not None
    assert test_none, (
            'You returned None instead of the training and test set splits.')
    
    test_none = all( data_set is not None for data_set in stud_splits)
    assert test_none, (
            'You returned None for one of the training and test set splits.')
    
    test_n_sets = len( train_test_splits ) == len( stud_splits )
    assert test_n_sets, ( 
            'You did not return the correct number of sets. Re-read template.' )
    
    for index, data_set in enumerate( split_names ):
        sol_set     = train_test_splits[  index ]
        stud_set    = stud_splits[ index ]
        test_keys   = _cmp_dict_keys(       sol_set, stud_set )
        test_values = _cmp_dict_vals_numpy( sol_set, stud_set ) 
        assert test_keys, (
                '''
                You did not return the correct splits with respect to the keys
                (protein identifiers). Double check if you passed previous test 
                to ensure that your keys are formatted correctly. Also double 
                check the order of the sets which you return (see template for 
                a description). The error occured in {}.
                '''.format( data_set ) )
        
        assert test_values, (
                '''
                You did not return the correct values (sliding window inputs and 
                integer-encoded outputs) after splitting your data in train & 
                test set. Do not modify the values here. Simply split the data set
                on a protein level into train & test set. The error occured in {}.
                '''.format( data_set ) )
                
       
############################ TEST DATA ANALYSIS ###############################
                
def test_num_samples(student_ann, n_samples, basic_security ):
    stud_nSamples = student_ann.get_num_samples()
    
    test_none = stud_nSamples is not None
    assert test_none, 'You returned None instead of the number of samples.'
    
    test_equality = n_samples == stud_nSamples
    assert test_equality, ( 
            'You returned the wrong number of samples. Remember removing ' + 
            'unresolved residues.')
    
def test_masked_samples( student_ann, n_masked, basic_security):
    stud_nMasked = student_ann.get_num_masked_out()
    
    test_none = stud_nMasked is not None
    assert test_none, ( 'You returned None instead of the number of masked-out ' + 
                           '(unresolved) residues.' )
    
    test_equality = n_masked == stud_nMasked
    assert test_equality, ( 
            'You returned the wrong number of masked-out (unresolved) residues.' )
    
    
def test_pos_neg_samples( student_ann, n_pos, n_neg, basic_security ):
    try:
        stud_pos, stud_neg = student_ann.get_num_pos_and_neg()
    except TypeError: # if only one variable was returned instead of a tuple
        assert False, 'You returned the wrong number of attributes.'
    
    test_none = stud_pos is not None and stud_neg is not None
    assert test_none, ( 'You returned None instead of the number of positive and ' +
                           'negative samples.' )
    
    test_equality_pos = n_pos == stud_pos
    assert test_equality_pos, ( 
            'You returned the wrong number of positive samples.' )
    test_equality_neg = n_neg == stud_neg
    assert test_equality_neg, ( 
            'You returned the wrong number of negative samples.' )

############################## TEST NETWORK ###################################
    

def test_softmax( student_ann, softmax_test_size, softmax_result, basic_security ):
    predictions  = np.arange(softmax_test_size**2).reshape(softmax_test_size, softmax_test_size)
    stud_softmax = student_ann._stable_softmax( predictions )
    
    test_none = stud_softmax is not None
    assert test_none, 'You have returned None instead of the correct softmax.'
    
    test_equality = _cmp_numpy( softmax_result, stud_softmax )
    assert test_equality, ( 
            '''
                Your softmax implementation did not return the correct values.
                Keep the following in mind:
                    - Numerical stable softmax should be implemented:
                        exp( predictions - max(predictions) )
                    - Vectorized version should be implemented taking several
                        predictions into account ( One prediction is one row; 
                        columns are classes )
            ''')
                        
def test_cross_entropy( student_ann, n_ce_samples, n_ce_classes, cross_entropy_result, seed, basic_security ):
    rnd_generator = np.random.RandomState( seed )
    predictions   = rnd_generator.rand(    n_ce_samples, n_ce_classes )
    ground_truth  = rnd_generator.randint( n_ce_classes, size=(n_ce_samples) )

    stud_entropy = student_ann._cross_entropy(  predictions, ground_truth )
    
    test_none = stud_entropy is not None
    assert test_none, 'You have returned None instead of the correct cross-entropy.'
    
    test_equality = _cmp_numpy( cross_entropy_result, stud_entropy )
    assert test_equality, ( 
            '''
                Your cross-entropy implementation did not return the correct values.
                Keep the following in mind:
                    - Vectorized version should be implemented taking several
                        predictions into account ( One prediction is one row; 
                        columns are classes ).
                    - Y_pred should have this shape: ( num_samples, num_classes )
                    - Y should have this shape:      ( num_samples, )
                    - Y just contains the class labels in [0,1] which are used 
                        as indices
            ''')


def test_delta_cross_entropy( student_ann, n_ce_samples, n_ce_classes, delta_ce_result, seed, basic_security ):
    rnd_generator = np.random.RandomState( seed )
    predictions   = rnd_generator.rand( n_ce_samples, n_ce_classes )
    ground_truth  = rnd_generator.randint( n_ce_classes, size=(n_ce_samples) )

    stud_delta_e = student_ann._delta_cross_entropy(  predictions, ground_truth )
    
    test_none = stud_delta_e is not None
    assert test_none, 'You have returned None instead of the correct delta cross-entropy.'
    
    test_equality = _cmp_numpy( delta_ce_result, stud_delta_e )
    assert test_equality, ( 
            '''
                Your delta cross-entropy implementation did not return the correct values.
                Keep the following in mind:
                    - Vectorized version should be implemented taking several
                        predictions into account ( One prediction is one row; 
                        columns are classes ).
                    - Y_pred should have this shape: ( num_samples, num_classes )
                    - Y should have this shape:      ( num_samples, )
                    - Y just contains the class labels in [0,1] which are used 
                        as indices
            ''' )
                    

FORWARD_PASSSED=False
def test_forward_pass( student_ann, predictions, window_size, alphabet_size, seed, basic_security ):
    rnd_generator = np.random.RandomState( seed )
    x = rnd_generator.randint( 2, size=( 5, window_size * alphabet_size ))
    y = rnd_generator.randint( 2, size=( 5, ))
    
    stud_pred = student_ann._predict(  x, y, is_training=False )
    
    test_none = stud_pred is not None
    assert test_none, 'You returned None instead of the predicted labels.'
    
    test_equality = _cmp_numpy( predictions, stud_pred )
    assert test_equality, (
            '''
                Your implementation of the forward pass is not correct.
                Double check whether you already passed the test for softmax
                as it is required here.
            ''')
    
    if test_equality: # not sure if necessary
        global FORWARD_PASSSED
        FORWARD_PASSSED = True

    
def test_forward_pass2():
    assert FORWARD_PASSSED, 'Failed repeated forward pass test to assign more points.'


BACKWARD_PASSED = False
def test_backward_pass( student_ann, trained_weights, basic_security ):

    stud_pred = student_ann.train_network()
    
    test_none = stud_pred is not None
    assert test_none, 'You returned None instead of the weights in the second layer.'
    
    test_equality = _cmp_numpy( trained_weights, stud_pred )
    assert test_equality, (
            '''
                Your implementation of the backward pass is not correct.
                Double check whether you already passed the test for the forward
                pass, stable softmax and delta cross-entropy.
                If you already passed all these tests, double check whether you
                take the non-linearity during the backward pass into account.
            ''')
    
    if test_equality:
        global BACKWARD_PASSED
        BACKWARD_PASSED = True

def test_backward_pass2( basic_security ):
    assert BACKWARD_PASSED, 'Failed repeated backward pass test to assign more points.'
    
def test_backward_pass3( basic_security ):
    assert BACKWARD_PASSED, 'Failed repeated backward pass test to assign more points.'
    
def test_backward_pass4( basic_security ):
    assert BACKWARD_PASSED, 'Failed repeated backward pass test to assign more points.'
    
def test_performance_assessment( student_ann, performance, basic_security ):
    # test only performance on test set. After calling backward pass in pervious
    # task, weights should be trained by now
    stud_perf = student_ann._test_performance(  eval_train_set=False )
    
    test_none = stud_perf is not None
    assert test_none, 'You returned None instead of performance values.'
    
    sol_acc,  sol_loss,  sol_confmat  = performance
    stud_acc, stud_loss, stud_confmat = stud_perf
    
    test_none = ( stud_acc is not None and 
                  stud_loss is not None and 
                  stud_confmat is not None )
    assert test_none, 'Either your accuracy, loss or confusion matrix is None.'
    
    test_acc = np.isclose( sol_acc, stud_acc )
    assert test_acc, ( 
            'Your accuracy is not correct. Sum all correct samples and divide by ' +
            'the number of all samples in a given set (training or testing). ' +
            'You need to pass all previous test to pass this test.'
            )
    
    test_loss = np.isclose( sol_loss, stud_loss )
    assert test_loss,  ( 
            'Your average loss is not correct. Sum loss over all samples and ' +
            'calculate the average for those samples. ' +
            'You need to pass all previous test to pass this test.'
            )
    
    test_confmat = _cmp_numpy( sol_confmat, stud_confmat )
    assert test_confmat,  ( 
            'Your confusion matrix is not correct. Remember that true labels are ' +
            'given in rows and predictions in columns (see template).' +
            'You need to pass all previous test to pass this test.'
            )

