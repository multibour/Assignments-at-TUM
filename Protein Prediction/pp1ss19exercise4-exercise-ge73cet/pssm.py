import numpy as np

"""
ATTENTION: Use the following dictionaries to get the correct index for each
           amino acid when accessing any type of matrix or array provided as
           parameters. Further, use those indices when generating or returning
           any matrices or arrays. Failure to do so will most likely result in
           not passing the tests.
EXAMPLE: To access the substitution frequency from alanine 'A' to proline 'P'
         in the bg_matrix use bg_matrix[AA_TO_INT['A'], AA_TO_INT['P']].
"""
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'
AA_TO_INT = {aa: index for index, aa in enumerate(ALPHABET)}
INT_TO_AA = {index: aa for index, aa in enumerate(ALPHABET)}
GAP_INDEX = AA_TO_INT['-']
ALPHABET = set(ALPHABET)
AA_TYPES_COUNT = len(ALPHABET) - 1  # number of different amino acids = 20


class MSA:

    def __init__(self, sequences):
        """
        Initialize the MSA class with the provided list of sequences. Check the
        sequences for correctness. Pre-calculate any statistics you seem fit.

        :param sequences: List containing the MSA sequences.
        """
        if len(sequences) == 0\
                or any(len(sequences[0]) != len(seq) for seq in sequences)\
                or any(not set(seq).issubset(ALPHABET) for seq in sequences):
            raise TypeError('Invalid sequences.')

        self.sequences = sequences
        self.sequences_matrix = np.array([list(seq) for seq in self.sequences])
        self.sequence_length = len(self.sequences[0])

    def get_pssm(self, *, bg_matrix=None, beta=10, use_sequence_weights=False,
                 redistribute_gaps=False, add_pseudocounts=False):
        """
        Return a PSSM for the underlying MSA. Use the appropriate refinements 
        according to the parameters. If no bg_matrix is specified, use uniform 
        background frequencies.
        Every row in the resulting PSSM corresponds to a non-gap position in 
        the primary sequence of the MSA (i.e. the first one).
        Every column in the PSSM corresponds to one of the 20 amino acids.
        Values that would be -inf must be replaced by -20 in the final PSSM.
        Before casting to dtype=numpy.int64, round all values to the nearest
        integer (do not just FLOOR all values).

        :param bg_matrix: Amino acid pair frequencies as numpy array (20, 20).
                          Access the matrix using the indices from AA_TO_INT.
        :param beta: Beta value (float) used to weight the pseudocounts 
                     against the observed amino acids in the MSA.
        :param use_sequence_weights: Calculate and apply sequence weights.
        :param redistribute_gaps: Redistribute the gaps according to the 
                                  background frequencies.
        :param add_pseudocounts: Calculate and add pseudocounts according 
                                 to the background frequencies.

        :return: PSSM as numpy array of shape (L x 20, dtype=numpy.int64).
                 L = ungapped length of the primary sequence.
        """
        # initial checks
        if bg_matrix is None:  # set the substitution matrix uniformly
            bg_matrix = np.full((AA_TYPES_COUNT, AA_TYPES_COUNT), 1 / (AA_TYPES_COUNT**2), dtype=np.float64)
        else:  # convert list of lists into numpy array
            bg_matrix = np.array(bg_matrix, dtype=np.float64)

        # calculate background frequencies
        background_frequencies = np.sum(bg_matrix, axis=1)[np.newaxis, :]

        # initialisation
        pssm = np.zeros((self.sequence_length, AA_TYPES_COUNT))
        gaps = np.zeros(self.sequence_length)

        # calculate sequence weights
        if use_sequence_weights is True:
            sequence_weights = self.get_sequence_weights()
        else:
            sequence_weights = np.ones(len(self.sequences))  # set default weight of 1

        # fill PSSM and gaps vector
        for column_index, column in enumerate(self.sequences_matrix.T):
            for seq_index, aa in enumerate(column):
                if aa == '-':  # if a gap
                    gaps[column_index] += sequence_weights[seq_index]
                else:  # if an amino acid
                    pssm[column_index, AA_TO_INT[aa]] += sequence_weights[seq_index]

        # redistribute gaps
        if redistribute_gaps is True:
            pssm += gaps[:, np.newaxis] @ background_frequencies

        # add pseudocounts
        if add_pseudocounts is True:
            alpha = self.get_number_of_observations() - 1
            pseudocounts = (pssm / background_frequencies)  @ bg_matrix

            # pssm = (alpha * pssm + beta * pseudocounts) / (alpha + beta)
            pssm *= alpha
            pssm += beta * pseudocounts
            pssm /= alpha + beta

        # last steps
        pssm /= np.abs(pssm).sum(axis=1).reshape(-1, 1)  # l1 normalization
        pssm /= background_frequencies
        pssm = 2 * np.log2(pssm)  # NOTE: might give a warning
        pssm[pssm == -np.inf] = -20  # replace all -inf with -20

        # omit columns (rows in PSSM) corresponding to gaps in the primary sequence and return the PSSM
        indices_to_include = (self.sequences_matrix[0] != '-')
        return np.rint(pssm[indices_to_include, :]).astype(np.int64)  # round to integers

    def get_size(self):
        """
        Return the number of sequences in the MSA and the MSA length, i.e.
        the number of columns in the MSA. This includes gaps.

        :return: Tuple of two integers. First element is the number of
                 sequences in the MSA, second element is the MSA length.
        """
        return self.sequences_matrix.shape

    def get_primary_sequence(self):
        """
        Return the primary sequence of the MSA. In this exercise, the primary
        sequence is always the first sequence of the MSA. The returned 
        sequence must NOT include gap characters.

        :return: String containing the ungapped primary sequence.
        """
        return self.sequences[0].replace('-', '')

    def get_sequence_weights(self):
        """
        Return the calculated sequence weights for all sequences in the MSA.
        The order of weights in the array must be equal to the order of the
        sequences in the MSA.

        :return: Numpy array (dtype=numpy.float64) containing the weights for
                 all sequences in the MSA.
        """
        weights = np.zeros(self.sequences_matrix.shape, dtype=np.float64)

        for column_index, column in enumerate(self.sequences_matrix.T):
            aa_count_dict = dict(zip(*np.unique(column, return_counts=True)))
            r = len(aa_count_dict)
            if r > 1:
                for seq_index, aa in enumerate(column):
                    weights[seq_index, column_index] = 1 / (r * aa_count_dict[aa])

        return np.sum(weights, axis=1)

    def get_number_of_observations(self):
        """
        Return the estimated number of independent observations in the MSA.

        :return: Estimate of independent observation (dtype=numpy.float64).
        """
        num_obs = sum(len(np.unique(column)) for column in self.sequences_matrix.T)
        num_obs /= np.float64(self.sequences_matrix.shape[1])
        return num_obs
