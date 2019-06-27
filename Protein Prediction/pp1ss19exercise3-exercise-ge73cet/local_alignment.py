import numpy as np
from tests.matrices import MATRICES


class LocalAlignment:
    def __init__(self, string1, string2, gap_penalty, matrix):
        """
        :param string1: first string to be aligned, string
        :param string2: second string to be aligned, string
        :param gap_penalty: gap penalty, integer
        :param matrix: substitution matrix containing scores for amino acid
                       matches and mismatches, dict

        Attention! string1 is used to index columns, string2 is used to index rows
        """
        self.string1 = string1
        self.string2 = string2
        self.gap_penalty = gap_penalty
        self.substitution_matrix = matrix
        self.score_matrix = np.zeros((len(string2) + 1, len(string1) + 1), dtype=np.int)
        self.aligned_residues = {1: set(), 2: set()}
        self.alignment = ('', '')
        self.align()

    def align(self):
        """
        Align given strings using the Smith-Waterman algorithm.
        NB: score matrix and the substitution matrix are different matrices!
        """
        # Initialize the Traceback Matrix
        traceback_matrix = np.ndarray(self.score_matrix.shape, dtype=object)
        traceback_matrix[:, :] = ''

        # Set Score Matrix and the Traceback Matrix
        for i in range(1, self.score_matrix.shape[0]):
            for j in range(1, self.score_matrix.shape[1]):
                diag = self.score_matrix[i-1, j-1] + self.substitution_matrix[self.string2[i-1]][self.string1[j-1]]
                up = self.score_matrix[i-1, j] + self.gap_penalty
                left = self.score_matrix[i, j-1] + self.gap_penalty

                max_value = max(diag, up, left, 0)
                self.score_matrix[i, j] = max_value

                if diag == max_value:
                    traceback_matrix[i, j] += 'd'
                if up == max_value:
                    traceback_matrix[i, j] += 'u'
                if left == max_value:
                    traceback_matrix[i, j] += 'l'

        # Find Alignments
        i, j = np.unravel_index(self.score_matrix.argmax(), self.score_matrix.shape)
        str1, str2 = [], []
        while self.score_matrix[i, j] > 0:
            traceback_cell = traceback_matrix[i, j]

            if traceback_cell == '':
                break
            diag = self.score_matrix[i-1, j-1] if 'd' in traceback_cell else 0
            up = self.score_matrix[i-1, j] if 'u' in traceback_cell else 0
            left = self.score_matrix[i, j-1] if 'l' in traceback_cell else 0

            max_value = max(diag, up, left)

            if max_value == diag:
                str1.append(self.string1[j-1])
                str2.append(self.string2[i-1])
                i -= 1
                j -= 1
            elif max_value == up:
                str1.append('-')
                str2.append(self.string2[i-1])
                i -= 1
            elif max_value == left:
                str1.append(self.string1[j-1])
                str2.append('-')
                j -= 1

            self.aligned_residues[1].add(j)
            self.aligned_residues[2].add(i)

        if len(str1) > 0:
            self.alignment = (''.join(reversed(str1)), ''.join(reversed(str2)))

    def has_alignment(self):
        """
        :return: True if a local alignment has been found, False otherwise
        """
        return self.alignment[0] != ''

    def get_alignment(self):
        """
        :return: alignment represented as a tuple of aligned strings
        """
        return self.alignment

    def is_residue_aligned(self, string_number, residue_index):
        """
        :param string_number: number of the string (1 for string1, 2 for string2) to check
        :param residue_index: index of the residue to check
        :return: True if the residue with a given index in a given string has been alined
                 False otherwise
        """
        if string_number not in [1, 2]:
            raise ValueError('The first parameter has to be either 1 for the 1st string or 2 for the 2nd string.')

        return residue_index in self.aligned_residues[string_number]


if __name__ == '__main__':
    x = LocalAlignment("ARNDCEQGHI", "DDCEQHG", -6, MATRICES['blosum'])
    pass
