import numpy as np
from tests.matrices import MATRICES


class GlobalAlignment:
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
        self.alignments = []

        # Traceback Matrix for convenience
        # f: done, u: up, l: left, d: diagonal
        self.traceback_matrix = np.ndarray(self.score_matrix.shape, dtype=object)
        self.traceback_matrix[0, 0] = 'f'  # done/finished
        self.traceback_matrix[1:, 0] = 'u'  # up
        self.traceback_matrix[0, 1:] = 'l' # left
        self.traceback_matrix[1:, 1:] = ''  # initially empty

        self.align()

    def _backtrack(self, row, column, path, alignment_paths):
        path.append((row, column))

        cell = self.traceback_matrix[row, column]
        next_index = len(path)

        if 'u' in cell:
            self._backtrack(row - 1, column, path, alignment_paths)

        if 'l' in cell:
            if len(path) > next_index:
                del path[next_index:]
            self._backtrack(row, column - 1, path, alignment_paths)

        if 'd' in cell:
            if len(path) > next_index:
                del path[next_index:]
            self._backtrack(row - 1, column - 1, path, alignment_paths)

        if 'f' in cell:
            alignment_paths.append(path[::-1])  # reverse indexing already creates a copy, not a memory view

    def align(self):
        """
        Align given strings using the Needleman-Wunsch algorithm,
        store the alignments and the score matrix used to compute those alignments.
        NB: score matrix and the substitution matrix are different matrices!
        """
        # Set Score Matrix and Traceback Matrix
        for index in range(1, self.score_matrix.shape[0]):
            self.score_matrix[index, 0] = self.score_matrix[index-1, 0] + self.gap_penalty
        for index in range(1, self.score_matrix.shape[1]):
            self.score_matrix[0, index] = self.score_matrix[0, index-1] + self.gap_penalty

        for i in range(1, self.score_matrix.shape[0]):
            for j in range(1, self.score_matrix.shape[1]):
                diag = self.score_matrix[i-1, j-1] + self.substitution_matrix[self.string2[i-1]][self.string1[j-1]]
                up = self.score_matrix[i-1, j] + self.gap_penalty
                left = self.score_matrix[i, j-1] + self.gap_penalty

                max_val = max(diag, left, up)
                self.score_matrix[i, j] = max_val

                if max_val == diag:
                    self.traceback_matrix[i, j] += 'd'
                if max_val == left:
                    self.traceback_matrix[i, j] += 'l'
                if max_val == up:
                    self.traceback_matrix[i, j] += 'u'

        # Find Alignments with the Traceback Matrix
        alignment_paths = []
        self._backtrack(self.traceback_matrix.shape[0]-1, self.traceback_matrix.shape[1]-1, [], alignment_paths)

        for path in alignment_paths:
            str1, str2 = list(), list()

            for i in range(1, len(path)):
                prev_cell, cur_cell = path[i-1], path[i]

                if prev_cell[0] + 1 == cur_cell[0] and prev_cell[1] + 1 == cur_cell[1]:  # diagonal
                    str1.append(self.string1[cur_cell[1]-1])
                    str2.append(self.string2[cur_cell[0]-1])

                elif prev_cell[0] == cur_cell[0] and prev_cell[1] + 1 == cur_cell[1]:  # left
                    str1.append(self.string1[cur_cell[1]-1])
                    str2.append('-')

                elif prev_cell[0] + 1 == cur_cell[0] and prev_cell[1] == cur_cell[1]:  # up
                    str1.append('-')
                    str2.append(self.string2[cur_cell[0]-1])

                else:
                    raise Exception('Alignments were not constructed correctly. Reason not known...')

            self.alignments.append((''.join(str1), ''.join(str2)))

    def get_best_score(self):
        """
        :return: the highest score for the aligned strings, int

        """
        return self.score_matrix[-1, -1]

    def get_number_of_alignments(self):
        """
        :return: number of found alignments with the best score
        """
        return len(self.alignments)

    def get_alignments(self):
        """
        :return: list of alignments, where each alignment is represented
                 as a tuple of aligned strings
        """
        return self.alignments

    def get_score_matrix(self):
        """
        :return: matrix built during the alignment process as a list of lists
        """
        return self.score_matrix.tolist()


if __name__ == '__main__':
    x = GlobalAlignment("AVNCCEGQHI", "ARNDEQ", -1, MATRICES['identity'])
    pass
