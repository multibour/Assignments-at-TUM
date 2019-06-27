import numpy as np
from itertools import product, combinations
# from pathlib import Path


"""
ATTENTION: Use the following dictionaries to get the correct index for each
           amino acid when accessing any type of matrix (PSSM or substitution
           matrix) parameters. Failure to do so will most likely result in not
           passing the tests.
"""
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: index for index, aa in enumerate(ALPHABET)}
INT_TO_AA = {index: aa for index, aa in enumerate(ALPHABET)}


class BlastDb:
    alphabet = frozenset(ALPHABET)
    word_length = 3

    def __init__(self):
        """
        Initialize the BlastDb class.

        Database (self.db) Format:
            Key: 
                string: the amino acid sequence
            Value:
                tuple:
                    - frozenset: all words the sequence has
                    - integer: number of instances of the sequence in database
        """
        self.db = dict()
        self.words = set()

    def add_sequence(self, sequence):
        """
        Add a sequence to the database.

        :param sequence: a protein sequence (string).
        """
        if not set(sequence).issubset(BlastDb.alphabet):
            raise ValueError('Given sequence contains invalid residues.')

        if sequence not in self.db:
            word_set = set()
            for index in range(0, len(sequence)-BlastDb.word_length+1):
                word_set.add(sequence[index:index+BlastDb.word_length])

            self.db[sequence] = (frozenset(word_set), 1)
            self.words.update(word_set)

        else:
            val = self.db[sequence]
            self.db[sequence] = (val[0], val[1]+1)  # Might be a costly operation if the set is copied, not moved

    def get_sequences(self, word):
        """
        Return all sequences in the database containing a given word.

        :param word: a word (string).

        :return: List with sequences.
        """
        return [seq for seq, val in self.db.items() for _ in range(val[1]) if word in seq]

    def iterate_sequences(self, word=None) -> str:
        for seq in self.db.keys():
            if word is None or word in seq:
                yield seq

    def get_db_stats(self):
        """
        Return some database statistics:
            - Number of sequences in database
            - Number of different words in database
            - Average number of words per sequence (rounded to nearest int)
            - Average number of sequences per word (rounded to nearest int)

        :return: Tuple with four integer numbers corresponding to the mentioned
                 statistics (in order of listing above).
        """
        num_sequences = sum(count for _, count in self.db.values())
        num_diff_words = len(self.words)

        total_count_dif_words_per_seq = sum(len(word_set) * count for word_set, count in self.db.values())

        avg_num_words_per_seq = total_count_dif_words_per_seq / num_sequences
        avg_num_seq_per_word = total_count_dif_words_per_seq / num_diff_words  # faster than this: # avg_num_seq_per_word = sum(count for word in self.words for word_set, count in self.db.values() if word in word_set) / num_diff_words

        avg_num_words_per_seq = np.rint(avg_num_words_per_seq).astype(int)
        avg_num_seq_per_word = np.rint(avg_num_seq_per_word).astype(int)

        return num_sequences, num_diff_words, avg_num_words_per_seq, avg_num_seq_per_word


class Blast:
    word_length = 3

    def __init__(self, substitution_matrix):
        """
        Initialize the Blast class with the given substitution_matrix.

        :param substitution_matrix: 20x20 amino acid substitution score matrix.
        """
        self.substitution_matrix = substitution_matrix

    @staticmethod
    def _all_possible_words(loop_count=None):
        if loop_count is None:
            loop_count = Blast.word_length

        for word in product(ALPHABET, repeat=loop_count):
            yield ''.join(word)

    @staticmethod
    def _iterate_words(sequence):
        for index in range(0, len(sequence) - Blast.word_length + 1):
            yield sequence[index:index+Blast.word_length]

    def _calculate_score(self, word1, word2, is_pssm=False):
        if not is_pssm:
            return sum(self.substitution_matrix[AA_TO_INT[aa1], AA_TO_INT[aa2]] for aa1, aa2 in zip(word1, word2))
        else:
            return sum(substitution_row[AA_TO_INT[aa2]] for substitution_row, aa2 in zip(word1, word2))

    def _calculate_score_single(self, a, b, is_pssm=False):
        if not is_pssm:
            return self.substitution_matrix[AA_TO_INT[a], AA_TO_INT[b]]
        else:
            return a[AA_TO_INT[b]]

    def _get_words_and_scores_per_index(self, data, T=11, is_pssm=False):
        out = dict()
        scores = dict()

        for index, word in enumerate(Blast._iterate_words(data)):  # (m-2) iterations
            out[index] = set()
            for word_to_compare in Blast._all_possible_words():  # 20^3 iterations
                score = self._calculate_score(word, word_to_compare, is_pssm=is_pssm)
                if score >= T:
                    out[index].add(word_to_compare)
                    scores[(index, word_to_compare)] = score

        return out, scores

    @staticmethod
    def _get_seq_to_ind_and_words(blast_db: BlastDb):
        '''
        Returns a dictionary that maps sequences to a list of tuples containing a position index and the word
        it is pointing to.
        :return: a dictionary of format:
                    key: protein sequence (string)
                    value: (list of tuples)
                        0: index (integer)
                        1: word (string)
        '''
        return {seq: list(enumerate(Blast._iterate_words(seq))) for seq in blast_db.iterate_sequences()}

    @staticmethod
    def _get_seq_to_ind_to_word(blast_db: BlastDb):
        return {seq: dict(enumerate(Blast._iterate_words(seq))) for seq in blast_db.iterate_sequences()}

    def _get_hsp(self, current_score, start_query, start_target, query, target, X, is_pssm=False):
        end_query = start_query + Blast.word_length - 1
        end_target = start_target + Blast.word_length - 1

        # align right hand side
        current_score, (end_query, end_target) = self.__extend(current_score, end_query, end_target, query, target,
                                                               X=X, is_pssm=is_pssm, right=True)

        # align left hand side
        current_score, (start_query, start_target) = self.__extend(current_score, start_query, start_target,
                                                                   query, target, X=X, is_pssm=is_pssm, right=False)

        return start_query, start_target, end_query-start_query+1, current_score

    def __extend(self, current_score, pos_q, pos_t, query, target, X, is_pssm=False, right=True):
        add_val = 1 if right is True else -1

        i = pos_q + add_val
        j = pos_t + add_val
        cache = (pos_q, pos_t)
        max_score = current_score

        check = (lambda: i < len(query) and j < len(target)) if right is True else (lambda: i >= 0 and j >= 0)
        while check():
            current_score += self._calculate_score_single(query[i], target[j], is_pssm=is_pssm)

            if current_score <= max_score - X:
                break
            elif current_score > max_score:
                max_score = current_score
                cache = (i, j)

            i += add_val
            j += add_val

        current_score = max_score
        return current_score, cache

    def get_words(self, *, sequence=None, pssm=None, T=11):
        """
        Return all words with score >= T for given protein sequence or PSSM.
        Only a sequence or PSSM will be provided, not both at the same time.
        A word may only appear once in the list.

        :param sequence: a protein sequence (string).
        :param pssm: a PSSM (Lx20 matrix, where L is length of sequence).
        :param T: score threshold T for the words.

        :return: List of unique words.
        """

        if (sequence is None and pssm is None) or (sequence is not None and pssm is not None):
            raise Exception('either sequence or pssm should be provided, not both nor none!')

        words = set()
        is_pssm = pssm is not None
        data = pssm if is_pssm else sequence

        for word, word_to_compare in product(Blast._iterate_words(data), Blast._all_possible_words()):

            score = self._calculate_score(word, word_to_compare, is_pssm=is_pssm)

            if score >= T:
                words.add(word_to_compare)

        return list(words)

    def search_one_hit(self, blast_db: BlastDb, *, query=None, pssm=None, T=13, X=5, S=30):
        """
        Search a database for target sequences with a given query sequence or
        PSSM. Return a dictionary where the keys are the target sequences for
        which HSPs have been found and the corresponding values are lists of
        tuples. Each tuple is a HSP with the following elements (and order):
            - Start position of HSP in query sequence
            - Start position of HSP in target sequence
            - Length of the HSP
            - Total score of the HSP
        The same HSP may not appear twice in the list (remove duplicates).
        Only a sequence or PSSM will be provided, not both at the same time.

        :param blast_db: BlastDB class object with protein sequences.
        :param query: query protein sequence.
        :param pssm: query PSSM (Lx20 matrix, where L is length of sequence).
        :param T: score threshold T for the words.
        :param X: drop-off threshold X during extension.
        :param S: score threshold S for the HSP.

        :return: dictionary of target sequences and list of HSP tuples.
        """
        if (query is None and pssm is None) or (query is not None and pssm is not None):
            raise Exception('either sequence or pssm should be provided, not both nor none!')

        out = dict()
        is_pssm = pssm is not None
        data = pssm if is_pssm else query

        ind_to_candidate_words, ind_word_to_score = self._get_words_and_scores_per_index(data, T=T, is_pssm=is_pssm)
        target_seq_to_ind_and_words = Blast._get_seq_to_ind_and_words(blast_db)

        for (position_query, candidate_words), (target_sequence, target_words_and_positions) \
            in product(ind_to_candidate_words.items(), target_seq_to_ind_and_words.items()):

            for position_target, target_word in target_words_and_positions:
                if target_word not in candidate_words:
                    continue

                current_score = ind_word_to_score[(position_query, target_word)]
                hsp = self._get_hsp(current_score, position_query, position_target,
                                    query=data, target=target_sequence, X=X, is_pssm=is_pssm)
                current_score = hsp[3]

                # add to dictionary
                if current_score >= S:
                    if target_sequence not in out:
                        out[target_sequence] = set()
                    out[target_sequence].add(hsp)

        return {k: list(v) for k, v in out.items()}

    def search_two_hit(self, blast_db: BlastDb, *, query=None, pssm=None, T=11, X=5, S=30, A=40):
        """
        Search a database for target sequences with a given query sequence or
        PSSM. Return a dictionary where the keys are the target sequences for
        which HSPs have been found and the corresponding values are lists of
        tuples. Each tuple is a HSP with the following elements (and order):
            - Start position of HSP in query sequence
            - Start position of HSP in target sequence
            - Length of the HSP
            - Total score of the HSP
        The same HSP may not appear twice in the list (remove duplictes).
        Only a sequence or PSSM will be provided, not both at the same time.

        :param blast_db: BlastDB class object with protein sequences.
        :param query: query protein sequence.
        :param pssm: query PSSM (Lx20 matrix, where L is length of sequence).
        :param T: score threshold T for the words.
        :param X: drop-off threshold X during extension.
        :param S: score threshold S for the HSP.
        :param A: max distance A between two hits for the two-hit method.

        :return: dictionary of target sequences and list of HSP tuples.
        """
        '''
        NOTE: Two hit search sometimes does not work correctly
        TODO: fix this issue and do further performance optimizations
        '''
        if (query is None and pssm is None) or (query is not None and pssm is not None):
            raise Exception('either sequence or pssm should be provided, not both nor none!')

        out = dict()
        is_pssm = pssm is not None
        data = pssm if is_pssm else query

        ind_to_candidate_words, ind_word_to_score = self._get_words_and_scores_per_index(data, T=T, is_pssm=is_pssm)
        target_seq_to_ind_to_word = Blast._get_seq_to_ind_to_word(blast_db)

        for target_sequence, target_position_to_word in target_seq_to_ind_to_word.items():
            query_pos_to_target_positions = {q_pos: sorted(t_pos for t_pos, t_word in target_position_to_word.items() if t_word in candidate_words) for q_pos, candidate_words in ind_to_candidate_words.items()}
            forbidden = list()
            found_hsps = set()

            for q_pos_l, q_pos_r in combinations(sorted(query_pos_to_target_positions.keys()), 2):
                if q_pos_l + Blast.word_length - 1 >= q_pos_r:
                    continue

                for t_pos_l, t_pos_r in product(query_pos_to_target_positions[q_pos_l], query_pos_to_target_positions[q_pos_r]):
                    if t_pos_r <= t_pos_l:
                        continue

                    # are they valid HSP candidates?
                    if q_pos_r - q_pos_l != t_pos_r - t_pos_l \
                            or t_pos_l + Blast.word_length - 1 >= t_pos_r \
                            or t_pos_r - t_pos_l > A \
                            or any((s-2 <= t_pos_l <= e+2 and q_pos_l-t_pos_l == d)
                                   or (s-2 <= t_pos_r <= e+2 and q_pos_r-t_pos_r == d) for s, e, d in forbidden):
                        continue

                    # extend left from candidate on the right
                    current_score = ind_word_to_score[(q_pos_r, target_position_to_word[t_pos_r])]
                    current_score, (start_q, start_t) = self.__extend(current_score, q_pos_r, t_pos_r, query=data,
                                                                      target=target_sequence, X=X, is_pssm=is_pssm,
                                                                      right=False)

                    # if left extension does not cross the left candidate
                    if t_pos_l + Blast.word_length <= start_t:
                        continue

                    # set the end positions
                    end_q, end_t = q_pos_r + Blast.word_length - 1, t_pos_r + Blast.word_length - 1

                    # extend right from the unified hsp candidate
                    current_score, (end_q, end_t) = self.__extend(current_score, end_q, end_t, query=data,
                                                                  target=target_sequence, X=X, is_pssm=is_pssm,
                                                                  right=True)

                    if current_score >= S:
                        found_hsps.add((start_q, start_t, end_q - start_q + 1, current_score))
                        forbidden.append((start_t, end_t, start_q - start_t))

            if len(found_hsps) > 0:
                out[target_sequence] = list(found_hsps)

        return out
