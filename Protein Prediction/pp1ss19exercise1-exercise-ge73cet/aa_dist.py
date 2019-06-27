##############
# Exercise 2.6
##############


class AADist:
    """
    The class provides a method to read fasta files and to calculate certain statistics on the read sequences.
    """
    
    def __init__(self, filepath):
        self.__sequences = []
        self.read_fasta(filepath)

    def get_counts(self):
        return len(self.__sequences)

    def get_average_length(self):
        sum = 0
        for seq in self.__sequences:
            sum += len(seq)
        return sum / len(self.__sequences)

    def read_fasta(self, path):
        with open(path, 'r') as infile:
            seq = ''
            sequence_started = False
            for line in infile:
                if line.startswith('>') or line.startswith(';'):
                    if sequence_started:
                        self.__sequences.append(seq)
                        seq = ''
                        sequence_started = False
                    continue
                sequence_started = True
                seq += line.strip()
                seq = seq.rstrip('*')  # trim the stop signal
            self.__sequences.append(seq)

    def get_abs_frequencies(self):
        # return number of occurences not normalized by length
        abs_frequencies = {}

        for seq in self.__sequences:
            for aa in seq:
                if aa not in abs_frequencies:
                    abs_frequencies[aa] = 0
                abs_frequencies[aa] += 1

        return abs_frequencies

    def get_av_frequencies(self):
        # return number of occurences normalized by length
        av_frequencies = {}
        counter = 0

        for seq in self.__sequences:
            for aa in seq:
                if aa not in av_frequencies:
                    av_frequencies[aa] = 0
                av_frequencies[aa] += 1
                counter += 1

        for key in av_frequencies:
            av_frequencies[key] /= counter

        return av_frequencies
