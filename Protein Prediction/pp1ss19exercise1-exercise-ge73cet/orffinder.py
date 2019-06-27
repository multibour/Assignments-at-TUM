##############
# Exercise 2.5
##############

# You can use the supplied test cases for your own testing. Good luck!
from funcs import *


def get_orfs(genome):
    if len(set(genome) - {'A', 'T', 'G', 'C'}) > 0:
        raise TypeError('The genome is not a valid DNA sequence')

    orfs = []

    for genome_, reversed in [(genome, False), (complementary(genome)[::-1], True)]:
        for frame in range(3):  # 3 reading frames of the ORF
            genome_reframed = genome_[frame:] + genome_ + genome_[:frame]  # consider circular DNA

            reading = False
            start = end = -1

            for index in range(0, len(genome_reframed), 3):
                if index + 3 > len(genome_reframed):
                    break

                if not reading:
                    if codon_dict[genome_reframed[index:index+3]] == 'M':  # start signal received
                        reading = True
                        start = index
                elif codon_dict[genome_reframed[index:index+3]] == 'STOP':  # stop signal received
                    end = index + 2

                    if (index - start) // 3 > 33:  # if the sequence is long enough
                        aa_sequence = codons_to_aa(genome_reframed[start:index])
                        start = (frame+start if not reversed else len(genome) - (frame+start) - 1) % len(genome_)
                        end = (frame+end if not reversed else len(genome) - (frame+end) - 1) % len(genome_)

                        orfs.append((start, end, aa_sequence, reversed))

                    reading = False
                    start = end = -1

    # check for redundant proteins
    to_remove = set()
    for index, orf in enumerate(orfs):
        for index_to_check, orf_to_check in enumerate(orfs):
            if index_to_check == index:
                continue
            if orf == orf_to_check:
                continue

            if orf[1] == orf_to_check[1] and orf[3] == orf_to_check[3]:
                if orf[2].endswith(orf_to_check[2]):
                    to_remove.add(index_to_check)
                elif orf_to_check[2].endswith(orf[2]):
                    to_remove.add(index)

    # delete elements by indices
    for index in sorted(to_remove, reverse=True):
        orfs.pop(index)

    return orfs
