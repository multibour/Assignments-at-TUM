from collections import Counter

codon_dict = {
    'TAA': 'STOP',
    'TGA': 'STOP',
    'TAG': 'STOP',

    'GCT': 'A',
    'GCC': 'A',
    'GCA': 'A',
    'GCG': 'A',

    'CGT': 'R',
    'CGC': 'R',
    'CGA': 'R',
    'CGG': 'R',
    'AGA': 'R',
    'AGG': 'R',

    'AAT': 'N',
    'AAC': 'N',

    'GAT': 'D',
    'GAC': 'D',

    'TGT': 'C',
    'TGC': 'C',

    'CAA': 'Q',
    'CAG': 'Q',

    'GAA': 'E',
    'GAG': 'E',

    'GGT': 'G',
    'GGC': 'G',
    'GGA': 'G',
    'GGG': 'G',

    'CAT': 'H',
    'CAC': 'H',

    'ATT': 'I',
    'ATC': 'I',
    'ATA': 'I',

    'TTA': 'L',
    'TTG': 'L',
    'CTT': 'L',
    'CTC': 'L',
    'CTA': 'L',
    'CTG': 'L',

    'AAA': 'K',
    'AAG': 'K',

    'ATG': 'M',  # START

    'TTT': 'F',
    'TTC': 'F',

    'CCT': 'P',
    'CCC': 'P',
    'CCA': 'P',
    'CCG': 'P',

    'TCT': 'S',
    'TCC': 'S',
    'TCA': 'S',
    'TCG': 'S',
    'AGT': 'S',
    'AGC': 'S',

    'ACT': 'T',
    'ACC': 'T',
    'ACA': 'T',
    'ACG': 'T',

    'TGG': 'W',

    'TAT': 'Y',
    'TAC': 'Y',

    'GTT': 'V',
    'GTC': 'V',
    'GTA': 'V',
    'GTG': 'V'
}

complement_dict = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}


def codons_to_aa(orf):
    if len(orf) % 3 != 0:
        return None
    codons = [orf[i:i+3] for i in range(0, len(orf), 3)]
    return ''.join(codon_dict[c] for c in codons if codon_dict[c] != 'STOP')


def aa_dist(aa_sequence):
    counted = Counter(aa_sequence)
    for key in counted:
        counted[key] /= len(aa_sequence)
    return counted


def complementary(sequence):
    return ''.join(complement_dict[s] for s in sequence.upper())
2