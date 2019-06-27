##############
# Exercise 2.7
##############


def isCharged(aa):
    return isPositivelyCharged(aa) or isNegativelyCharged(aa)


def isPositivelyCharged(aa):
    return aa in ['R', 'H', 'K']


def isNegativelyCharged(aa):
    return aa in ['D', 'E']


def isHydrophobic(aa):
    return aa in ['V', 'I', 'L', 'F', 'W', 'Y', 'M', 'A']


def isAromatic(aa):
    return aa in ['F', 'W', 'Y', 'H']


def isPolar(aa):
    return aa in ['R', 'N', 'D', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']


def isProline(aa):
    return aa == 'P'


def containsSulfur(aa):
    return aa == 'C' or aa == 'M'


def isAcid(aa):
    return aa == 'D' or aa == 'E'


def isBasic(aa):
    return aa == 'R' or aa == 'H' or aa == 'K'
