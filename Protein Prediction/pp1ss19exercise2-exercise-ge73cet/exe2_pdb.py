# -*- coding: utf-8 -*-
"""
IMPORTANT!:
Before writing an email asking questions such as
'What does this input has to be like?' or 
'What return value do you expect?' PLEASE read our
exercise sheet and the information in this template
carefully.
If something is still unclear, PLEASE talk to your
colleagues before writing an email!

If you experience technical issues or if you find a
bug we are happy to answer your questions. However,
in order to provide quick help in such cases we need 
to avoid unnecessary emails such as the examples
shown above.
"""


from Bio.PDB.MMCIFParser import MMCIFParser  # Tip: This module might be useful for parsing...
from Bio.Data.IUPACData import protein_letters_3to1
import numpy as np


############# Exercise 2: Protein Data Bank #############
# General remark: In our exercise every structure will have EXACTLY ONE model.
# This is true for nearly all X-Ray structures. NMR structures have several models.
class PDB_Parser:
    
    def __init__(self, path):
        '''
            Initialize every PDB_Parser with a path to a structure-file in CIF format.
            An example file is included in the repository (7ahl.cif).
            Tip: Store the parsed structure in an object variable instead of parsing it
            again & again ...
        '''
        cif_parser = MMCIFParser(QUIET=True)  # parser object for reading in structure in CIF format
        self.structure = cif_parser.get_structure('structure', path)
        self.model = self.structure[0]
        self.residue_dict = {k.upper(): v for d in [protein_letters_3to1, {'HOH': ''}] for k, v in d.items()}

    # 3.8 Chains
    def get_number_of_chains(self):
        '''
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
            Return:
                Number of chains in this structure as integer.
        '''
        n_chains = len(self.model)
        return n_chains
    
    # 3.9 Sequence  
    def get_sequence(self, chain_id):
        '''
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
                chain_id  : String (usually in ['A','B', 'C' ...]. The number of chains
                        depends on the specific protein and the resulting structure)
            Return:
                Return the amino acid sequence (single-letter alphabet!) of a given chain (chain_id)
                in a Biopython.PDB structure as a string.
        '''
        chain = self.model[chain_id]
        sequence = ''.join(self.residue_dict[residue.resname] for residue in chain)
        return sequence
        
    # 3.10 Water molecules
    def get_number_of_water_molecules(self, chain_id):
        '''
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
                chain_id  : String (usually in ['A','B', 'C' ...]. The number of chains
                        depends on the specific protein and the resulting structure)
            Return:
                Return the number of water molecules of a given chain (chain_id)
                in a Biopython.PDB structure as an integer.
        '''
        chain = self.model[chain_id]
        n_waters = sum(1 for residue in chain if residue.id[0] == 'W')
        return n_waters
    
    # 3.11 C-Alpha distance    
    def get_ca_distance(self, chain_id_1, index_1, chain_id_2, index_2):
        ''' 
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
                chain_id_1 : String (usually in ['A','B', 'C' ...]. The number of chains
                                depends on the specific protein and the resulting structure)
                index_1    : index of a residue in a given chain in a Biopython.PDB structure
                chain_id_2 : String (usually in ['A','B', 'C' ...]. The number of chains
                            depends on the specific protein and the resulting structure)
                index_2    : index of a residue in a given chain in a Biopython.PDB structure
        
                chain_id_1 and index_1 describe precisely one residue in a PDB structure,
                chain_id_2 and index_2 describe the second residue.
        
            Return: 
                Return the C-alpha (!) distance between the two residues, described by 
                chain_id_1/index_1 and chain_id_2/index_2. Round the returned value via int().
            
            The reason for using two different chains as an input is that also the distance
            between residues of different chains can be interesting.
            Different chains in a PDB structure can either occur between two different proteins 
            (Heterodimers) or between different copies of the same protein (Homodimers).
        '''

        residue1 = self.model[chain_id_1][index_1]
        residue2 = self.model[chain_id_2][index_2]
        
        ca_distance = np.linalg.norm(residue1['CA'].coord - residue2['CA'].coord)
        return int(ca_distance)

    # 3.12 Contact Map  
    def get_contact_map(self, chain_id):
        '''
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
                chain_id  : String (usually in ['A','B', 'C' ...]. The number of chains
                        depends on the specific protein and the resulting structure)
            Return:
                Return a complete contact map (see description in exercise sheet) 
                for a given chain in a Biopython.PDB structure as numpy array. 
                The values in the matrix describe the c-alpha distance between all residues 
                in a chain of a Biopython.PDB structure.
                Only integer values of the distance have to be given (see below).
        '''

        chain = self.model[chain_id]
        is_aa = lambda res: res.id[0] == ' '  # is amino acid?

        length = sum(1 for res in chain if is_aa(res))
        contact_map = np.zeros((length, length), dtype=np.float32)

        for i, residue_i in zip(range(0, length),
                                (res for res in chain if is_aa(res))):
            for j, residue_j in zip(range(i, length),
                                    (res for res in chain if res.id[1] >= residue_i.id[1] and is_aa(res))):
                try:
                    contact_map[i, j] = self.get_ca_distance(chain_id, residue_i.id, chain_id, residue_j.id)
                    contact_map[j, i] = contact_map[i, j]
                except KeyError as err:
                    contact_map[i, j], contact_map[j, i] = np.nan, np.nan
                    print(err)

        return contact_map.astype(np.int64)  # return rounded (integer) values

    # 3.13 B-Factors    
    def get_bfactors(self, chain_id):
        '''
            Input:
                self: Use Biopython.PDB structure which has been stored in an object variable
                chain_id  : String (usually in ['A','B', 'C' ...]. The number of chains
                        depends on the specific protein and the resulting structure)
            Return:
                Return the B-Factors for all residues in a chain of a Biopython.PDB structure.
                The B-Factors describe the mobility of an atom or a residue.
                In a Biopython.PDB structure B-Factors are given for each atom in a residue.
                Calculate the mean B-Factor for a residue by averaging over the B-Factor 
                of all atoms in a residue.
                Sometimes B-Factors are not available for a certain residue; 
                (e.g. the residue was not resolved); insert np.nan for those cases.
            
                Finally normalize your B-Factors using Standard scores (zero mean, unit variance).
                You have to use np.nanmean, np.nanvar etc. if you have nan values in your array.
                The returned data structure has to be a numpy array rounded again to integer.
        '''
        chain = self.model[chain_id]
        length = len(chain) - self.get_number_of_water_molecules(chain_id)
        b_factors = np.zeros(length, dtype=np.float32)

        for i, residue in enumerate(chain):
            if residue.id[0] == 'W':  # if water molecule
                break
            temp_list = [(atom.bfactor if hasattr(atom, 'bfactor') else np.nan) for atom in residue.get_atoms()]
            b_factors[i] = np.nanmean(temp_list)

        b_factors = (b_factors - np.nanmean(b_factors)) / np.nanstd(b_factors)
        return b_factors.astype(np.int64)  # return rounded (integer) values


def main():
    print('PDB parser class.')
    x = PDB_Parser('tests/6aa6.cif')
    return None
    

if __name__ == '__main__':
    main()