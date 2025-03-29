from rdkit import Chem
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings ( 'ignore' )
class Molcombiner:
    def __init__(self,class1,class2,class3,maker_a='Fr',maker_b='Cs'):
        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.maker_a = maker_a
        self.maker_b = maker_b
        self.smiles_z = []
    
    def get_neiid_bysymbol(self,mol,marker):
        try:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == marker:
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) > 1:
                        print('Cannot process more than one neighbor, will only return one of them')
                    atom_nb = neighbors[0]
                    return atom_nb.GetIdx()
        except Exception as e:
            print(e)
            return None
       
    def get_id_bysymbol(self,mol,marker):
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == marker:
                return atom.GetIdx()
    

    def combine2frags(self,mol_a,mol_b):
        merged_mol = Chem.CombineMols(mol_a,mol_b)
        bind_pos_a = self.get_neiid_bysymbol(merged_mol,self.maker_a)
        bind_pos_b = self.get_neiid_bysymbol(merged_mol,self.maker_b)
        ed_merged_mol = Chem.EditableMol(merged_mol)
        ed_merged_mol.AddBond(bind_pos_a,bind_pos_b,order = Chem.rdchem.BondType.SINGLE)
        marker_a_idx = self.get_id_bysymbol(ed_merged_mol.GetMol(),self.maker_a)
        ed_merged_mol.RemoveAtom(marker_a_idx)
        temp_mol = ed_merged_mol.GetMol()
        marker_b_idx = self.get_id_bysymbol(temp_mol,self.maker_b)
        ed_merged_mol =Chem.EditableMol(temp_mol)
        ed_merged_mol.RemoveAtom(marker_b_idx)
        final_mol = ed_merged_mol.GetMol()
        return final_mol
    
    def generate_combined_smiles(self):
        for smiles1 in self.class1:
            for smiles2 in self.class2:
                for smiles3 in self.class3:
                    try:
                        mol1 = Chem.MolFromSmiles(smiles1)
                        mol2 = Chem.MolFromSmiles(smiles2)
                        mol3 = Chem.MolFromSmiles(smiles3)
                        mol_intermediate = self.combine2frags(mol1,mol2)
                        mol_final = self.combine2frags(mol_intermediate,mol3)
                        smiles_final = Chem.MolToSmiles(mol_final)
                        self.smiles_z.append(smiles_final)
                    except Exception as e:
                        print(f"Error: Failed to convert SMILES to molecule - {e}")
        return self.smiles_z

def process_data(input_file, output_file):
    df = pd.read_csv(input_file)
    combiner = Molcombiner(class1=df["D_Fr"], class3=df["A_Cs"],class2="[Fr]c1ccc([Cs])cc1")
    output_rows = []
    for row1 in df.itertuples(index=False):
        for row2 in df.itertuples(index=False):
            try:
                mol1 = Chem.MolFromSmiles(row1.D_Fr)
                mol2 = Chem.MolFromSmiles(combiner.class2)
                mol3 = Chem.MolFromSmiles(row2.A_Cs)
                mol_intermediate = combiner.combine2frags(mol1, mol2)
                mol_final = combiner.combine2frags(mol_intermediate, mol3)
                smiles_final = Chem.MolToSmiles(mol_final)

                d_homo = row1.D_HOMO
                d_lumo = row1.D_LUMO
                a_homo = row2.A_HOMO
                a_lumo = row2.A_LUMO
                output_rows.append({"Combined_SMILES": smiles_final,
                                    "D_Fr": row1.D_Fr,
                                    "A_Cs": row2.A_Cs,
                                    "D_HOMO": d_homo,
                                    "D_LUMO": d_lumo,
                                    "A_HOMO": a_homo,
                                    "A_LUMO": a_lumo})
            except Exception as e:
                print(f"Error: Failed to process row - {e}")
        output_df = pd.DataFrame(output_rows, columns=["Combined_SMILES", "D_Fr", "A_Cs", "D_HOMO", "D_LUMO", "A_HOMO", "A_LUMO"])
        output_df.to_csv(output_file, index=False)

dataLocation = '../data'

datafile = "zuhe_d102a120_scores.csv"
data = pd.read_csv ( os.path.join ( dataLocation , datafile ) )

resultSaveLocation = '../combined_data/'
if not os.path.exists ( resultSaveLocation ) :
	os.makedirs ( resultSaveLocation )
        
combined_data = process_data(os.path.join(dataLocation,'combine_d102a120.csv'),os.path.join(resultSaveLocation,'combined_d102a120.csv'))

