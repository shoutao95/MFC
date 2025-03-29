import numpy as np
import pandas as pd
import os

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import warnings

warnings.filterwarnings ( 'ignore' )

dataLocation = '../data'

datafile = "data.csv"
data = pd.read_csv ( os.path.join ( dataLocation , datafile ) )

data [ "mol" ] = [ Chem.MolFromSmiles ( x ) for x in data [ "smiles" ] ]

columns = list ( range ( 2048 ) )
#Generate Morgan fingerprints
morg_fp = [ Chem.GetMorganFingerprintAsBitVect ( m , 3 , nBits = 2048 ) for m in data [ 'mol' ] ]
morg_fp_np = [ ]

for fp in morg_fp:
    arr = np.zeros ( (1 ,) )
    DataStructs.ConvertToNumpyArray ( fp , arr )
    morg_fp_np.append ( arr )
x_morg = morg_fp_np
x_morg = np.array ( x_morg )
x_morg = pd.DataFrame ( x_morg )
x_morg.columns = columns
x_morg.to_csv ( os.path.join ( dataLocation , 'x_morg.csv' ) , index = False )

#Generate Daylight fingerprints
rd_fp = [ Chem.RDKFingerprint ( m ) for m in data [ "mol" ] ]
rd_fp_np = [ ]
for fp in rd_fp:
    arr = np.zeros ( (1 ,) )
    DataStructs.ConvertToNumpyArray ( fp , arr )
    rd_fp_np.append ( arr )
x_rd = rd_fp_np
x_rd = np.array ( x_rd )
x_rd = pd.DataFrame ( x_rd )
x_rd.columns = columns
x_rd.to_csv ( os.path.join ( dataLocation , 'x_rd.csv' ) , index = False )

#Generate Atompairs fingerprints
AP_fp = [ Chem.GetHashedAtomPairFingerprintAsBitVect ( m ) for m in data [ "mol" ] ]
AP_fp_np = [ ]
for fp in AP_fp:
    arr = np.zeros ( (1 ,) )
    DataStructs.ConvertToNumpyArray ( fp , arr )
    AP_fp_np.append ( arr )
x_AP = AP_fp_np
x_AP = np.array ( x_AP )
x_AP = pd.DataFrame ( x_AP )
x_AP.columns = columns
x_AP.to_csv ( os.path.join ( dataLocation , 'x_AP.csv' ) , index = False )

#Generate Topological fingerprints
torsion_fp = [ Chem.GetHashedTopologicalTorsionFingerprintAsBitVect ( m ) for m in data [ "mol" ] ]
torsion_fp_np = [ ]
for fp in torsion_fp:
    arr = np.zeros ( (1 ,) )
    DataStructs.ConvertToNumpyArray ( fp , arr )
    torsion_fp_np.append ( arr )
x_torsion = torsion_fp_np
x_torsion = np.array ( x_torsion )
x_torsion = pd.DataFrame ( x_torsion )
x_torsion.columns = columns
x_torsion.to_csv ( os.path.join ( dataLocation , 'x_torsion.csv' ) , index = False )

pass





