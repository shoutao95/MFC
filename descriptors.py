import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
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

desc_sec = pd.read_csv(os.path.join (dataLocation,'descriptors_sec.csv'))
descriptors_sec = desc_sec [ 'descriptors' ].to_list( )

descs = [ desc_name [ 0 ] for desc_name in Descriptors._descList ]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator ( descs )
descriptor = pd.DataFrame ( [ desc_calc.CalcDescriptors ( mol ) for mol in data [ "mol" ] ] )
descriptor.columns = descs
descriptors = descriptor [ descriptors_sec ]
columns = descriptors.columns.to_list()
x = np.array(descriptors)

savemodel = os.path.join ( 'descriptors' + '_ss.pkl' )
with open ( savemodel , 'rb' ) as file :
    SS = pickle.load ( file )
    
x_ss = SS.transform(x)
ss = pd.DataFrame(x_ss)
ss.columns = columns
ss.to_csv ( os.path.join ( dataLocation , 'descriptors.csv' ) , index = False )
