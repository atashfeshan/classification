import os
import pandas as pd

address = os.path.join(os.path.realpath(os.getcwd()), '1.csv')
raw_data = pd.read_csv(address)

