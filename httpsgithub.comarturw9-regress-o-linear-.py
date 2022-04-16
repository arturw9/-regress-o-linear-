import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer

dados = pd.read_csv("./base_funcionarios.csv")
dados['start date'] = pd.to_datetime(dados['start date'])
dados.loc[dados['location'] == 'CA', 'location'] = 'California'
dados.loc[dados['location'] == 'Calif.', 'location'] = 'California'
dados.loc[dados['location'] == 'NY', 'location'] = 'New York'
dados['team matrix'] = dados['team matrix'].str.replace(" ", "")
dados.insert(2, 'team code', dados['team matrix'].str.slice(0, 3))
dados.insert(3, 'team name', dados['team matrix'].str.slice(4,  15).str.lower())
dados['team matrix'] = dados['team code'].str.cat(dados['team name'], sep = '-')
dados['employee'] = dados['employee'].str.title()
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
dataframe = pd.DataFrame(dados)
dataframe = mean_imputer.fit_transform(dataframe[['annual salary']])
dados['annual salary'] = dataframe
dataframe = pd.DataFrame(dados['performance level'])
dataframe = dataframe.fillna('NI', inplace = False)
dados['performance level'] = dataframe

dados