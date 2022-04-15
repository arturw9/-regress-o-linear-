# importando as libs
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# fonte: # fonte: http://www.portalaction.com.br/analise-de-regressao/12-estimacao-dos-parametros-do-modelo
# criando variáveis que serão usadas daqui para frente
# variável preditora
X = np.array([ 220, 220, 220, 220, 220, 225, 225, 225, 225, 225, 230, 230, 230, 230, 230, 235, 235, 235, 235, 235 ])
# variável alvo
y = np.array([ 137, 137, 137, 136, 135, 135, 133, 132, 133, 133, 128, 124, 126, 129, 126, 122, 122, 122, 119, 122 ])
# é necessário adicionar uma constante a matriz X
X_sm = sm.add_constant(X)
# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo
results = sm.OLS(y, X_sm).fit()
# mostrando as estatísticas do modelo
results.summary()
# mostrando as previsões para o mesmo conjunto passado
results.predict(X_sm)
df = pd.DataFrame()
df['x'] = X
df['y'] = y
# passando os valores de x e y como Dataframes
x_v = df[['x']]
y_v = df[['y']]
# criando e treinando o modelo
model = LinearRegression()
model.fit(x_v, y_v)
# para visualizar os coeficientes encontrados
model.coef_
# para visualizar o R²
model.score()
# mostrando as previsões para o mesmo conjunto passado
model.predict(X_sm)


class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.formula = None
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.X = X
        self.y = y
        soma_xy = sum(X * y)
        soma_x_ao_quadrado = sum(X * X)
        soma_x = sum(X)
        soma_y = sum(y)
        n = len(X)
        media_x = X.mean()
        media_y = y.mean()
        
        # build formula y = ax + b
        a = ( soma_xy - n * media_x * media_y ) / ( soma_x_ao_quadrado - n * ( media_x ** 2 ) )
        b = media_y - (a * media_x)
        
        self.coef_ = np.array([ b ])
        self.intercept_ = np.array([ a ])
        
        self.formula = lambda _x : (a * _x) + b
    
    def predict(self, x):
        return np.array(list(map(self.formula, x)))
    
    # fonte: https://edisciplinas.usp.br/pluginfile.php/1479289/mod_resource/content/0/regr_lin.pdf
    def sum_total_quadratic(self):
        median = self.y.mean()
        return sum( ( y - median ) ** 2 )
    
    def sum_error_quadratic(self):
        predicted = self.predict(x=self.X)
        return sum( ( self.y - predicted ) ** 2 )

    def regression_quadratic_sum(self):
        return self.sum_total_quadratic() - self.sum_error_quadratic()
    
    def score(self):
        return self.regression_quadratic_sum() / self.sum_total_quadratic()
