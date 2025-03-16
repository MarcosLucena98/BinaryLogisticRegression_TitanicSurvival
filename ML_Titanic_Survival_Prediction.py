# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:14:20 2025

@author: Marcos
"""

# In[0.1]: Package Installation

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Package Importation

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo

import warnings
warnings.filterwarnings('ignore')    

# In[0.3]: Impotando os dados

df_train = pd.read_csv('train.csv')

df_train.describe()
# In[0.4]: Analise descritiva

df_train.info()

df_train.describe()

df_train['Survived'].value_counts().sort_index()

# Transformação do 'Passengerid' para o tipo 'str'
df_train['PassengerId'] = df_train['PassengerId'].astype('str')

df_train['Cabin'] = df_train['Cabin'].notna().astype(int)

df_train.info()

# In[0.5] Tabela de frequencias absolutas das variaveis qualitativas

df_train['Sex'].value_counts().sort_index()
df_train['Embarked'].value_counts().sort_index()
df_train['Pclass'].value_counts().sort_index()
df_train['SibSp'].value_counts().sort_index()
df_train['Parch'].value_counts().sort_index()

# In[0.6]: Vamos Dummizar as variaveis Sex, Embarked, PClas, SibSp e Parch

# df_train['SibSp'] = df_train['SibSp'].between(0, 4).astype(int)

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())



# In[0.6]: Vamos Dummizar as variaveis Sex, Embarked, PClas, SibSp e Parch

df_titanic_dummies = pd.get_dummies(df_train,
                                    columns=['Sex',
                                             'Embarked',
                                             'Pclass',
                                             'SibSp',
                                             'Parch'],
                                    dtype=int,
                                    drop_first=True)

df_titanic_dummies.info()
# In[0.7]: Estimando o modelo

lista_dummies = list(df_titanic_dummies.drop(columns=['PassengerId', 
                                                      'Name', 
                                                      'Ticket',
                                                      'Survived']))

formula_titanic_model = ' + '.join(lista_dummies)
formula_titanic_model = "Survived ~ " + formula_titanic_model
print(formula_titanic_model)

# In[0.8] modelo

modelo_titanic = sm.Logit.from_formula(formula_titanic_model,
                                       df_titanic_dummies).fit()

modelo_titanic.summary()

# In[3.6]: Procedimento Stepwise

#Estimação do modelo por meio do procedimento Stepwise
step_modelo_titanic = stepwise(modelo_titanic, pvalue_limit=0.05)

# In[3.6]: Procedimento

# Filtrando as variáveis com p-value > 0.05
p_values = modelo_titanic.pvalues[1:]  # Ignorar o intercepto
cols_to_remove = p_values[p_values > 0.05].index.tolist()

# Remover as colunas com p-value > 0.05
df_cleaned = df_titanic_dummies.drop(columns=cols_to_remove)

lista_dummies_new = list(df_cleaned.drop(columns=['Survived',
                                                  'Name',
                                                  'Ticket',
                                                  'PassengerId']))

formula_titanic_model_new = ' + '.join(lista_dummies_new)
formula_titanic_model_new = "Survived ~ " + formula_titanic_model_new
print(formula_titanic_model_new)

# In[]
modelo_titanic_new = sm.Logit.from_formula(formula_titanic_model_new,
                                       df_cleaned).fit()

modelo_titanic_new.summary()


# In[]

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# In[]

# Adicionando os valores previstos de probabilidade na base de dados
df_titanic_dummies['phat'] = modelo_titanic_new.predict()

# In[]
# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_titanic_dummies['Survived'],
                predicts=df_titanic_dummies['phat'],
                cutoff=0.5)

# In[0.3]: Impotando os dados

df_test = pd.read_csv('test.csv')

df_test.describe()

# In[0.4]: Analise descritiva

# Transformação do 'Passengerid' para o tipo 'str'
df_test['PassengerId'] = df_test['PassengerId'].astype('str')

df_test['Cabin'] = df_train['Cabin'].notna().astype(int)

df_test.info()

# In[0.6]: Vamos Dummizar as variaveis Sex, Embarked, PClas, SibSp e Parch

# df_train['SibSp'] = df_train['SibSp'].between(0, 4).astype(int)

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())

df_test.info()

# In[0.6]: Vamos Dummizar as variaveis Sex, Embarked, PClas, SibSp e Parch
df_test_dummies = pd.get_dummies(df_test,
                                    columns=['Sex',
                                             'Embarked',
                                             'Pclass',
                                             'SibSp',
                                             'Parch'],
                                    dtype=int,
                                    drop_first=True)

# In[0.6]: 

# 3. Garantir que as colunas do teste sejam iguais às do treino
df_test_dummies = df_test_dummies.reindex(columns=df_titanic_dummies.columns, fill_value=0)


# In[0.6]: Vamos Dummizar as variaveis Sex, Embarked, PClas, SibSp e Parch

# Lista de colunas a serem mantidas
colunas_desejadas = ['Age', 'Cabin', 'Sex_male', 'Pclass_3', 'SibSp_3', 'SibSp_4']

# Garantir que apenas as colunas da lista sejam mantidas e remover 'Survived' se existir
df_test_dummies = df_test_dummies.loc[:, colunas_desejadas]

# In[]
# 4. Fazer previsões
predictions = modelo_titanic_new.predict(df_test_dummies)

# In[]
# Converter as probabilidades em classes (0 ou 1) usando 0.5 como limite
df_test['Survived'] = (predictions > 0.5).astype(int)

# In[]
submission = df_test[['PassengerId']].copy()
submission['Survived'] = df_test['Survived'].copy()

# Salvar o arquivo CSV corretamente
submission.to_csv('submission.csv', index=False, sep=",")
