import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Carregar o dataset
dataset = pd.read_csv('gym_members_exercise_tracking.csv', index_col=False)

# Verificar informações básicas do dataset
print(dataset.info())

# Exibir as colunas do dataset
print(dataset.columns)

# Normalizar colunas numéricas
scaler = preprocessing.MinMaxScaler()
colunas_para_normalizar = [
    'Weight (kg)', 'Max_BPM', 'Avg_BPM',
    'Resting_BPM', 'Calories_Burned', 'Fat_Percentage', 'BMI'
]
dataset[colunas_para_normalizar] = scaler.fit_transform(dataset[colunas_para_normalizar])

# Transformar colunas categóricas em valores numéricos usando OrdinalEncoder
columns = dataset.select_dtypes(include=['object', 'bool']).columns
ordinal = OrdinalEncoder()
dataset[columns] = ordinal.fit_transform(dataset[columns])

# Separar as features (X) e o target (y)
x = dataset.drop(['Workout_Type'], axis=1)
y = dataset['Workout_Type']

# Dividir os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Criar e treinar o modelo de regressão logística
model_classification = LogisticRegression(max_iter=1000)
model_classification.fit(x_train, y_train)

# Fazer previsões com o modelo
y_predict = model_classification.predict(x_test)

# Avaliar o modelo com classification_report
report = classification_report(y_test, y_predict, target_names=['Cardio', 'HIIT', 'Strength', 'Yoga'])
print(report)
