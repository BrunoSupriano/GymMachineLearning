import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Função para calcular o BMI
def calcular_bmi(peso, altura):
    return peso / (altura ** 2)

# Função para definir o valor médio de BPM com base na intensidade do treino
def definir_avg_bpm(intensidade, genero):
    if intensidade == 'Baixa':
        return 110 if genero == 'Male' else 100
    elif intensidade == 'Média':
        return 140 if genero == 'Male' else 130
    elif intensidade == 'Alta':
        return 180 if genero == 'Male' else 170
    else:
        return 0

# Carregar dados do arquivo CSV
csv_file = "../gym_members_exercise_tracking.csv"  # Substitua pelo caminho do seu arquivo
df = pd.read_csv(csv_file)

print(df.columns)

# Classificar os valores de Workout_Type e contar as ocorrências de cada
workout_type_counts = df['Workout_Type'].value_counts()
print("Contagem de ocorrências de cada Workout_Type:")
print(workout_type_counts)

# Análise exploratória dos dados
print("inicio")
numeric_df = df.select_dtypes(include=[float, int])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)
print(df['Workout_Type'].value_counts())

# Boxplot para verificar a presença de outliers
sns.boxplot(x='Workout_Type', y='Age', data=df)
print("Fim")

# Agrupar os dados por atividade e calcular a idade média
print(df.groupby('Workout_Type')['Age'].describe())
print(df[df['Age'] >= 50].groupby('Workout_Type').size())

# Analisar a distribuição de idade por tipo de exercício
sns.boxplot(x='Workout_Type', y='Age', data=df)
plt.title('Distribuição de Idade por Tipo de Exercício')
plt.show()

# Preprocessamento dos dados
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male: 1, Female: 0
df['Workout_Type'] = encoder.fit_transform(df['Workout_Type'])  # Codificar Workout_Type

# Criar nova coluna 'BMI'
df['BMI'] = df.apply(lambda row: calcular_bmi(row['Weight (kg)'], row['Height (m)']), axis=1)

# Remover colunas desnecessárias
columns_to_drop = ['Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'Max_BPM']
df.drop(columns=columns_to_drop, inplace=True)

# Histograma para cada variável numérica
df.hist(figsize=(12, 8))
plt.show()

# Selecionar colunas relevantes para o modelo
X = df[["Age", "Gender", "BMI", "Avg_BPM"]]
y = df["Workout_Type"]

# Padronizar os dados de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Correlação entre as variáveis (incluindo Workout_Type)
numeric_df = df.select_dtypes(include=[float, int])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlação")
plt.show()

# Visualização das importâncias das features
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Importância das Features")
plt.show()

# Mostrar as classes associadas ao LabelEncoder
print("Classes associadas ao 'Workout_Type':")
print(encoder.classes_)

# Matriz de Confusão com nomes das classes
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Relatório de Classificação com as classes associadas
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
