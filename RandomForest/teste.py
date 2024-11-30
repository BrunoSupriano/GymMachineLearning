import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Função para calcular o BMI
def calcular_bmi(peso, altura):
    return peso / (altura ** 2)

# Função para definir a frequência cardíaca recomendada com base na intensidade e gênero
def definir_frequencia_recomendada(intensidade, genero):
    if intensidade == 'Baixa':
        return 'Menos de 120 BPM' if genero == 'Male' else 'Menos de 110 BPM'
    elif intensidade == 'Média':
        return '120 a 160 BPM' if genero == 'Male' else '110 a 150 BPM'
    elif intensidade == 'Alta':
        return 'Acima de 160 BPM' if genero == 'Male' else 'Acima de 150 BPM'
    else:
        return 'Intensidade inválida'

# Carregar dados do arquivo CSV
csv_file = "gym_members_exercise_tracking.csv"  # Substitua pelo caminho do seu arquivo
df = pd.read_csv(csv_file)

# Preprocessamento dos dados
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male: 1, Female: 0

# Criar nova coluna 'BMI'
df['BMI'] = df.apply(lambda row: calcular_bmi(row['Weight (kg)'], row['Height (m)']), axis=1)

# Selecionar colunas relevantes para o modelo
X = df[["Age", "Gender", "BMI"]]
y = df["Workout_Type"]

# Padronizar os dados de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# Valores manuais
idade = 60
genero_input = 'Male'
peso = 70.0
altura = 1.75
intensidade_treino = 'Média'

# Converter gênero para valor numérico
genero = encoder.transform([genero_input])[0]

# Calcular BMI
bmi = calcular_bmi(peso, altura)

# Definir frequência cardíaca recomendada
frequencia_recomendada = definir_frequencia_recomendada(intensidade_treino, genero_input)

# Preparar dados para a previsão
novo_dado = pd.DataFrame({"Age": [idade], "Gender": [genero], "BMI": [bmi]})
novo_dado_scaled = scaler.transform(novo_dado)

# Fazer a previsão
predicted_workout = model.predict(novo_dado_scaled)

# Exibir os resultados
print(f"\n--- Resultado da Previsão ---")
print(f"Idade: {idade} anos")
print(f"Gênero: {genero_input}")
print(f"Peso: {peso} kg")
print(f"Altura: {altura} m")
print(f"BMI: {bmi:.2f}")
print(f"Intensidade de treino desejada: {intensidade_treino}")
print(f"Frequência Cardíaca Recomendada: {frequencia_recomendada}")
print(f"Tipo de treino recomendado: {predicted_workout[0]}")