import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar dados do arquivo CSV
csv_file = "gym_members_exercise_tracking.csv"  # Substitua pelo caminho do seu arquivo
df = pd.read_csv(csv_file)

# Preprocessamento
# Codificar 'Gender' em valores numéricos
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male: 1, Female: 0

# Selecionar colunas relevantes (features) e o target (Workout_Type)
X = df[["Age", "Gender", "Weight (kg)", "Height (m)"]]
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

# Prever para novos dados fornecidos
new_data = pd.DataFrame({
    "Age": [40],
    "Gender": [encoder.transform(["Female"])[0]],  # Converter 'Female' para valor numérico
    "Weight (kg)": [70],
    "Height (m)": [1.60]
})

# Padronizar os novos dados
new_data_scaled = scaler.transform(new_data)

# Fazer previsão
predicted_workout = model.predict(new_data_scaled)
print(f"Workout recomendado: {predicted_workout[0]}")
