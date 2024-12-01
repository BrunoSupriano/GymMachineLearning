import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configurar a página para usar o layout "wide"
st.set_page_config(layout="wide")

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

# Preprocessamento dos dados
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male: 1, Female: 0

# Criar nova coluna 'BMI'
df['BMI'] = df.apply(lambda row: calcular_bmi(row['Weight (kg)'], row['Height (m)']), axis=1)

# Remover colunas desnecessárias
columns_to_drop = ['Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage',
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'Max_BPM']
df.drop(columns=columns_to_drop, inplace=True)

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

# Streamlit App
st.title("Gerador de Treinos 🏋️")

# Layout de colunas
col1, col2 = st.columns(2)

with col1:
    # Inputs do usuário
    idade = st.number_input("Idade", min_value=1, max_value=100, value=25, key='idade', format='%d')
    genero_input = st.selectbox("Gênero", options=['Male', 'Female'], key='genero')
    peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, key='peso', format='%.2f')
    altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, key='altura', format='%.2f')
    intensidade_treino = st.selectbox("Intensidade do Treino", options=['Baixa', 'Média', 'Alta'], key='intensidade')
    gerar_treino = st.button("Gerar Treino")

if gerar_treino:
    # Calcular BMI
    bmi = calcular_bmi(peso, altura)

    # Definir frequência cardíaca recomendada
    frequencia_recomendada = definir_avg_bpm(intensidade_treino, genero_input)

    # Definir valor médio de BPM
    avg_bpm = definir_avg_bpm(intensidade_treino, genero_input)

    # Converter gênero para valor numérico
    genero = encoder.transform([genero_input])[0]

    # Preparar dados para a previsão
    novo_dado = pd.DataFrame({"Age": [idade], "Gender": [genero], "BMI": [bmi], "Avg_BPM": [avg_bpm]})
    novo_dado_scaled = scaler.transform(novo_dado)

    # Fazer a previsão
    predicted_workout = model.predict(novo_dado_scaled)

    with col2:
        # Exibir resultados
        # Streamlit App
        st.title("Treino Recomendado")
        st.markdown(f"**Tipo de treino recomendado:** {predicted_workout[0]}")
        st.markdown(f"**IMC:** {bmi:.2f}")
        st.markdown(f"**Freq. Cardíaca Média:** {frequencia_recomendada}")

        # Calcular importância das features
        importances = model.feature_importances_
        feature_names = X.columns