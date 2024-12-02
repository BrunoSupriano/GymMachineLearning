# Certifique-se de ter o Streamlit instalado: pip install streamlit
# Para rodar o aplicativo, use: python -m streamlit run app.py

import streamlit as st

# Configuração inicial da página
st.set_page_config(page_title="Previsão de Treinos", page_icon="🏋️", layout="wide")

# Título do aplicativo
st.title("🏋️ Previsão de Treinos Personalizados")

# Introdução
st.write("""
Insira suas informações abaixo para receber previsões sobre os treinos mais adequados ao seu perfil, incluindo:
- Melhor tipo de treino
- Frequência cardíaca ideal
- Calorias estimadas a serem gastas
""")

# Criação das colunas
col1, col2 = st.columns([1, 1])

# Entrada de dados na coluna da esquerda
with col1:
    st.header("Informações do Usuário")
    nome = st.text_input("Nome", placeholder="Digite seu nome")
    idade = st.number_input("Idade", min_value=1, max_value=120, step=1)
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
    peso = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, step=0.1)
    altura = st.number_input("Altura (cm)", min_value=50.0, max_value=250.0, step=0.1)
    calcular = st.button("Calcular")

# Resultados na coluna da direita
with col2:
    if calcular:
        st.write("Processando os dados...")

        # Simula o resultado (integre com o modelo de machine learning aqui)
        st.subheader("Resultados")
        st.write("Nome:", nome)
        st.write("Idade:", idade, "anos")
        st.write("Sexo:", sexo)
        st.write("Peso:", f"{peso} kg")
        st.write("Altura:", f"{altura} cm")
        
        # Exemplo de integração futura
        st.success("Integre com o modelo de machine learning para calcular os treinos ideais, batimentos e calorias!")
    else:
        st.info("Preencha os dados e clique no botão para calcular.")

# Nota final
st.caption("© 2024 Previsão de Treinos - Todos os direitos reservados.")
