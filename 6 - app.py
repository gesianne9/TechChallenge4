import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="SADC - Risco de Obesidade",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DASHBOARD POWER BI


LINK_DO_POWER_BI = "https://app.powerbi.com/view?r=eyJrIjoiMTkyMmU1ZWItODgyZi00MDczLTk4ODgtODA4M2ZiNTAxYzRlIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9&pageName=9b7751db79ee408bd0d6"

# ==============================================================================

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load('modelo_obesidade.pkl')
        encoder = joblib.load('encoder_target.pkl')
        return pipeline, encoder
    except FileNotFoundError:
        return None, None

pipeline, label_encoder = load_assets()

# --- TÍTULO DO SISTEMA ---
st.title("🏥 Sistema de Apoio à Decisão Clínica (SADC): Risco de Obesidade")
st.markdown("**Análise de risco multiclasse baseada em dados biométricos e estilo de vida.**")

# --- CRIAÇÃO DAS DUAS ABAS PRINCIPAIS ---
tab1, tab2 = st.tabs(["🔮 Simulador Preditivo (IA)", "📊 Dashboard Analítico (BI)"])

# ==============================================================================
# ABA 1: O MODELO DE MACHINE LEARNING (PREDITOR)
# ==============================================================================
with tab1:
    st.markdown("### Preencha o Prontuário para Análise de Risco")
    
    # Dividindo a tela para inputs ficarem organizados
    col_input, col_result = st.columns([1, 1.5], gap="large")
    
    with col_input:
        with st.form("formulario_paciente"):
            st.markdown("**Dados Biométricos**")
            genero = st.selectbox("Gênero", ["Feminino", "Masculino"], index=None, placeholder="Selecione...")
            idade = st.number_input("Idade", min_value=0, max_value=120, value=None, placeholder="Digite a idade...")
            altura = st.number_input("Altura (m)", min_value=0.0, max_value=2.5, value=None, placeholder="Ex: 1.75")
            peso_int = st.number_input(
                "Peso (kg)",
                min_value=30, 
                max_value=200, 
                value=None, 
                step=1,
                placeholder="Ex: 70"
            )
            
            st.markdown("---")
            st.markdown("**Hábitos e Histórico**")
            hist_fam = st.selectbox("Histórico Familiar de Obesidade?", ["sim", "nao"], index=None, placeholder="Selecione...")
            favc = st.selectbox("Alimentos Calóricos Frequentes?", ["sim", "nao"], index=None, placeholder="Selecione...")
            vegetais = st.selectbox("Consumo de Vegetais", ["raramente", "as vezes", "sempre"], index=None, placeholder="Selecione...")
            refeicoes = st.slider("Quantidade de Refeições ao dia", min_value=1, max_value=4, value=1)
            lanches = st.selectbox("Lanches entre refeições", ["nao consome", "as vezes", "frequentemente", "sempre"], index=None, placeholder="Selecione...")
            fumante = st.selectbox("Fumante?", ["sim", "nao"], index=None, placeholder="Selecione...")
            agua = st.select_slider("Água (Litros/dia)",
                options=["Menos de 1L", "Entre 1L e 2L", "Mais de 2L"],
                value="Menos de 1L")
            monitora = st.selectbox("Monitora Calorias?", ["sim", "nao"], index=None, placeholder="Selecione...")
            atv_fisica = st.selectbox("Atividade Física", ["nenhuma", "1 a 2x/sem", "3 a 4x/sem", "5x/sem ou mais"], index=None, placeholder="Selecione...")
            eletronicos = st.selectbox("Tempo de Tela", ["0 a 2h/dia", "3 a 5h/dia", "> 5h/dia"], index=None, placeholder="Selecione...")
            alcool = st.selectbox("Álcool", ["nao bebe", "as vezes", "frequentemente", "sempre"], index=None, placeholder="Selecione...")
            transporte = st.selectbox("Transporte", ["transporte publico", "a pe", "carro", "moto", "bicicleta"], index=None, placeholder="Selecione...")

            btn_calcular = st.form_submit_button("🔍 Calcular Risco")

    with col_result:
        if btn_calcular:
            # 1. Lista com TODAS as variáveis de input para validação
            campos_obrigatorios = [
                genero, idade, altura, peso_int, hist_fam, favc, vegetais, 
                refeicoes, lanches, fumante, agua, monitora, 
                atv_fisica, eletronicos, alcool, transporte
            ]

            # 2. TRAVA DE SEGURANÇA: Verifica se algum campo é Vazio
            if None in campos_obrigatorios:
                st.warning("⚠️ **Atenção:** Existem campos não preenchidos no formulário. Por favor, responda a todas as perguntas antes de calcular.")
            
            else:
                # 3. Se tudo estiver preenchido, segue para a predição
                if pipeline:
                    
                    # --- TRAMENTO DE DADOS 
                    

                    # Peso: Garante que o número inteiro vire decimal (70 -> 70.0)
                    peso_final = float(peso_int)

                    # Aqui transformamos o texto do slider no número que o modelo entende
                    mapa_agua = {
                        "Menos de 1L": 1,
                        "Entre 1L e 2L": 2,
                        "Mais de 2L": 3
                    }
                    
                    # Pegamos o valor numérico correspondente
                    agua_final = mapa_agua[agua]        
                    
                    
                    # Criar DataFrame com os dados
                    input_df = pd.DataFrame({
                        'genero': [genero], 'idade': [idade], 'altura': [altura], 'peso_kg': [peso_final],
                        'hist_familiar_excesso': [hist_fam], 'consumo_calorico': [favc], 
                        'consumo_vegetais': [vegetais], 'numero_refeicoes': [refeicoes], 
                        'lanches_entre_refeicoes': [lanches], 'fumante': [fumante], 
                        'consumo_agua_dia': [agua_final], 'monitora_calorias': [monitora], 
                        'atividade_fisica': [atv_fisica], 'tempo_eletronicos': [eletronicos], 
                        'consumo_alcool': [alcool], 'tipo_transporte': [transporte]
                    })
                    
                    # Predição
                    probs = pipeline.predict_proba(input_df)[0]
                    classes = label_encoder.classes_
                    
                    # Organizar Resultados
                    df_res = pd.DataFrame({'Classificação': classes, 'Probabilidade': probs})
                    df_res = df_res.sort_values(by='Probabilidade', ascending=False).reset_index(drop=True)
                    
                    vencedor = df_res.iloc[0]['Classificação']
                    prob_vencedor = df_res.iloc[0]['Probabilidade']
                    
                    # Exibição
                    st.info(f"Diagnóstico mais provável: **{vencedor.upper()}** ({prob_vencedor:.1%})")
                    
                    # Gráfico e Tabela
                    st.dataframe(
                        df_res,
                        column_config={
                            "Classificação": "Diagnóstico",
                            "Probabilidade": st.column_config.ProgressColumn(
                                "Probabilidade", format="%.2f%%", min_value=0, max_value=1
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Alerta de Risco
                    soma_obesidade = df_res[df_res['Classificação'].str.contains('obesidade')]['Probabilidade'].sum()
                    if soma_obesidade > 0.5:
                        st.error(f"⚠️ Atenção: Risco acumulado de Obesidade é de **{soma_obesidade:.1%}**.")
                else:
                    st.error("Erro: Modelo não carregado corretamente.")
        else:
            st.markdown("👈 **Preencha o formulário ao lado e clique em Calcular Risco.**")
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=200, caption="IA Médica")




# ABA 2: O DASHBOARD DO POWER BI

with tab2:
    st.markdown("### 📊 Indicadores de Saúde Populacional")
    st.markdown("Visualização interativa dos dados históricos utilizados no estudo.")
    
    # Aqui fazemos a mágica da incorporação (Embed)
    if LINK_DO_POWER_BI:
        st.markdown(
            f'<iframe title="Dashboard Obesity" width="100%" height="600" src="{LINK_DO_POWER_BI}" frameborder="0" allowFullScreen="true"></iframe>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ O link do Power BI ainda não foi configurado no código.")



# ==============================================================================
# RODAPÉ (FOOTER)
# ==============================================================================
st.markdown("---") # Linha divisória

# Layout em colunas para ficar organizado
col_info, col_dev = st.columns([2, 1])

# --- Coluna da Esquerda: Sobre o Projeto ---
with col_info:
    st.subheader("📌 Sobre o Projeto")
    st.markdown("""
    Este sistema foi desenvolvido como parte do **Tech Challenge (Fase 4)** da pós graduação em Data Analytics da FIAP + Alura.
    
    **Desafio:** Criar uma solução de Machine Learning capaz de analisar fatores de risco 
    clínicos e comportamentais para auxiliar no diagnóstico precoce da obesidade, integrada a um painel analítico que transforma esses dados em insights visuais para a equipe médica.         
    """)

# --- Coluna da Direita: Desenvolvedora ---
with col_dev:
    st.subheader("👩‍💻 Desenvolvido por")
    
    # AJUSTE 1: Mudei a proporção para [0.6, 2]. 
    # O 0.6 deixa a coluna da foto bem estreita, "puxando" o texto para a esquerda.
    c_img, c_txt = st.columns([0.6, 2])
    
    with c_img:
        # Foto do GitHub
        st.image("https://github.com/gesianne9.png", width=90) 
    
    with c_txt:
        # AJUSTE 2: HTML para controlar tamanho e opacidade
        st.markdown("""
        <div style='margin-top: 5px;'>
            <span style='font-size: 18px; font-weight: bold;'>Gesianne de Azevedo Ferreira</span>
            <br>
            <span style='font-size: 14px; color: rgba(250, 250, 250, 0.6);'>Cientista de Dados em Formação</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Badges (Links)
        st.markdown("""
        <div style='margin-top: 10px;'>
            <a href='https://www.linkedin.com/in/gesianne-azevedo/'><img src='https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white' height='20'></a>
            <a href='https://github.com/gesianne9'><img src='https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white' height='20'></a>
        </div>
        """, unsafe_allow_html=True)

# Copyright simples no final
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 12px;'>© 2024 Health Analytics. Todos os direitos reservados.</div>", 
    unsafe_allow_html=True
)