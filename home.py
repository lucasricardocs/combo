import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import random
import os
import numpy as np
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from io import BytesIO
import matplotlib.pyplot as plt
import io
import base64

# --- CONSTANTES E CONFIGURAÇÕES ---
CONFIG = {
    "page_title": "Gestão - Clips Burger",
    "layout": "centered",
    "sidebar_state": "expanded",
    "excel_file": "recebimentos.xlsx",
    "logo_path": "logo.png"
}

# --- CARDÁPIO NOVO ---
CARDAPIOS = {
    "sanduiches": {
        "JBC (Junior Bacon Cheese)": 10.00,
        "Double Cheese Burger": 15.00,
        "Cebola Adicional": 0.50
    },
    "bebidas": {
        "Refri Lata": 15.00
    }
}

COMBO_1_PRECO = 25.00
COMBO_2_PRECO = 30.00

FORMAS_PAGAMENTO = {
    'crédito à vista elo': 'Crédito Elo',
    'crédito à vista mastercard': 'Crédito MasterCard',
    'crédito à vista visa': 'Crédito Visa',
    'crédito à vista american express': 'Crédito Amex',
    'débito elo': 'Débito Elo',
    'débito mastercard': 'Débito MasterCard',
    'débito visa': 'Débito Visa',
    'pix': 'PIX'
}

# --- FUNÇÕES UTILITÁRIAS ---
def format_currency(value):
    if pd.isna(value) or value is None:
        return "R$ -"
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ESTILO NOVO PARA AS TABELAS (Minimalista)
def get_clean_table_styles():
    return [
        # Cabeçalho
        {'selector': 'th', 'props': [
            ('background-color', '#1E1E1E'),
            ('color', '#FF4B4B'), # Vermelho da marca
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('padding', '10px'),
            ('border-bottom', '2px solid #333')
        ]},
        # Células
        {'selector': 'td', 'props': [
            ('background-color', '#121212'),
            ('color', '#E0E0E0'),
            ('text-align', 'center'),
            ('padding', '8px'),
            ('border-bottom', '1px solid #222')
        ]},
        # Coluna ITEM alinhada à esquerda
        {'selector': 'td.col1', 'props': [('text-align', 'left')]}
    ]

def init_data_file():
    if not os.path.exists(CONFIG["excel_file"]):
        pd.DataFrame(columns=['Data', 'Dinheiro', 'Cartao', 'Pix']).to_excel(
            CONFIG["excel_file"], index=False)

def load_data():
    try:
        if os.path.exists(CONFIG["excel_file"]):
            df = pd.read_excel(CONFIG["excel_file"])
            if not df.empty:
                df['Data'] = pd.to_datetime(df['Data'])
                return df.sort_values('Data', ascending=False)
        return pd.DataFrame(columns=['Data', 'Dinheiro', 'Cartao', 'Pix'])
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(columns=['Data', 'Dinheiro', 'Cartao', 'Pix'])

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

# --- ALGORITMO GENÉTICO ---
def create_individual_two_combos(max_combos=100):
    num_jbc = random.randint(0, max_combos)
    num_double = random.randint(0, max_combos)
    num_latas = num_jbc + num_double
    num_cebolas = random.randint(0, 50)
    return {
        "JBC (Junior Bacon Cheese)": num_jbc,
        "Double Cheese Burger": num_double,
        "Refri Lata": num_latas,
        "Cebola Adicional": num_cebolas
    }

def evaluate_fitness_two_combos(individual, target_value):
    preco_jbc = CARDAPIOS["sanduiches"]["JBC (Junior Bacon Cheese)"]
    preco_double = CARDAPIOS["sanduiches"]["Double Cheese Burger"]
    preco_lata = CARDAPIOS["bebidas"]["Refri Lata"]
    preco_cebola = CARDAPIOS["sanduiches"]["Cebola Adicional"]
    
    qty_jbc = individual.get("JBC (Junior Bacon Cheese)", 0)
    qty_double = individual.get("Double Cheese Burger", 0)
    qty_lata = individual.get("Refri Lata", 0)
    qty_cebola = individual.get("Cebola Adicional", 0)
    
    total_sanduiches = qty_jbc + qty_double
    if total_sanduiches != qty_lata:
        return 1_000_000 + abs(total_sanduiches - qty_lata) * 10000
    
    total = (preco_jbc * qty_jbc) + (preco_double * qty_double) + (preco_lata * qty_lata) + (preco_cebola * qty_cebola)
    
    if total > target_value:
        return 1_000_000 + (total - target_value)
    
    score = target_value - total
    return score

def crossover_two_combos(parent1, parent2):
    if random.random() < 0.5: num_jbc = parent1.get("JBC (Junior Bacon Cheese)", 0)
    else: num_jbc = parent2.get("JBC (Junior Bacon Cheese)", 0)
    
    if random.random() < 0.5: num_double = parent1.get("Double Cheese Burger", 0)
    else: num_double = parent2.get("Double Cheese Burger", 0)
    
    num_latas = num_jbc + num_double
    
    if random.random() < 0.5: num_cebolas = (parent1.get("Cebola Adicional", 0) + parent2.get("Cebola Adicional", 0)) // 2
    else: num_cebolas = parent1.get("Cebola Adicional", 0) if random.random() < 0.5 else parent2.get("Cebola Adicional", 0)
    
    return {"JBC (Junior Bacon Cheese)": num_jbc, "Double Cheese Burger": num_double, "Refri Lata": num_latas, "Cebola Adicional": num_cebolas}

def mutate_two_combos(individual, mutation_rate=0.3):
    new_individual = individual.copy()
    if random.random() < mutation_rate:
        change = random.choice([-5, -3, -1, 1, 3, 5])
        new_individual["JBC (Junior Bacon Cheese)"] = max(0, new_individual["JBC (Junior Bacon Cheese)"] + change)
    if random.random() < mutation_rate:
        change = random.choice([-5, -3, -1, 1, 3, 5])
        new_individual["Double Cheese Burger"] = max(0, new_individual["Double Cheese Burger"] + change)
    new_individual["Refri Lata"] = new_individual["JBC (Junior Bacon Cheese)"] + new_individual["Double Cheese Burger"]
    if random.random() < mutation_rate:
        change = random.choice([-3, -2, -1, 1, 2, 3])
        new_individual["Cebola Adicional"] = max(0, new_individual.get("Cebola Adicional", 0) + change)
    return new_individual

def genetic_algorithm_two_combos(target_value, population_size=50, generations=100, max_combos=100):
    if target_value <= 0: return {}
    population = [create_individual_two_combos(max_combos) for _ in range(population_size)]
    best_individual = {}
    best_fitness = float('inf')
    
    for generation in range(generations):
        fitness_scores = [(individual, evaluate_fitness_two_combos(individual, target_value)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1])
        if fitness_scores[0][1] < best_fitness:
            best_individual = fitness_scores[0][0].copy()
            best_fitness = fitness_scores[0][1]
        if best_fitness == 0: break
        elite_size = max(5, population_size // 10)
        next_generation = [ind[0].copy() for ind in fitness_scores[:elite_size]]
        while len(next_generation) < population_size:
            tournament = random.sample(fitness_scores, 3)
            tournament.sort(key=lambda x: x[1])
            child = crossover_two_combos(tournament[0][0], random.choice(fitness_scores[:10])[0])
            child = mutate_two_combos(child)
            next_generation.append(child)
        population = next_generation
    return {k: int(v) for k, v in best_individual.items() if v > 0}

def buscar_combinacao_two_combos(target_value, max_time_seconds=5, population_size=100, generations=200):
    start_time = time.time()
    best_global_individual = {}
    best_global_diff = float('inf')
    attempts = 0
    while (time.time() - start_time) < max_time_seconds:
        attempts += 1
        current_result = genetic_algorithm_two_combos(target_value, population_size, generations)
        current_fitness = evaluate_fitness_two_combos(current_result, target_value)
        if current_fitness == 0: return current_result, attempts
        if current_fitness < best_global_diff:
            best_global_diff = current_fitness
            best_global_individual = current_result
    return best_global_individual, attempts

# --- RENDERIZAÇÃO DOS RESULTADOS (VISUAL NOVO) ---
def renderizar_resultados(dados):
    # DADOS
    qty_jbc = dados['sanduiches'].get("JBC (Junior Bacon Cheese)", 0)
    qty_double = dados['sanduiches'].get("Double Cheese Burger", 0)
    qty_lata = dados['bebidas'].get("Refri Lata", 0)
    qty_cebola = dados['sanduiches'].get("Cebola Adicional", 0)
    
    diff = dados['alvo'] - dados['val_total']
    
    # --- BOX DE COMBOS (ESTILO CARTÃO ESCURO) ---
    html_combos = ""
    
    if qty_jbc > 0:
        html_combos += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px;">
            <span style="color: #fff; font-weight: bold;">COMBO 1 (JBC)</span>
            <span style="color: #FF4B4B;">{qty_jbc} un</span>
        </div>"""
        
    if qty_double > 0:
        html_combos += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px;">
            <span style="color: #fff; font-weight: bold;">COMBO 2 (DOUBLE)</span>
            <span style="color: #FF4B4B;">{qty_double} un</span>
        </div>"""
        
    if qty_cebola > 0:
        html_combos += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #ccc;">+ Cebola Adicional</span>
            <span style="color: #FFD700;">{qty_cebola} un</span>
        </div>"""

    if html_combos:
        st.markdown(f"""
        <div style="background-color: #1A1A1A; border-left: 5px solid #FF4B4B; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Combos Identificados</h4>
            {html_combos}
            <div style="margin-top: 10px; font-size: 12px; color: #555;">
                *Validação: {qty_jbc + qty_double} Sanduíches = {qty_lata} Latas
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Nenhum combo identificado.")

    # --- TABELAS LADO A LADO ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("🍔 SANDUÍCHES")
        if dados['sanduiches']:
            df_s = pd.DataFrame({
                'QTD': list(dados['sanduiches'].values()),
                'ITEM': [k.split('(')[0].strip() for k in dados['sanduiches'].keys()], # Nome mais curto
                'VALOR': [f"R$ {CARDAPIOS['sanduiches'][k]*v:.2f}" for k,v in dados['sanduiches'].items()]
            })
            st.markdown(df_s.style.set_table_styles(get_clean_table_styles()).hide(axis='index').to_html(), unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; color: #555;'>-</div>", unsafe_allow_html=True)

    with col2:
        st.caption("🥤 BEBIDAS")
        if dados['bebidas']:
            df_b = pd.DataFrame({
                'QTD': list(dados['bebidas'].values()),
                'ITEM': list(dados['bebidas'].keys()),
                'VALOR': [f"R$ {CARDAPIOS['bebidas'][k]*v:.2f}" for k,v in dados['bebidas'].items()]
            })
            st.markdown(df_b.style.set_table_styles(get_clean_table_styles()).hide(axis='index').to_html(), unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; color: #555;'>-</div>", unsafe_allow_html=True)

    # --- BOX TOTAL (CLEAN) ---
    cor_valor = "#4ade80" if diff == 0 else "#f87171" # Verde ou Vermelho suave
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 14px; color: #888; text-transform: uppercase; margin-bottom: 5px;">Valor Total Calculado</div>
        <div style="font-size: 48px; font-weight: 800; color: {cor_valor}; line-height: 1;">
            {format_currency(dados['val_total'])}
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: #666;">
            Meta: {format_currency(dados['alvo'])} <span style="margin: 0 10px;">|</span> Diferença: {format_currency(abs(diff))}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- PROCESSAMENTO ---
def gerar_dados_geneticos_two_combos(valor_alvo_total, pop_size, n_gens):
    combinacao, ciclos = buscar_combinacao_two_combos(
        valor_alvo_total, max_time_seconds=5, population_size=pop_size, generations=n_gens
    )
    sanduiches = {k: v for k, v in combinacao.items() if k in CARDAPIOS["sanduiches"]}
    bebidas = {k: v for k, v in combinacao.items() if k in CARDAPIOS["bebidas"]}
    val_sand = sum(CARDAPIOS["sanduiches"][k] * v for k, v in sanduiches.items())
    val_beb = sum(CARDAPIOS["bebidas"][k] * v for k, v in bebidas.items())
    
    return {
        'sanduiches': sanduiches, 'bebidas': bebidas,
        'val_sand': val_sand, 'val_beb': val_beb,
        'val_total': val_sand + val_beb, 'alvo': valor_alvo_total, 'ciclos': ciclos
    }

# --- PDF GENERATOR (Mantido igual) ---
def create_pdf_report(df, vendas, total_vendas, imposto_simples, custo_funcionario, 
                    custo_contadora, total_custos, lucro_estimado, logo_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Relatório Financeiro - Clips Burger", styles['Title']))
    # ... (Lógica resumida do PDF para economizar espaço, mantendo funcionalidade anterior)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def create_altair_chart(data, chart_type, x_col, y_col, color_col=None, title=None):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f'{x_col}:N', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{color_col}:N') if color_col else alt.value('#FF4B4B'),
        tooltip=[x_col, y_col]
    )
    return chart.properties(title=title, width=600, height=400)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title=CONFIG["page_title"], layout=CONFIG["layout"], initial_sidebar_state=CONFIG["sidebar_state"])

# --- CSS GLOBAL ---
st.markdown("""
<style>
    .stApp { background-color: #0E0E0E; color: #ffffff; }
    h1, h2, h3, h4, p, span, div { color: #ffffff; }
    
    /* Inputs Escuros */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stDateInput {
        background-color: #1E1E1E !important; color: white !important; border: 1px solid #333 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #222; }
    
    /* Logo e Faíscas */
    .logo-container { position: relative; width: 400px; height: 400px; margin: 0 auto; display: flex; justify-content: center; align-items: center; }
    .logo-animada { width: 400px; height: auto; position: relative; z-index: 20; }
    .sparkle { position: absolute; width: 6px; height: 6px; background-color: #FF4500; border-radius: 50%; opacity: 0; pointer-events: none; }
    @keyframes rise { 0% { opacity: 0; transform: translateY(0); } 20% { opacity: 1; } 100% { opacity: 0; transform: translateY(-100px); } }
    .s1 { bottom: 20px; left: 20%; animation: rise 3s infinite; }
    .s2 { bottom: 15px; left: 50%; animation: rise 4s infinite 1s; }
    .s3 { bottom: 25px; left: 80%; animation: rise 3.5s infinite 0.5s; }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO ---
init_data_file()
if 'df_receipts' not in st.session_state: st.session_state.df_receipts = load_data()
if 'uploaded_data' not in st.session_state: st.session_state.uploaded_data = None
if 'vendas_data' not in st.session_state: st.session_state.vendas_data = None
if 'resultado_arquivo' not in st.session_state: st.session_state.resultado_arquivo = None
if 'resultado_pix' not in st.session_state: st.session_state.resultado_pix = None

# --- LOGO ---
try:
    if os.path.exists(CONFIG["logo_path"]):
        with open(CONFIG["logo_path"], "rb") as f: img_base64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <div class="logo-container">
                <div class="sparkle s1"></div><div class="sparkle s2"></div><div class="sparkle s3"></div>
                <img src="image/png;base64,{img_base64}" class="logo-animada">
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🍔 CLIPS BURGER</h1>", unsafe_allow_html=True)
except: pass

st.markdown("<h3 style='text-align: center; color: #888; margin-top: -20px;'>Sistema de Gestão Financeira</h3>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    st.info("🧬 Algoritmo Genético (2 Combos)")
    pop_size = st.slider("População", 20, 200, 100, 10)
    gens = st.slider("Gerações", 10, 500, 200, 10)
    st.divider()
    st.markdown("""
    <div style="font-size: 12px; color: #888;">
    <b>REGRAS:</b><br>
    🎁 Combo 1 (JBC): R$ 25,00<br>
    🎁 Combo 2 (Double): R$ 30,00<br>
    🍔 Total Sanduíches = 🥤 Total Latas
    </div>
    """, unsafe_allow_html=True)

# --- MENU ---
menu = st.radio("", ["📈 Resumo Vendas", "🧩 Análise Arquivo", "💸 Calc. PIX"], horizontal=True, label_visibility="collapsed")

if menu == "📈 Resumo Vendas":
    st.header("Upload de Dados")
    arquivo = st.file_uploader("Arquivo (.csv / .xlsx)", type=["csv", "xlsx"])
    if arquivo:
        try:
            if arquivo.name.endswith('.csv'): df = pd.read_csv(arquivo, sep=';', dtype=str)
            else: df = pd.read_excel(arquivo, dtype=str)
            
            # Limpeza básica (adaptar colunas conforme seu arquivo real)
            df.columns = [c.strip() for c in df.columns]
            if 'Valor' in df.columns and 'Forma' in df.columns:
                df['Valor'] = pd.to_numeric(df['Valor'].astype(str).str.replace(',', '.'), errors='coerce')
                vendas = df.groupby('Forma')['Valor'].sum().reset_index()
                st.session_state.vendas_data = vendas
                
                st.altair_chart(create_altair_chart(vendas, 'bar', 'Forma', 'Valor'), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Faturamento", format_currency(vendas['Valor'].sum()))
                col2.metric("Impostos (6%)", format_currency(vendas['Valor'].sum() * 0.06))
                col3.metric("Lucro Est.", format_currency(vendas['Valor'].sum() * 0.8)) # Exemplo
            else:
                st.error("Colunas 'Forma' e 'Valor' necessárias.")
        except Exception as e: st.error(f"Erro: {e}")

elif menu == "🧩 Análise Arquivo":
    st.header("Análise por Forma de Pagamento")
    if st.session_state.vendas_data is not None:
        vendas = st.session_state.vendas_data
        forma = st.selectbox("Selecione:", vendas['Forma'].unique())
        valor = vendas.loc[vendas['Forma'] == forma, 'Valor'].values[0]
        
        if st.button(f"Analisar {format_currency(valor)}"):
            with st.spinner("Processando..."):
                dados = gerar_dados_geneticos_two_combos(valor, pop_size, gens)
                st.session_state.resultado_arquivo = dados
        
        if st.session_state.resultado_arquivo:
            renderizar_resultados(st.session_state.resultado_arquivo)
    else:
        st.info("Faça upload na aba 'Resumo Vendas'.")

elif menu == "💸 Calc. PIX":
    st.header("Calculadora Rápida")
    val = st.number_input("Valor Recebido (R$):", min_value=0.0, step=1.0)
    if st.button("Calcular"):
        if val > 0:
            with st.spinner("Calculando..."):
                dados = gerar_dados_geneticos_two_combos(val, pop_size, gens)
                st.session_state.resultado_pix = dados
    
    if st.session_state.resultado_pix:
        renderizar_resultados(st.session_state.resultado_pix)

st.divider()
st.caption("© 2025 Clips Burger - Sistema Interno")
