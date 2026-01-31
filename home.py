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
import urllib.request

# --- CONSTANTES E CONFIGURAÇÕES ---
CONFIG = {
    "page_title": "Gestão - Clips Burger",
    "layout": "centered",
    "sidebar_state": "expanded",
    "excel_file": "recebimentos.xlsx",
    "logo_path": "logo.png",
    "logo_url": "https://raw.githubusercontent.com/lucasricardocs/combo/main/logo.png"
}

# --- CARDÁPIO ATUALIZADO (JBC / DOUBLE) ---
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

# Preços dos combos
COMBO_1_PRECO = 25.00  # JBC + Refri Lata
COMBO_2_PRECO = 30.00  # Double Cheese + Refri Lata

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

def get_global_centered_styles():
    return [
        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('background-color', '#000033'), ('color', '#ffffff'), ('padding', '8px'), ('border', '1px solid #444')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('padding', '8px'), ('color', '#e0e0e0'), ('background-color', '#1a1a4a')]},
        {'selector': 'table', 'props': [('width', '100%'), ('margin-left', 'auto'), ('margin-right', 'auto')]}
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

def get_logo_base64():
    """Tenta carregar a logo localmente ou da URL"""
    if os.path.exists(CONFIG["logo_path"]):
        try:
            with open(CONFIG["logo_path"], "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception as e:
            pass
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(CONFIG["logo_url"], headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

# --- FUNÇÕES PARA ALGORITMO GENÉTICO COM 2 COMBOS (ORIGINAIS) ---
def create_individual_two_combos(max_combos=100):
    num_jbc = random.randint(0, max_combos)
    num_double = random.randint(0, max_combos)
    num_latas = num_jbc + num_double
    num_cebolas = random.randint(0, 50)
    
    individual = {
        "JBC (Junior Bacon Cheese)": num_jbc,
        "Double Cheese Burger": num_double,
        "Refri Lata": num_latas,
        "Cebola Adicional": num_cebolas
    }
    
    return individual

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
    if random.random() < 0.5:
        num_jbc = parent1.get("JBC (Junior Bacon Cheese)", 0)
    else:
        num_jbc = parent2.get("JBC (Junior Bacon Cheese)", 0)
    
    if random.random() < 0.5:
        num_double = parent1.get("Double Cheese Burger", 0)
    else:
        num_double = parent2.get("Double Cheese Burger", 0)
    
    num_latas = num_jbc + num_double
    
    if random.random() < 0.5:
        num_cebolas = (parent1.get("Cebola Adicional", 0) + parent2.get("Cebola Adicional", 0)) // 2
    else:
        num_cebolas = parent1.get("Cebola Adicional", 0) if random.random() < 0.5 else parent2.get("Cebola Adicional", 0)
    
    return {
        "JBC (Junior Bacon Cheese)": num_jbc,
        "Double Cheese Burger": num_double,
        "Refri Lata": num_latas,
        "Cebola Adicional": num_cebolas
    }

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
    if target_value <= 0:
        return {}
    
    population = [create_individual_two_combos(max_combos) for _ in range(population_size)]
    best_individual = {}
    best_fitness = float('inf')
    
    for generation in range(generations):
        fitness_scores = [(individual, evaluate_fitness_two_combos(individual, target_value)) 
                         for individual in population]
        fitness_scores.sort(key=lambda x: x[1])
        
        if fitness_scores[0][1] < best_fitness:
            best_individual = fitness_scores[0][0].copy()
            best_fitness = fitness_scores[0][1]
        
        if best_fitness == 0:
            break
        
        elite_size = max(5, population_size // 10)
        next_generation = [ind[0].copy() for ind in fitness_scores[:elite_size]]
        
        while len(next_generation) < population_size:
            tournament_size = 3
            tournament = random.sample(fitness_scores, tournament_size)
            tournament.sort(key=lambda x: x[1])
            parent1 = tournament[0][0]
            parent2 = random.choice(fitness_scores[:10])[0]
            
            child = crossover_two_combos(parent1, parent2)
            child = mutate_two_combos(child)
            next_generation.append(child)
        
        population = next_generation
    
    final_combination = {k: int(v) for k, v in best_individual.items() if v > 0}
    
    return final_combination

def buscar_combinacao_two_combos(target_value, max_time_seconds=5, population_size=100, generations=200):
    start_time = time.time()
    best_global_individual = {}
    best_global_diff = float('inf')
    attempts = 0
    
    while (time.time() - start_time) < max_time_seconds:
        attempts += 1
        current_result = genetic_algorithm_two_combos(target_value, population_size, generations)
        current_fitness = evaluate_fitness_two_combos(current_result, target_value)
        
        if current_fitness == 0:
            return current_result, attempts
        
        if current_fitness < best_global_diff:
            best_global_diff = current_fitness
            best_global_individual = current_result
    
    return best_global_individual, attempts

# --- FUNÇÕES PARA GERAR PDF (ORIGINAIS) ---
def create_watermark(canvas, logo_path, width=400, height=400, opacity=0.1):
    try:
        if os.path.exists(logo_path):
            canvas.saveState()
            canvas.setFillColorRGB(255, 255, 255, alpha=opacity)
            canvas.drawImage(logo_path, (A4[0] - width) / 2, (A4[1] - height) / 2, 
                             width=width, height=height, mask='auto', preserveAspectRatio=True)
            canvas.restoreState()
    except Exception as e:
        pass

def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def create_pdf_report(df, vendas, total_vendas, imposto_simples, custo_funcionario, 
                    custo_contadora, total_custos, lucro_estimado, logo_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    elements = []
    elements.append(Paragraph("Relatório de Gestão - Clips Burger", styles['Title']))
    elements.append(Spacer(1, 12))
    
    data_resumo = [
        ["Total Bruto", format_currency(total_vendas)],
        ["Imposto (Simples)", format_currency(imposto_simples)],
        ["Custos Operacionais", format_currency(total_custos)],
        ["Lucro Estimado", format_currency(lucro_estimado)]
    ]
    t = Table(data_resumo, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ff4b4b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(t)
    
    doc.build(elements, onFirstPage=lambda c, d: create_watermark(c, logo_path), 
              onLaterPages=lambda c, d: create_watermark(c, logo_path))
    buffer.seek(0)
    return buffer

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title=CONFIG["page_title"],
    layout=CONFIG["layout"],
    initial_sidebar_state=CONFIG["sidebar_state"]
)

# --- CSS GLOBAL (ORIGINAL COM CORREÇÕES) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #1e005e 0%, #000033 50%, #000000 100%);
        background-size: cover;
        background-attachment: fixed;
        color: #f0f2f6;
    }

    th, td {
        text-align: center !important;
        vertical-align: middle !important;
        color: #e0e0e0 !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox, .stDateInput, div[data-baseweb="select"] > div {
        background-color: #1a1c24 !important; 
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* LOGO ANIMADA */
    .logo-container {
        position: relative;
        width: 300px;
        height: 300px;
        margin: 0 auto 20px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .logo-animada {
        width: 250px;
        height: auto;
        position: relative;
        z-index: 20; 
    }

    .sparkle {
        position: absolute;
        width: 8px; 
        height: 8px;
        background-color: #FF4500;
        border-radius: 50%;
        bottom: 10px;
        z-index: 1;
        opacity: 0;
        box-shadow: 0 0 5px #FFD700, 0 0 10px #FF8C00;
    }

    @keyframes steady-rise-high {
        0% { opacity: 0; transform: translateY(0) scale(0.5); }
        10% { opacity: 0.8; }
        80% { opacity: 0.6; }
        100% { opacity: 0; transform: translateY(-400px) scale(0.1); }
    }

    .s1 { bottom: 20px; left: 10%; animation: steady-rise-high 5s linear infinite; }
    .s2 { bottom: 10px; left: 30%; animation: steady-rise-high 6s linear infinite; animation-delay: 1.5s; }
    .s3 { bottom: 25px; left: 50%; animation: steady-rise-high 5.5s linear infinite; animation-delay: 3.0s; }
    .s4 { bottom: 15px; left: 70%; animation: steady-rise-high 4.5s linear infinite; animation-delay: 0.5s; }
    .s5 { bottom: 5px;  left: 90%; animation: steady-rise-high 5.2s linear infinite; animation-delay: 2.2s; }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO SESSION STATE ---
init_data_file()
if 'df_receipts' not in st.session_state:
    st.session_state.df_receipts = load_data()
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'vendas_data' not in st.session_state:
    st.session_state.vendas_data = None
if 'resultado_arquivo' not in st.session_state:
    st.session_state.resultado_arquivo = None
if 'resultado_pix' not in st.session_state:
    st.session_state.resultado_pix = None

# --- LOGO ANIMADA ---
logo_base64 = get_logo_base64()

if logo_base64:
    st.markdown(
        f"""
        <div class="logo-container">
            <div class="sparkle s1"></div>
            <div class="sparkle s2"></div>
            <div class="sparkle s3"></div>
            <div class="sparkle s4"></div>
            <div class="sparkle s5"></div>
            <img src="data:image/png;base64,{logo_base64}" class="logo-animada">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown("""
    <div class='logo-container'>
        <h1 style='color: #ff4b4b; font-size: 80px; margin: 0;'>🍔</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #ff4b4b;'>🍔 Clips Burger - Gestão</h2>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    population_size = st.slider("População", 20, 200, 100)
    generations = st.slider("Gerações", 10, 500, 200)

# --- MENU ---
menu_opcoes = ["📈 Resumo das Vendas", "🧩 Análise com Arquivo", "💸 Calculadora PIX"]
escolha_menu = st.radio("Navegação", menu_opcoes, horizontal=True, label_visibility="collapsed")

# --- CONTEÚDO ---
if escolha_menu == "📈 Resumo das Vendas":
    st.subheader("📤 Upload de Dados")
    arquivo = st.file_uploader("Envie o arquivo (.csv ou .xlsx)", type=["csv", "xlsx"])
    if arquivo:
        st.success("Arquivo carregado!")

elif escolha_menu == "🧩 Análise com Arquivo":
    st.subheader("🧩 Otimizador")
    target_val = st.number_input("Valor Alvo (R$)", min_value=0.0, value=100.0)
    if st.button("🚀 Otimizar"):
        with st.spinner("Calculando..."):
            res, _ = buscar_combinacao_two_combos(target_val, population_size, generations)
            st.session_state.resultado_arquivo = {'resultado': res, 'alvo': target_val}
    
    if st.session_state.resultado_arquivo:
        res = st.session_state.resultado_arquivo['resultado']
        st.write("### Resultado:")
        for k, v in res.items():
            st.write(f"**{v}x** {k}")

elif escolha_menu == "💸 Calculadora PIX":
    st.subheader("💸 Conciliação PIX")
    pix_input = st.text_area("Valores (um por linha)")
    if st.button("Calcular"):
        try:
            vals = [float(x.strip().replace(',', '.')) for x in pix_input.split('\n') if x.strip()]
            st.metric("Total", format_currency(sum(vals)))
        except:
            st.error("Erro nos valores.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>© 2026 Clips Burger</p>", unsafe_allow_html=True)
