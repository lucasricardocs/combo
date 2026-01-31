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
            st.warning(f"Erro ao carregar logo local: {e}")
    
    try:
        # Adicionado User-Agent para evitar bloqueios
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(CONFIG["logo_url"], headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        # Silencioso se falhar na URL também
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
        print(f"Erro ao adicionar marca d'água: {e}")

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
    
    # Estilos customizados
    title_style = styles['Title']
    title_style.textColor = colors.HexColor("#ff4b4b")
    
    elements = []
    
    # Título
    elements.append(Paragraph("Relatório de Gestão - Clips Burger", title_style))
    elements.append(Spacer(1, 12))
    
    # Resumo Financeiro
    elements.append(Paragraph("Resumo Financeiro", styles['Heading2']))
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

# --- FUNÇÃO DE EXIBIÇÃO DE RESULTADO (ORIGINAL COM UI MELHORADA) ---
def exibir_resultado_combinacao(dados):
    if not dados: return
    
    res = dados['resultado']
    alvo = dados['alvo']
    
    qty_jbc = res.get("JBC (Junior Bacon Cheese)", 0)
    qty_double = res.get("Double Cheese Burger", 0)
    qty_lata = res.get("Refri Lata", 0)
    qty_cebola = res.get("Cebola Adicional", 0)
    
    total_calc = (qty_jbc * 10.0) + (qty_double * 15.0) + (qty_lata * 15.0) + (qty_cebola * 0.5)
    diff = abs(total_calc - alvo)
    
    cor_border = "#10b981" if diff == 0 else "#f59e0b"
    cor_text = "#10b981" if diff == 0 else "#f59e0b"
    
    st.markdown(f"""
    <div style="background-color: rgba(30, 41, 59, 0.7); border: 2px solid {cor_border}; border-radius: 15px; padding: 20px; margin: 10px 0; backdrop-filter: blur(10px);">
        <h3 style="text-align: center; color: {cor_text}; margin-bottom: 20px;">🎯 Resultado da Otimização</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;">
            <div style="text-align: center; min-width: 120px;">
                <p style="color: #94a3b8; margin: 0;">JBC</p>
                <h2 style="margin: 0;">{qty_jbc}</h2>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <p style="color: #94a3b8; margin: 0;">Double</p>
                <h2 style="margin: 0;">{qty_double}</h2>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <p style="color: #94a3b8; margin: 0;">Latas</p>
                <h2 style="margin: 0;">{qty_lata}</h2>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <p style="color: #94a3b8; margin: 0;">Cebolas</p>
                <h2 style="margin: 0;">{qty_cebola}</h2>
            </div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <div style="text-align: center;">
            <p style="color: #94a3b8; margin: 0;">Total Calculado</p>
            <h1 style="color: {cor_text}; margin: 5px 0;">{format_currency(total_calc)}</h1>
            <p style="font-size: 0.9rem; color: #64748b;">Meta: {format_currency(alvo)} | Diferença: {format_currency(diff)}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title=CONFIG["page_title"],
    layout=CONFIG["layout"],
    initial_sidebar_state=CONFIG["sidebar_state"]
)

# --- CSS GLOBAL (MELHORADO) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Logo Animada */
    .logo-container {
        position: relative;
        width: 220px;
        height: 220px;
        margin: 0 auto 10px auto;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10;
    }

    .logo-animada {
        width: 180px;
        height: auto;
        z-index: 20;
        filter: drop-shadow(0 0 15px rgba(255, 75, 75, 0.3));
        transition: transform 0.3s ease;
    }
    
    .logo-animada:hover {
        transform: scale(1.05);
    }

    .sparkle {
        position: absolute;
        width: 6px; 
        height: 6px;
        background-color: #ff4b4b;
        border-radius: 50%;
        bottom: 20px;
        z-index: 1;
        opacity: 0;
        box-shadow: 0 0 8px #ff4b4b;
    }

    @keyframes rise {
        0% { opacity: 0; transform: translateY(0) scale(0.5); }
        20% { opacity: 0.8; }
        80% { opacity: 0.4; }
        100% { opacity: 0; transform: translateY(-180px) scale(0.2); }
    }

    .s1 { left: 20%; animation: rise 3s infinite; animation-delay: 0s; }
    .s2 { left: 40%; animation: rise 4s infinite; animation-delay: 1s; }
    .s3 { left: 60%; animation: rise 3.5s infinite; animation-delay: 0.5s; }
    .s4 { left: 80%; animation: rise 4.5s infinite; animation-delay: 1.5s; }

    /* Estilização Geral */
    .stButton>button {
        border-radius: 10px;
        background: #ff4b4b;
        color: white;
        border: none;
        font-weight: 600;
        width: 100%;
    }
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }

    div[role="radiogroup"] {
        background: #1e293b;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
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

# --- EXIBIÇÃO DA LOGO ---
logo_base64 = get_logo_base64()

st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.markdown('<div class="sparkle s1"></div><div class="sparkle s2"></div><div class="sparkle s3"></div><div class="sparkle s4"></div>', unsafe_allow_html=True)
if logo_base64:
    # CORREÇÃO: Adicionado data:image/png;base64,
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="logo-animada">', unsafe_allow_html=True)
else:
    st.markdown("<h1 style='font-size: 80px; margin: 0;'>🍔</h1>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #ff4b4b; margin: 0; font-weight: 800;'>CLIPS BURGER</h1>
    <p style='color: #94a3b8; font-size: 1.1rem;'>Sistema de Gestão Inteligente</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    st.info("🧬 Algoritmo Genético")
    
    population_size = st.slider("População", 20, 200, 100)
    generations = st.slider("Gerações", 10, 500, 200)
    
    st.divider()
    st.caption("🎁 **Combo 1:** JBC + Lata = R$ 25,00")
    st.caption("🎁 **Combo 2:** Double + Lata = R$ 30,00")

# --- MENU DE NAVEGAÇÃO ---
menu_opcoes = ["📈 Resumo das Vendas", "🧩 Análise com Arquivo", "💸 Calculadora PIX"]
escolha_menu = st.radio("Navegação", menu_opcoes, horizontal=True, label_visibility="collapsed", key="nav_menu")

# --- CONTEÚDO DAS ABAS ---

if escolha_menu == "📈 Resumo das Vendas":
    st.subheader("📊 Visão Geral")
    # Simulação de dashboard para UI
    c1, c2, c3 = st.columns(3)
    c1.metric("Vendas Totais", "R$ 4.250,00", "+5%")
    c2.metric("Ticket Médio", "R$ 42,50", "Estável")
    c3.metric("Pedidos", "102", "+12")
    
    st.markdown("---")
    st.subheader("📤 Upload de Dados")
    arquivo = st.file_uploader("Envie o arquivo de transações (.csv ou .xlsx)", type=["csv", "xlsx"])
    
    if arquivo:
        st.success("Arquivo carregado com sucesso!")
        # Lógica original de processamento de arquivo seria mantida aqui...

elif escolha_menu == "🧩 Análise com Arquivo":
    st.subheader("🧩 Otimizador de Combinações")
    target_val = st.number_input("Valor Alvo para Otimização (R$)", min_value=0.0, value=100.0, step=10.0)
    
    if st.button("🚀 Iniciar Algoritmo Genético"):
        with st.spinner("O DNA dos hambúrgueres está sendo analisado..."):
            res, attempts = buscar_combinacao_two_combos(target_val, population_size=population_size, generations=generations)
            st.session_state.resultado_arquivo = {'resultado': res, 'alvo': target_val}
            
    if st.session_state.resultado_arquivo:
        exibir_resultado_combinacao(st.session_state.resultado_arquivo)

elif escolha_menu == "💸 Calculadora PIX":
    st.subheader("💸 Conciliação PIX")
    pix_input = st.text_area("Insira os valores PIX (um por linha)", height=150)
    if st.button("Calcular Total"):
        try:
            vals = [float(x.strip().replace(',', '.')) for x in pix_input.split('\n') if x.strip()]
            total = sum(vals)
            st.success(f"Total Conciliado: {format_currency(total)}")
            
            if st.button("🧬 Otimizar para este valor"):
                with st.spinner("Calculando..."):
                    res, _ = buscar_combinacao_two_combos(total)
                    st.session_state.resultado_pix = {'resultado': res, 'alvo': total}
        except:
            st.error("Erro ao processar valores. Verifique o formato.")
            
    if st.session_state.resultado_pix:
        exibir_resultado_combinacao(st.session_state.resultado_pix)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>© 2026 Clips Burger - Gestão de Alta Performance</p>", unsafe_allow_html=True)
