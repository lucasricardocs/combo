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

# --- CARDÁPIO NOVO (JBC / DOUBLE) ---
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
        {'selector': 'th', 'props': [
            ('text-align', 'center'), 
            ('vertical-align', 'middle'), 
            ('background-color', '#FF6B35'), 
            ('color', '#FFFFFF'), 
            ('padding', '16px'), 
            ('font-weight', '700'),
            ('font-size', '15px'),
            ('letter-spacing', '0.5px'),
            ('text-transform', 'uppercase'),
            ('border', 'none')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'), 
            ('vertical-align', 'middle'), 
            ('padding', '14px'), 
            ('color', '#FFFFFF'), 
            ('background-color', '#1a1a1a'),
            ('font-size', '15px'),
            ('border-bottom', '1px solid #333333')
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#2a2a2a'),
            ('transition', 'all 0.3s ease')
        ]},
        {'selector': 'table', 'props': [
            ('width', '100%'), 
            ('margin-left', 'auto'), 
            ('margin-right', 'auto'),
            ('border-collapse', 'collapse'),
            ('box-shadow', '0 4px 20px rgba(0,0,0,0.3)'),
            ('border-radius', '12px'),
            ('overflow', 'hidden')
        ]}
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

# --- FUNÇÃO DA LOGO CORRIGIDA ---
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None

# --- FUNÇÕES PARA ALGORITMO GENÉTICO COM 2 COMBOS ---
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

# --- FUNÇÕES PARA GERAR PDF ---
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
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    elements = []
    
    try:
        if os.path.exists(logo_path):
            img = Image(logo_path, width=2*inch, height=1.5*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 0.5*inch))
    except Exception as e:
        print(f"Erro ao adicionar logo: {e}")
    
    elements.append(Paragraph("Relatório Financeiro - Clips Burger", title_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Data do relatório: {datetime.now().strftime('%d/%m/%Y')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph("Resumo Financeiro", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    data = [
        ["Métrica", "Valor"],
        ["Faturamento Bruto", format_currency(total_vendas)],
        ["Imposto Simples (6%)", format_currency(imposto_simples)],
        ["Custo Funcionário CLT", format_currency(custo_funcionario)],
        ["Custo Contadora", format_currency(custo_contadora)],
        ["Total de Custos", format_currency(total_custos)],
        ["Lucro Estimado", format_currency(lucro_estimado)]
    ]
    
    table = Table(data, colWidths=[doc.width/2.5, doc.width/2.5])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, -1), (1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5*inch))
    
    elements.append(Paragraph("Análise de Vendas", heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        vendas.plot(kind='bar', x='Forma', y='Valor', ax=ax, color='steelblue')
        ax.set_title('Vendas por Forma de Pagamento')
        ax.set_ylabel('Valor (R$)')
        ax.set_xlabel('')
        plt.tight_layout()
        img_buf = fig_to_buffer(fig)
        img = Image(img_buf, width=doc.width, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
        plt.close(fig)
    except Exception as e:
        elements.append(Paragraph(f"Erro ao gerar gráfico de vendas: {e}", normal_style))
    
    try:
        custos_df = pd.DataFrame({
            'Item': ['Impostos', 'Funcionário', 'Contadora'],
            'Valor': [imposto_simples, custo_funcionario, custo_contadora]
        })
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(custos_df['Valor'], labels=custos_df['Item'], autopct='%1.1f%%', startangle=90, shadow=True)
        ax.set_title('Composição dos Custos')
        plt.tight_layout()
        img_buf = fig_to_buffer(fig)
        img = Image(img_buf, width=doc.width, height=4*inch)
        elements.append(img)
        plt.close(fig)
    except Exception as e:
        elements.append(Paragraph(f"Erro ao gerar gráfico de custos: {e}", normal_style))
    
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("Detalhamento por Forma de Pagamento", subheading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    data = [["Forma de Pagamento", "Valor"]]
    for _, row in vendas.iterrows():
        data.append([row['Forma'], format_currency(row['Valor'])])
    
    table = Table(data, colWidths=[doc.width/2, doc.width/4])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, inch))
    footer_text = "Este relatório foi gerado automaticamente pelo Sistema de Gestão da Clips Burger."
    elements.append(Paragraph(footer_text, normal_style))
    
    def add_watermark(canvas, doc):
        create_watermark(canvas, logo_path, width=300, height=300, opacity=0.1)
    
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)
    buffer.seek(0)
    return buffer

def create_altair_chart(data, chart_type, x_col, y_col, color_col=None, title=None, interactive=True):
    if chart_type == 'line':
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f'{x_col}:T', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            tooltip=[x_col, y_col]
        )
    elif chart_type == 'bar':
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_col),
            y=alt.Y(f'{y_col}:Q', title=y_col),
            color=alt.Color(f'{color_col}:N') if color_col else alt.value('#FF6B35'),
            tooltip=[x_col, y_col]
        )
    elif chart_type == 'pie':
        chart = alt.Chart(data).mark_arc().encode(
            theta=alt.Theta(f'{y_col}:Q', stack=True),
            color=alt.Color(f'{x_col}:N', legend=alt.Legend(title=x_col)),
            tooltip=[x_col, y_col]
        )
    
    chart = chart.properties(
        title=title if title else f'{y_col} por {x_col}',
        width=700,
        height=400
    )
    return chart.interactive() if interactive else chart

# --- LÓGICA DE PROCESSAMENTO GENÉTICO ---
def gerar_dados_geneticos_two_combos(valor_alvo_total, pop_size, n_gens):
    combinacao, ciclos = buscar_combinacao_two_combos(
        valor_alvo_total, 
        max_time_seconds=5, 
        population_size=pop_size, 
        generations=n_gens
    )
    
    sanduiches = {}
    bebidas = {}
    
    for item, qty in combinacao.items():
        if item in CARDAPIOS["sanduiches"]:
            sanduiches[item] = qty
        elif item in CARDAPIOS["bebidas"]:
            bebidas[item] = qty
    
    val_sand = sum(CARDAPIOS["sanduiches"][k] * v for k, v in sanduiches.items())
    val_beb = sum(CARDAPIOS["bebidas"][k] * v for k, v in bebidas.items())
    val_total = val_sand + val_beb
    
    return {
        'sanduiches': sanduiches,
        'bebidas': bebidas,
        'val_sand': val_sand,
        'val_beb': val_beb,
        'val_total': val_total,
        'alvo': valor_alvo_total,
        'ciclos': ciclos
    }

def renderizar_resultados(dados):
    # Card principal com valor alvo
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                border-radius: 20px; padding: 30px; margin-bottom: 30px; 
                box-shadow: 0 10px 40px rgba(255, 107, 53, 0.3);
                animation: slideDown 0.5s ease-out;">
        <h2 style="color: #FFFFFF; margin: 0; text-align: center; font-size: 28px; font-weight: 700; letter-spacing: 1px;">
            🎯 VALOR ALVO
        </h2>
        <p style="color: #FFFFFF; text-align: center; margin: 15px 0 5px 0; font-size: 48px; font-weight: 900;">
            {format_currency(dados['alvo'])}
        </p>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0; font-size: 14px;">
            🤖 {dados['ciclos']} ciclos de evolução genética
        </p>
    </div>
    
    <style>
    @keyframes slideDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    qty_jbc = dados['sanduiches'].get("JBC (Junior Bacon Cheese)", 0)
    qty_double = dados['sanduiches'].get("Double Cheese Burger", 0)
    qty_lata = dados['bebidas'].get("Refri Lata", 0)
    qty_cebola = dados['sanduiches'].get("Cebola Adicional", 0)
    
    # Badges de combos
    col1, col2 = st.columns(2)
    
    with col1:
        if qty_jbc > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        border-radius: 16px; padding: 20px; margin-bottom: 15px;
                        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
                        animation: fadeIn 0.6s ease-out;">
                <div style="color: #FFFFFF; font-size: 16px; font-weight: 600; margin-bottom: 8px;">✅ COMBO 1</div>
                <div style="color: #FFFFFF; font-size: 32px; font-weight: 900; margin-bottom: 5px;">{qty_jbc}x</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 13px;">JBC + Refri Lata</div>
                <div style="color: #FFFFFF; font-size: 20px; font-weight: 700; margin-top: 10px;">{format_currency(qty_jbc * COMBO_1_PRECO)}</div>
            </div>
            
            <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            </style>
            """, unsafe_allow_html=True)
    
    with col2:
        if qty_double > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                        border-radius: 16px; padding: 20px; margin-bottom: 15px;
                        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
                        animation: fadeIn 0.6s ease-out 0.1s both;">
                <div style="color: #FFFFFF; font-size: 16px; font-weight: 600; margin-bottom: 8px;">✅ COMBO 2</div>
                <div style="color: #FFFFFF; font-size: 32px; font-weight: 900; margin-bottom: 5px;">{qty_double}x</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 13px;">Double Cheese + Refri Lata</div>
                <div style="color: #FFFFFF; font-size: 20px; font-weight: 700; margin-top: 10px;">{format_currency(qty_double * COMBO_2_PRECO)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Status da regra
    if qty_jbc + qty_double == qty_lata:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    border-radius: 16px; padding: 18px; margin: 20px 0;
                    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.3);">
            <div style="color: #FFFFFF; font-size: 15px; text-align: center;">
                ✅ <strong>Regra respeitada:</strong> {qty_jbc + qty_double} sanduíches = {qty_lata} latas
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    border-radius: 16px; padding: 18px; margin: 20px 0;
                    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3);">
            <div style="color: #FFFFFF; font-size: 15px; text-align: center;">
                ⚠️ <strong>Atenção:</strong> {qty_jbc + qty_double} sanduíches ≠ {qty_lata} latas
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Tabelas de produtos e bebidas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #1a1a1a; border-radius: 16px; padding: 24px; 
                    box-shadow: 0 6px 20px rgba(0,0,0,0.4); margin-bottom: 20px;">
            <h3 style="color: #FF6B35; margin: 0 0 20px 0; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">
                🍔 PRODUTOS
            </h3>
        """, unsafe_allow_html=True)
        
        if dados['sanduiches']:
            df_s = pd.DataFrame({
                'Produto': list(dados['sanduiches'].keys()),
                'Qnt': list(dados['sanduiches'].values()),
                'Preço Unit.': [CARDAPIOS["sanduiches"][k] for k in dados['sanduiches']],
                'Subtotal': [CARDAPIOS["sanduiches"][k]*v for k,v in dados['sanduiches'].items()]
            })
            df_s = df_s.sort_values('Subtotal', ascending=False)
            html_s = df_s.style.format({'Qnt':'{:.0f}', 'Preço Unit.':'R$ {:.2f}', 'Subtotal':'R$ {:.2f}'})\
                .set_table_styles(get_global_centered_styles()).hide(axis='index').to_html()
            st.markdown(html_s, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                        border-radius: 12px; padding: 16px; margin-top: 20px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 13px; margin-bottom: 5px;">TOTAL</div>
                <div style="color: #FFFFFF; font-size: 26px; font-weight: 900;">{format_currency(dados['val_sand'])}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Sem itens")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #1a1a1a; border-radius: 16px; padding: 24px; 
                    box-shadow: 0 6px 20px rgba(0,0,0,0.4); margin-bottom: 20px;">
            <h3 style="color: #3b82f6; margin: 0 0 20px 0; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">
                🥤 BEBIDAS
            </h3>
        """, unsafe_allow_html=True)
        
        if dados['bebidas']:
            df_b = pd.DataFrame({
                'Produto': list(dados['bebidas'].keys()),
                'Qnt': list(dados['bebidas'].values()),
                'Preço Unit.': [CARDAPIOS["bebidas"][k] for k in dados['bebidas']],
                'Subtotal': [CARDAPIOS["bebidas"][k]*v for k,v in dados['bebidas'].items()]
            })
            df_b = df_b.sort_values('Subtotal', ascending=False)
            html_b = df_b.style.format({'Qnt':'{:.0f}', 'Preço Unit.':'R$ {:.2f}', 'Subtotal':'R$ {:.2f}'})\
                .set_table_styles(get_global_centered_styles()).hide(axis='index').to_html()
            st.markdown(html_b, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                        border-radius: 12px; padding: 16px; margin-top: 20px; text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 13px; margin-bottom: 5px;">TOTAL</div>
                <div style="color: #FFFFFF; font-size: 26px; font-weight: 900;">{format_currency(dados['val_beb'])}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Sem itens")
        
        st.markdown("</div>", unsafe_allow_html=True)

    diff = dados['alvo'] - dados['val_total']
    
    # Resumo final dos combos
    if qty_jbc > 0 or qty_double > 0:
        valor_combo1 = qty_jbc * COMBO_1_PRECO
        valor_combo2 = qty_double * COMBO_2_PRECO
        valor_cebolas = qty_cebola * 0.50
        
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 2px solid #FF6B35; border-radius: 20px; 
                    padding: 30px; margin: 30px 0; box-shadow: 0 10px 40px rgba(255, 107, 53, 0.2);">
            <h3 style="color: #FF6B35; margin: 0 0 25px 0; text-align: center; font-size: 24px; font-weight: 700; letter-spacing: 1px;">
                🎁 RESUMO DOS COMBOS
            </h3>
            {'<div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 12px;"><span style="color: #FFFFFF; font-size: 16px;">Combo 1: ' + str(qty_jbc) + ' × R$ 25,00</span><span style="color: #10b981; font-size: 20px; font-weight: 700;">' + format_currency(valor_combo1) + '</span></div>' if qty_jbc > 0 else ''}
            {'<div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 12px;"><span style="color: #FFFFFF; font-size: 16px;">Combo 2: ' + str(qty_double) + ' × R$ 30,00</span><span style="color: #3b82f6; font-size: 20px; font-weight: 700;">' + format_currency(valor_combo2) + '</span></div>' if qty_double > 0 else ''}
            {'<div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 12px;"><span style="color: #FFFFFF; font-size: 16px;">Cebola Adicional: ' + str(qty_cebola) + ' × R$ 0,50</span><span style="color: #f59e0b; font-size: 20px; font-weight: 700;">' + format_currency(valor_cebolas) + '</span></div>' if qty_cebola > 0 else ''}
            <div style="height: 2px; background: linear-gradient(90deg, transparent, #FF6B35, transparent); margin: 25px 0;"></div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px; background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); border-radius: 12px;">
                <span style="color: #FFFFFF; font-size: 18px; font-weight: 700;">TOTAL COMBOS</span>
                <span style="color: #FFFFFF; font-size: 32px; font-weight: 900;">{format_currency(valor_combo1 + valor_combo2 + valor_cebolas)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card final com valor total
    cor_gradiente = "linear-gradient(135deg, #10b981 0%, #059669 100%)" if diff == 0 else "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
    
    st.markdown(f"""
    <div style="background: {cor_gradiente}; 
                border-radius: 24px; padding: 40px; margin: 30px 0; text-align: center;
                box-shadow: 0 15px 50px rgba(0,0,0,0.3);
                animation: pulse 2s ease-in-out infinite;">
        <div style="color: rgba(255,255,255,0.95); font-size: 18px; font-weight: 600; letter-spacing: 2px; margin-bottom: 15px;">
            💰 VALOR TOTAL
        </div>
        <div style="color: #FFFFFF; font-size: 64px; font-weight: 900; margin: 20px 0; text-shadow: 0 4px 10px rgba(0,0,0,0.3);">
            {format_currency(dados['val_total'])}
        </div>
        <div style="color: rgba(255,255,255,0.9); font-size: 16px; margin-top: 15px;">
            Meta: <strong>{format_currency(dados['alvo'])}</strong> • 
            Diferença: <strong>{format_currency(abs(diff))}</strong>
        </div>
    </div>
    
    <style>
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title=CONFIG["page_title"],
    layout=CONFIG["layout"],
    initial_sidebar_state=CONFIG["sidebar_state"]
)

# --- CSS GLOBAL MODERNO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #FFFFFF;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #FFFFFF !important;
    }

    /* Tabelas */
    th, td {
        text-align: center !important;
        vertical-align: middle !important;
        color: #FFFFFF !important;
    }
    
    /* Inputs modernos */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stDateInput, 
    div[data-baseweb="select"] > div {
        background-color: #2a2a2a !important; 
        color: #FFFFFF !important;
        border: 2px solid #3a3a3a !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #FF6B35 !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    /* Radio buttons - Menu navegação */
    div[role="radiogroup"] {
        background: #1a1a1a;
        padding: 8px;
        border-radius: 16px;
        gap: 8px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    div[role="radiogroup"] label {
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }

    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 15px !important; 
        font-weight: 600 !important;
        color: #888888 !important; 
        transition: all 0.3s ease !important;
    }

    div[role="radiogroup"] label:hover {
        background-color: #2a2a2a !important;
    }

    div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {
        color: #FF6B35 !important;
    }

    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
    }

    div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    
    /* Botões modernos */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(255, 107, 53, 0.4) !important;
    }
    
    /* Sidebar moderna */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Tabs modernas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888888;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2a2a2a;
        color: #FF6B35;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: white !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1a1a;
        border: 2px dashed #3a3a3a;
        border-radius: 16px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FF6B35;
        background-color: #2a2a2a;
    }
    
    /* Success, warning, error boxes */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background-color: #2a2a2a;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #FF6B35 !important;
        box-shadow: 0 0 0 4px rgba(255, 107, 53, 0.2) !important;
    }
    
    /* Logo container */
    .logo-container {
        position: relative;
        width: 350px;
        height: 350px;
        margin: 0 auto 30px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .logo-animada {
        width: 350px;
        height: auto;
        filter: drop-shadow(0 0 30px rgba(255, 107, 53, 0.4));
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .sparkle {
        position: absolute;
        width: 6px; 
        height: 6px;
        background: linear-gradient(135deg, #FFD700, #FF6B35);
        border-radius: 50%;
        bottom: 10px;
        opacity: 0;
        box-shadow: 0 0 10px #FFD700;
        pointer-events: none;
    }

    @keyframes steady-rise-high {
        0% { opacity: 0; transform: translateY(0) scale(0.5); }
        10% { opacity: 0.9; }
        80% { opacity: 0.6; }
        100% { opacity: 0; transform: translateY(-500px) scale(0.1); }
    }

    .s1 { bottom: 20px; left: 10%; animation: steady-rise-high 4s linear infinite; animation-delay: 0s; }
    .s2 { bottom: 10px; left: 20%; animation: steady-rise-high 5s linear infinite; animation-delay: 1s; }
    .s3 { bottom: 25px; left: 35%; animation: steady-rise-high 4.5s linear infinite; animation-delay: 2s; }
    .s4 { bottom: 15px; left: 50%; animation: steady-rise-high 5.5s linear infinite; animation-delay: 0.5s; }
    .s5 { bottom: 5px;  left: 65%; animation: steady-rise-high 4.8s linear infinite; animation-delay: 1.8s; }
    .s6 { bottom: 12px; left: 80%; animation: steady-rise-high 5.2s linear infinite; animation-delay: 0.8s; }
    .s7 { bottom: 18px; left: 90%; animation: steady-rise-high 4.3s linear infinite; animation-delay: 2.5s; }
    .s8 { bottom: 8px;  left: 30%; animation: steady-rise-high 5.5s linear infinite; animation-delay: 1.5s; }
    
    /* Scrollbar customizada */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF6B35, #F7931E);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #F7931E, #FF6B35);
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

# --- LOGO ANIMADA ---
try:
    if os.path.exists(CONFIG["logo_path"]):
        img_base64 = get_img_as_base64(CONFIG["logo_path"])
        if img_base64:
            st.markdown(
                f"""
                <div class="logo-container">
                    <div class="sparkle s1"></div>
                    <div class="sparkle s2"></div>
                    <div class="sparkle s3"></div>
                    <div class="sparkle s4"></div>
                    <div class="sparkle s5"></div>
                    <div class="sparkle s6"></div>
                    <div class="sparkle s7"></div>
                    <div class="sparkle s8"></div>
                    <img src="data:image/png;base64,{img_base64}" class="logo-animada" alt="Logo Clips Burger">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            raise Exception("Não foi possível carregar a imagem")
    else:
        raise Exception("Arquivo de logo não encontrado")
except Exception as e:
    st.markdown("""
    <div class='logo-container' style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                                       border-radius: 30px; padding: 60px; box-shadow: 0 15px 50px rgba(255, 107, 53, 0.4);'>
        <h1 style='color: #FFFFFF; font-size: 80px; margin: 0; text-shadow: 0 4px 10px rgba(0,0,0,0.3);'>🍔</h1>
        <p style='color: #FFFFFF; font-size: 36px; font-weight: 900; margin: 15px 0 0 0; letter-spacing: 2px;'>CLIPS BURGER</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"⚠️ Logo não encontrada: {CONFIG['logo_path']}")

st.markdown("""
<div style='text-align: center; margin: 40px 0;'>
    <h1 style='color: #FFFFFF; margin: 0; font-size: 42px; font-weight: 900; letter-spacing: 1px;'>
        🍔 Sistema de Gestão
    </h1>
    <p style='color: #888888; margin-top: 12px; font-size: 18px; font-weight: 500;'>
        Análise inteligente de vendas e combinações
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                padding: 20px; border-radius: 16px; margin-bottom: 30px; text-align: center;
                box-shadow: 0 6px 20px rgba(255, 107, 53, 0.3);'>
        <h2 style='color: white; margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 1px;'>
            ⚙️ CONFIGURAÇÕES
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🧬 **Algoritmo Genético**")
    
    population_size = st.slider(
        "Tamanho da População", 
        min_value=20, 
        max_value=200, 
        value=100, 
        step=10,
        help="Maior população = melhor resultado"
    )
    
    generations = st.slider(
        "Número de Gerações", 
        min_value=10, 
        max_value=500, 
        value=200, 
        step=10,
        help="Mais gerações = melhor convergência"
    )
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: #2a2a2a; padding: 20px; border-radius: 12px; border-left: 4px solid #FF6B35;'>
        <p style='color: #FF6B35; margin: 0 0 10px 0; font-weight: 700; font-size: 15px;'>⚠️ REGRA PRINCIPAL</p>
        <p style='color: #FFFFFF; margin: 0; font-size: 14px;'>Total Sanduíches = Total Latas</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: #2a2a2a; padding: 18px; border-radius: 12px;'>
        <p style='color: #10b981; margin: 8px 0; font-size: 14px;'>🎁 <strong>Combo 1:</strong> JBC + Lata = R$ 25,00</p>
        <p style='color: #3b82f6; margin: 8px 0; font-size: 14px;'>🎁 <strong>Combo 2:</strong> Double + Lata = R$ 30,00</p>
        <p style='color: #f59e0b; margin: 8px 0; font-size: 14px;'>🧅 <strong>Cebola:</strong> Ajuste (R$ 0,50)</p>
    </div>
    """, unsafe_allow_html=True)

# --- MENU DE NAVEGAÇÃO ---
menu_opcoes = ["📈 Resumo das Vendas", "🧩 Análise com Arquivo", "💸 Calculadora PIX"]
escolha_menu = st.radio("", menu_opcoes, horizontal=True, label_visibility="collapsed", key="nav_menu")

# --- CONTEÚDO DAS ABAS ---

if escolha_menu == "📈 Resumo das Vendas":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                padding: 24px; border-radius: 20px; margin-bottom: 30px; text-align: center;
                box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
        <h2 style='color: white; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
            📤 UPLOAD DE DADOS
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    arquivo = st.file_uploader("Envie o arquivo de transações (.csv ou .xlsx)", type=["csv", "xlsx"])
    
    if arquivo:
        try:
            with st.spinner("Processando arquivo..."):
                if arquivo.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(arquivo, sep=';', encoding='utf-8', dtype=str)
                    except pd.errors.ParserError:
                        arquivo.seek(0)
                        try:
                            df = pd.read_csv(arquivo, sep=',', encoding='utf-8', dtype=str)
                        except:
                            arquivo.seek(0)
                            df = pd.read_csv(arquivo, engine='python', dtype=str)
                else:
                    df = pd.read_excel(arquivo, dtype=str)
                
                required_cols = ['Tipo', 'Bandeira', 'Valor']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Erro: O arquivo precisa conter as colunas: {', '.join(required_cols)}")
                    st.stop()

                df['Tipo'] = df['Tipo'].str.lower().str.strip().fillna('desconhecido')
                df['Bandeira'] = df['Bandeira'].str.lower().str.strip().fillna('desconhecida')
                df['Valor'] = pd.to_numeric(
                    df['Valor'].str.replace('.', '').str.replace(',', '.'), 
                    errors='coerce')
                df = df.dropna(subset=['Valor'])
                
                df['Forma'] = (df['Tipo'] + ' ' + df['Bandeira']).map(FORMAS_PAGAMENTO)
                df = df.dropna(subset=['Forma'])
                
                if df.empty:
                    st.warning("Nenhuma transação válida encontrada.")
                    st.stop()

                vendas = df.groupby('Forma')['Valor'].sum().reset_index()
                total_vendas = vendas['Valor'].sum()
                
                st.session_state.uploaded_data = df
                st.session_state.vendas_data = vendas
                st.session_state.total_vendas = total_vendas
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                        padding: 24px; border-radius: 20px; margin: 40px 0 30px 0; text-align: center;
                        box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
                <h2 style='color: white; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
                    📊 VISUALIZAÇÃO DE DADOS
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            bar_chart = create_altair_chart(vendas, 'bar', 'Forma', 'Valor', 'Forma', title='').properties(width=900, height=500)
            st.altair_chart(bar_chart, use_container_width=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                        padding: 24px; border-radius: 20px; margin: 40px 0 30px 0; text-align: center;
                        box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
                <h2 style='color: white; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
                    ⚙️ PARÂMETROS FINANCEIROS
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                salario_minimo = st.number_input("Salário Mínimo (R$)", value=1518.0, step=50.0)
            with col2:
                custo_contadora = st.number_input("Custo com Contadora (R$)", value=316.0, step=10.0)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                        padding: 24px; border-radius: 20px; margin: 40px 0 30px 0; text-align: center;
                        box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
                <h2 style='color: white; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
                    💰 RESULTADOS FINANCEIROS
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faturamento Bruto", format_currency(total_vendas))
            with col2:
                imposto_simples = total_vendas * 0.06
                st.metric("Imposto Simples (6%)", format_currency(imposto_simples))
            with col3:
                fgts = salario_minimo * 0.08
                ferias = (salario_minimo / 12) * (4/3)
                decimo_terceiro = salario_minimo / 12
                custo_funcionario = salario_minimo + fgts + ferias + decimo_terceiro
                st.metric("Custo Funcionário CLT", format_currency(custo_funcionario))
            
            total_custos = imposto_simples + custo_funcionario + custo_contadora
            lucro_estimado = total_vendas - total_custos
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de Custos", format_currency(total_custos))
            with col2:
                st.metric("Lucro Estimado", format_currency(lucro_estimado))
            
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            
            tab_detalhes1, tab_detalhes2, tab_detalhes3 = st.tabs([
                "📝 Composição de Custos", 
                "📚 Explicação dos Cálculos",
                "🍰 Gráfico de Composição"
            ])
            
            with tab_detalhes1:
                st.markdown(f"""
                <div style='background-color: #1a1a1a; padding: 25px; border-radius: 16px; border: 2px solid #2a2a2a;'>
                    <p style='color: #FFFFFF; font-size: 16px; margin: 12px 0;'>
                        <span style='color: #FF6B35; font-weight: 700;'>•</span> <strong>Imposto Simples (6%):</strong> {format_currency(imposto_simples)}
                    </p>
                    <p style='color: #FFFFFF; font-size: 16px; margin: 12px 0;'>
                        <span style='color: #FF6B35; font-weight: 700;'>•</span> <strong>Custo Funcionário CLT:</strong> {format_currency(custo_funcionario)}
                    </p>
                    <p style='color: #FFFFFF; font-size: 16px; margin: 12px 0;'>
                        <span style='color: #FF6B35; font-weight: 700;'>•</span> <strong>Custo Contadora:</strong> {format_currency(custo_contadora)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_detalhes2:
                st.markdown("""
                <div style='background-color: #1a1a1a; padding: 25px; border-radius: 16px; border: 2px solid #2a2a2a;'>
                    <div style='margin: 15px 0;'>
                        <p style='color: #FF6B35; font-weight: 700; font-size: 16px; margin-bottom: 8px;'>1. Imposto Simples Nacional</p>
                        <code style='background-color: #2a2a2a; padding: 8px 12px; border-radius: 8px; color: #FFFFFF; font-size: 14px; display: block;'>
                        Faturamento Bruto × 6%
                        </code>
                    </div>
                    <div style='margin: 15px 0;'>
                        <p style='color: #FF6B35; font-weight: 700; font-size: 16px; margin-bottom: 8px;'>2. Custo Funcionário CLT</p>
                        <code style='background-color: #2a2a2a; padding: 8px 12px; border-radius: 8px; color: #FFFFFF; font-size: 14px; display: block;'>
                        Salário + FGTS (8%) + Férias + 13º Salário
                        </code>
                    </div>
                    <div style='margin: 15px 0;'>
                        <p style='color: #FF6B35; font-weight: 700; font-size: 16px; margin-bottom: 8px;'>3. Total de Custos</p>
                        <code style='background-color: #2a2a2a; padding: 8px 12px; border-radius: 8px; color: #FFFFFF; font-size: 14px; display: block;'>
                        Imposto + Funcionário + Contadora
                        </code>
                    </div>
                    <div style='margin: 15px 0;'>
                        <p style='color: #FF6B35; font-weight: 700; font-size: 16px; margin-bottom: 8px;'>4. Lucro Estimado</p>
                        <code style='background-color: #2a2a2a; padding: 8px 12px; border-radius: 8px; color: #FFFFFF; font-size: 14px; display: block;'>
                        Faturamento Bruto - Total de Custos
                        </code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_detalhes3:
                custos_df = pd.DataFrame({
                    'Item': ['Impostos', 'Funcionário', 'Contadora'],
                    'Valor': [imposto_simples, custo_funcionario, custo_contadora]
                })
                
                graf_composicao = alt.Chart(custos_df).mark_arc().encode(
                    theta='Valor',
                    color=alt.Color('Item', scale=alt.Scale(range=['#FF6B35', '#F7931E', '#FFA500'])),
                    tooltip=['Item', alt.Tooltip('Valor', format='$.2f')]
                ).properties(width=700, height=500)
                st.altair_chart(graf_composicao, use_container_width=True)
            
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
            
            if st.button("📥 GERAR RELATÓRIO PDF", type="primary", use_container_width=True):
                with st.spinner("Gerando relatório..."):
                    pdf_buffer = create_pdf_report(
                        df, vendas, total_vendas, imposto_simples, custo_funcionario, 
                        custo_contadora, total_custos, lucro_estimado, CONFIG["logo_path"]
                    )
                    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                    st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="relatorio_clips_burger.pdf" style="display: inline-block; background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 16px 32px; border-radius: 12px; text-decoration: none; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">📥 Download Relatório PDF</a>', unsafe_allow_html=True)
                    st.success("✅ Relatório gerado com sucesso!")
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
    else:
        st.info("ℹ️ Aguardando upload do arquivo de transações.")

elif escolha_menu == "🧩 Análise com Arquivo":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                padding: 24px; border-radius: 20px; margin-bottom: 30px; text-align: center;
                box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
        <h2 style='color: white; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
            🧩 ANÁLISE DE COMBINAÇÕES
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.vendas_data is not None:
        vendas = st.session_state.vendas_data
        
        forma_selecionada = st.selectbox(
            "Selecione a forma de pagamento",
            options=vendas['Forma'].tolist(),
            format_func=lambda x: f"{x} ({format_currency(vendas.loc[vendas['Forma'] == x, 'Valor'].iloc[0])})"
        )
        
        valor_selecionado = vendas.loc[vendas['Forma'] == forma_selecionada, 'Valor'].iloc[0]
        
        if st.button("🔎 ANALISAR COMBINAÇÃO", type="primary", use_container_width=True):
            with st.spinner("Calculando a melhor combinação..."):
                dados = gerar_dados_geneticos_two_combos(
                    valor_selecionado, 
                    population_size, 
                    generations
                )
                st.session_state.resultado_arquivo = dados
        
        if st.session_state.resultado_arquivo:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            renderizar_resultados(st.session_state.resultado_arquivo)
        
    else:
        st.info("ℹ️ Faça o upload de dados na aba 'Resumo das Vendas' primeiro.")

elif escolha_menu == "💸 Calculadora PIX":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                padding: 24px; border-radius: 20px; margin-bottom: 30px; text-align: center;
                box-shadow: 0 8px 30px rgba(255, 107, 53, 0.3);'>
        <h2 style='color: white; margin: 0 0 10px 0; font-size: 28px; font-weight: 700; letter-spacing: 1px;'>
            💸 CALCULADORA RÁPIDA
        </h2>
        <p style='color: rgba(255,255,255,0.95); margin: 0; font-size: 15px;'>
            Digite um valor e descubra a combinação ideal de combos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_action = st.columns([0.5, 0.5])
    
    with col_input:
        valor_pix_input = st.number_input(
            "Digite o Valor (R$):", 
            min_value=0.0, 
            step=1.0, 
            format="%.2f",
            help="Insira o valor para análise"
        )
    
    with col_action:
        st.write("")
        st.write("")
        if st.button("🚀 CALCULAR COMBINAÇÃO", type="primary", use_container_width=True):
            if valor_pix_input > 0:
                with st.spinner("Calculando a melhor combinação..."):
                    dados = gerar_dados_geneticos_two_combos(
                        valor_pix_input, 
                        population_size, 
                        generations
                    )
                    st.session_state.resultado_pix = dados
            else:
                st.error("❌ Por favor, insira um valor maior que zero.")

    if st.session_state.resultado_pix:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        renderizar_resultados(st.session_state.resultado_pix)

# Rodapé
st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #555555; font-size: 14px; padding: 30px 0;'>
        © 2025 Clips Burger - Sistema de Gestão de Combos
    </div>
    """, 
    unsafe_allow_html=True
)
