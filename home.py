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
            ('background-color', '#1e3a8a'), 
            ('color', '#ffffff'), 
            ('padding', '12px'), 
            ('border', '1px solid #60a5fa'),
            ('font-weight', 'bold'),
            ('font-size', '14px')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'), 
            ('vertical-align', 'middle'), 
            ('padding', '10px'), 
            ('color', '#f3f4f6'), 
            ('background-color', '#1e293b'),
            ('border', '1px solid #475569'),
            ('font-size', '13px')
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#334155'),
        ]},
        {'selector': 'table', 'props': [
            ('width', '100%'), 
            ('margin-left', 'auto'), 
            ('margin-right', 'auto'),
            ('border-collapse', 'collapse')
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

# --- FUNÇÕES PARA ALGORITMO GENÉTICO COM 2 COMBOS (LÓGICA NOVA) ---
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
            color=alt.Color(f'{color_col}:N') if color_col else alt.value('steelblue'),
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
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                border-radius: 15px; padding: 20px; margin-bottom: 20px; 
                border: 2px solid #60a5fa; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <h2 style="color: #ffffff; margin: 0; text-align: center;">
            🎯 Valor Alvo: {format_currency(dados['alvo'])}
        </h2>
        <p style="color: #e0e7ff; text-align: center; margin: 10px 0 0 0; font-size: 14px;">
            🤖 Algoritmo realizou {dados['ciclos']} ciclos de evolução
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    qty_jbc = dados['sanduiches'].get("JBC (Junior Bacon Cheese)", 0)
    qty_double = dados['sanduiches'].get("Double Cheese Burger", 0)
    qty_lata = dados['bebidas'].get("Refri Lata", 0)
    qty_cebola = dados['sanduiches'].get("Cebola Adicional", 0)
    
    if qty_jbc > 0:
        st.markdown(f"""
        <div style="background-color: #065f46; border-left: 4px solid #10b981; 
                    padding: 15px; margin: 10px 0; border-radius: 8px;">
            <p style="color: #d1fae5; margin: 0; font-size: 16px; font-weight: bold;">
                ✅ Combo 1: {qty_jbc} unidades (JBC + Refri Lata) = {format_currency(qty_jbc * COMBO_1_PRECO)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if qty_double > 0:
        st.markdown(f"""
        <div style="background-color: #065f46; border-left: 4px solid #10b981; 
                    padding: 15px; margin: 10px 0; border-radius: 8px;">
            <p style="color: #d1fae5; margin: 0; font-size: 16px; font-weight: bold;">
                ✅ Combo 2: {qty_double} unidades (Double Cheese + Refri Lata) = {format_currency(qty_double * COMBO_2_PRECO)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if qty_jbc + qty_double == qty_lata:
        st.markdown(f"""
        <div style="background-color: #1e40af; border-left: 4px solid #60a5fa; 
                    padding: 15px; margin: 10px 0; border-radius: 8px;">
            <p style="color: #dbeafe; margin: 0; font-size: 15px;">
                ✅ Regra respeitada: Total de Sanduíches ({qty_jbc + qty_double}) = Total de Latas ({qty_lata})
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #92400e; border-left: 4px solid #f59e0b; 
                    padding: 15px; margin: 10px 0; border-radius: 8px;">
            <p style="color: #fef3c7; margin: 0; font-size: 15px;">
                ⚠️ Total de Sanduíches ({qty_jbc + qty_double}) ≠ Total de Latas ({qty_lata})
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    border-radius: 12px; padding: 20px; border: 2px solid #475569;">
            <h3 style="color: #fbbf24; margin: 0 0 15px 0; text-align: center;">🍔 Produtos</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if dados['sanduiches']:
            df_s = pd.DataFrame({
                'Produto': list(dados['sanduiches'].keys()),
                'Qnt': list(dados['sanduiches'].values()),
                'Preço Unitário': [CARDAPIOS["sanduiches"][k] for k in dados['sanduiches']],
                'Subtotal': [CARDAPIOS["sanduiches"][k]*v for k,v in dados['sanduiches'].items()]
            })
            df_s = df_s.sort_values('Subtotal', ascending=False)
            html_s = df_s.style.format({'Qnt':'{:.0f}', 'Preço Unitário':'R$ {:.2f}', 'Subtotal':'R$ {:.2f}'})\
                .set_table_styles(get_global_centered_styles()).hide(axis='index').to_html()
            st.markdown(html_s, unsafe_allow_html=True)
            st.write("")
            st.markdown(f"""
            <div style="background-color: #1e40af; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;">
                <p style="color: #ffffff; font-size: 18px; font-weight: bold; margin: 0;">
                    Total: {format_currency(dados['val_sand'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Sem itens")

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    border-radius: 12px; padding: 20px; border: 2px solid #475569;">
            <h3 style="color: #60a5fa; margin: 0 0 15px 0; text-align: center;">🥤 Bebidas</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if dados['bebidas']:
            df_b = pd.DataFrame({
                'Produto': list(dados['bebidas'].keys()),
                'Qnt': list(dados['bebidas'].values()),
                'Preço Unitário': [CARDAPIOS["bebidas"][k] for k in dados['bebidas']],
                'Subtotal': [CARDAPIOS["bebidas"][k]*v for k,v in dados['bebidas'].items()]
            })
            df_b = df_b.sort_values('Subtotal', ascending=False)
            html_b = df_b.style.format({'Qnt':'{:.0f}', 'Preço Unitário':'R$ {:.2f}', 'Subtotal':'R$ {:.2f}'})\
                .set_table_styles(get_global_centered_styles()).hide(axis='index').to_html()
            st.markdown(html_b, unsafe_allow_html=True)
            st.write("")
            st.markdown(f"""
            <div style="background-color: #1e40af; padding: 15px; border-radius: 8px; text-align: center; margin-top: 10px;">
                <p style="color: #ffffff; font-size: 18px; font-weight: bold; margin: 0;">
                    Total: {format_currency(dados['val_beb'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Sem itens")

    diff = dados['alvo'] - dados['val_total']
    
    cor_border = "#10b981" if diff == 0 else "#f97316"
    cor_bg = "#065f46" if diff == 0 else "#92400e"
    cor_text = "#d1fae5" if diff == 0 else "#fef3c7" 
    
    st.markdown("---")
    
    if qty_jbc > 0 or qty_double > 0:
        valor_combo1 = qty_jbc * COMBO_1_PRECO
        valor_combo2 = qty_double * COMBO_2_PRECO
        valor_cebolas = qty_cebola * 0.50
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                    border: 3px solid #60a5fa; border-radius: 15px; 
                    padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h4 style="margin:0 0 15px 0; color: #ffffff; text-align: center; font-size: 20px;">🎁 RESUMO DOS COMBOS</h4>
            {f'<p style="text-align: center; color: #dbeafe; margin: 8px 0; font-size: 16px;"><b>Combo 1:</b> {qty_jbc} unidade(s) × R$ 25,00 = <b>{format_currency(valor_combo1)}</b></p>' if qty_jbc > 0 else ''}
            {f'<p style="text-align: center; color: #dbeafe; margin: 8px 0; font-size: 16px;"><b>Combo 2:</b> {qty_double} unidade(s) × R$ 30,00 = <b>{format_currency(valor_combo2)}</b></p>' if qty_double > 0 else ''}
            {f'<p style="text-align: center; color: #fef3c7; margin: 8px 0; font-size: 15px;">+ {qty_cebola} Cebola(s) Adicional(is) × R$ 0,50 = <b>{format_currency(valor_cebolas)}</b></p>' if qty_cebola > 0 else ''}
            <p style="text-align: center; color: #10b981; margin: 15px 0 0 0; font-size: 18px; 
                      border-top: 2px solid #60a5fa; padding-top: 15px; font-weight: bold;">
                Total Combos: {format_currency(valor_combo1 + valor_combo2 + valor_cebolas)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {cor_bg} 0%, {cor_border} 100%); 
                border: 4px solid {cor_border}; border-radius: 20px; 
                padding: 30px; text-align: center; margin-top: 10px; 
                margin-bottom: 20px; box-shadow: 0 6px 12px rgba(0,0,0,0.4);">
        <h3 style="margin:0; color: #ffffff; font-family: sans-serif; font-size: 22px;">💰 VALOR TOTAL DA COMBINAÇÃO</h3>
        <p style="font-size: 48px; font-weight: 900; margin: 15px 0; color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            {format_currency(dados['val_total'])}
        </p>
        <p style="font-size: 16px; margin: 0; color: #f3f4f6;">
            Meta: <b style="color: #ffffff;">{format_currency(dados['alvo'])}</b> | 
            Diferença: <b style="color: {'#fca5a5' if diff != 0 else '#86efac'}">{format_currency(abs(diff))}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title=CONFIG["page_title"],
    layout=CONFIG["layout"],
    initial_sidebar_state=CONFIG["sidebar_state"]
)

# --- CSS GLOBAL MELHORADO ---
st.markdown("""
<style>
    /* Background principal */
    .stApp {
        background: linear-gradient(to bottom, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-size: cover;
        background-attachment: fixed;
        color: #f0f2f6;
    }

    /* Tabelas com melhor contraste */
    th, td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    
    div[data-testid="stTable"] table {
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Inputs com melhor visibilidade */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stDateInput, 
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important; 
        color: #f3f4f6 !important;
        border: 2px solid #475569 !important;
        font-size: 14px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #f3f4f6 !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background-color: #1e293b;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #3b82f6 !important;
    }

    /* Radio buttons - Menu de navegação */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        width: 100%;
        background-color: transparent;
        gap: 15px;
    }
    
    div[role="radiogroup"] label {
        background-color: transparent !important;
        border: none !important;
        padding: 5px 15px !important;
        margin: 0 !important;
        box-shadow: none !important;
        cursor: pointer;
        transition: all 0.3s ease;
        border-right: 2px solid #475569 !important;
        border-radius: 0 !important;
    }

    div[role="radiogroup"] label:last-child {
        border-right: none !important;
    }

    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 16px !important; 
        white-space: nowrap !important;
        font-weight: 500;
        margin: 0;
        padding-bottom: 2px;
        color: #94a3b8; 
        border-bottom: 2px solid transparent; 
    }

    div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {
        color: #60a5fa !important;
        border-bottom: 2px solid #60a5fa !important;
    }

    div[role="radiogroup"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
        font-weight: bold;
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #334155;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #f3f4f6 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 14px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 6px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* LOGO ANIMADA COM FAÍSCAS */
    .logo-container {
        position: relative;
        width: 400px;
        height: 400px;
        margin: 0 auto 20px auto;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 99999 !important;
    }

    .logo-animada {
        width: 400px;
        height: auto;
        position: relative;
        z-index: 20; 
        filter: drop-shadow(0 0 20px rgba(255, 69, 0, 0.5));
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
        pointer-events: none;
    }

    @keyframes steady-rise-high {
        0% {
            opacity: 0;
            transform: translateY(0) scale(0.5);
        }
        10% {
             opacity: 0.8; 
        }
        80% {
            opacity: 0.6; 
        }
        100% {
            opacity: 0; 
            transform: translateY(-550px) scale(0.1); 
        }
    }

    .s1 { bottom: 20px; left: 10%; animation: steady-rise-high 5s linear infinite; animation-delay: 0s; }
    .s2 { bottom: 10px; left: 20%; animation: steady-rise-high 6s linear infinite; animation-delay: 1.5s; }
    .s3 { bottom: 25px; left: 35%; animation: steady-rise-high 5.5s linear infinite; animation-delay: 3.0s; }
    .s4 { bottom: 15px; left: 50%; animation: steady-rise-high 4.5s linear infinite; animation-delay: 0.5s; }
    .s5 { bottom: 5px;  left: 65%; animation: steady-rise-high 5.2s linear infinite; animation-delay: 2.2s; }
    .s6 { bottom: 12px; left: 80%; animation: steady-rise-high 4.8s linear infinite; animation-delay: 1.2s; }
    .s7 { bottom: 18px; left: 90%; animation: steady-rise-high 5.8s linear infinite; animation-delay: 2.8s; }
    .s8 { bottom: 8px;  left: 30%; animation: steady-rise-high 5.0s linear infinite; animation-delay: 0.8s; }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #475569;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Success, warning, error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 8px;
        border-left: 4px solid;
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

# --- LOGO ANIMADA CORRIGIDA ---
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
    <div class='logo-container' style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                                       border-radius: 20px; padding: 40px; box-shadow: 0 8px 16px rgba(0,0,0,0.4);'>
        <h1 style='color: #ffffff; font-size: 80px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🍔</h1>
        <p style='color: #ffffff; font-size: 32px; font-weight: bold; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>CLIPS BURGER</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"⚠️ Logo não encontrada: {e}")

st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h2 style='color: #3b82f6; margin: 0; font-size: 32px; font-weight: bold;'>🍔 Sistema de Gestão - Clips Burger</h2>
    <p style='color: #94a3b8; margin-top: 10px; font-size: 16px;'>Análise inteligente de vendas e combinações</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0; text-align: center;'>⚙️ Configurações</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🧬 **Algoritmo Genético com 2 Combos**")
    
    population_size = st.slider(
        "Tamanho da População", 
        min_value=20, 
        max_value=200, 
        value=100, 
        step=10,
        help="Maior população = melhor resultado, mas mais lento"
    )
    
    generations = st.slider(
        "Número de Gerações", 
        min_value=10, 
        max_value=500, 
        value=200, 
        step=10,
        help="Mais gerações = melhor convergência"
    )
    
    st.divider()
    
    st.markdown("""
    <div style='background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;'>
        <p style='color: #fbbf24; margin: 0; font-weight: bold;'>⚠️ Regra Principal:</p>
        <p style='color: #e2e8f0; margin: 5px 0 0 0;'><b>Total Sanduíches = Total Latas</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #1e293b; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #10b981;'>
        <p style='color: #d1fae5; margin: 5px 0;'>🎁 <b>Combo 1:</b> JBC + Lata = R$ 25,00</p>
        <p style='color: #d1fae5; margin: 5px 0;'>🎁 <b>Combo 2:</b> Double + Lata = R$ 30,00</p>
        <p style='color: #fef3c7; margin: 5px 0;'>🧅 <b>Cebola:</b> Ajuste fino (R$ 0,50)</p>
    </div>
    """, unsafe_allow_html=True)

# --- MENU DE NAVEGAÇÃO ---
menu_opcoes = ["📈 Resumo das Vendas", "🧩 Análise com Arquivo", "💸 Calculadora PIX"]
escolha_menu = st.radio("Navegação", menu_opcoes, horizontal=True, label_visibility="collapsed", key="nav_menu")

# --- CONTEÚDO DAS ABAS ---

if escolha_menu == "📈 Resumo das Vendas":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid #60a5fa;'>
        <h2 style='color: white; margin: 0; text-align: center;'>📤 Upload de Dados</h2>
    </div>
    """, unsafe_allow_html=True)
    
    arquivo = st.file_uploader("Envie o arquivo de transações (.csv ou .xlsx)", 
                             type=["csv", "xlsx"])
    
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
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                        padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #60a5fa;'>
                <h2 style='color: white; margin: 0; text-align: center;'>📊 Visualização de Dados</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Total de Vendas por Forma de Pagamento")
            bar_chart = create_altair_chart(
                vendas, 'bar', 'Forma', 'Valor', 'Forma',
                title=''
            ).properties(
                width=800,
                height=500
            )
            st.altair_chart(bar_chart, use_container_width=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                        padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #60a5fa;'>
                <h2 style='color: white; margin: 0; text-align: center;'>⚙️ Parâmetros Financeiros</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                salario_minimo = st.number_input("Salário Mínimo (R$)", value=1518.0, step=50.0)
            with col2:
                custo_contadora = st.number_input("Custo com Contadora (R$)", value=316.0, step=10.0)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                        padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #60a5fa;'>
                <h2 style='color: white; margin: 0; text-align: center;'>💰 Resultados Financeiros</h2>
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
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                        padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #60a5fa;'>
                <h2 style='color: white; margin: 0; text-align: center;'>🔍 Detalhamento</h2>
            </div>
            """, unsafe_allow_html=True)
            
            tab_detalhes1, tab_detalhes2, tab_detalhes3 = st.tabs([
                "📝 Composição de Custos", 
                "📚 Explicação dos Cálculos",
                "🍰 Gráfico de Composição"
            ])
            
            with tab_detalhes1:
                st.subheading("Composição dos Custos")
                st.markdown(f"""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 2px solid #475569;'>
                    <p style='color: #f3f4f6; font-size: 16px; margin: 10px 0;'>
                        <b style='color: #60a5fa;'>• Imposto Simples Nacional (6%):</b> {format_currency(imposto_simples)}
                    </p>
                    <p style='color: #f3f4f6; font-size: 16px; margin: 10px 0;'>
                        <b style='color: #60a5fa;'>• Custo Funcionário CLT:</b> {format_currency(custo_funcionario)}
                    </p>
                    <p style='color: #f3f4f6; font-size: 16px; margin: 10px 0;'>
                        <b style='color: #60a5fa;'>• Custo Contadora:</b> {format_currency(custo_contadora)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_detalhes2:
                st.subheader("Fórmulas Utilizadas")
                st.markdown("""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 2px solid #475569;'>
                    <p style='color: #f3f4f6; font-size: 15px; margin: 10px 0;'>
                        <b style='color: #fbbf24;'>1. Imposto Simples Nacional:</b><br>
                        <code style='background-color: #334155; padding: 4px 8px; border-radius: 4px; color: #e2e8f0;'>
                        Faturamento Bruto × 6%
                        </code>
                    </p>
                    <p style='color: #f3f4f6; font-size: 15px; margin: 10px 0;'>
                        <b style='color: #fbbf24;'>2. Custo Funcionário CLT:</b><br>
                        <code style='background-color: #334155; padding: 4px 8px; border-radius: 4px; color: #e2e8f0;'>
                        Salário + FGTS (8%) + Férias (1 mês + 1/3) + 13º Salário
                        </code>
                    </p>
                    <p style='color: #f3f4f6; font-size: 15px; margin: 10px 0;'>
                        <b style='color: #fbbf24;'>3. Total de Custos:</b><br>
                        <code style='background-color: #334155; padding: 4px 8px; border-radius: 4px; color: #e2e8f0;'>
                        Imposto + Funcionário + Contadora
                        </code>
                    </p>
                    <p style='color: #f3f4f6; font-size: 15px; margin: 10px 0;'>
                        <b style='color: #fbbf24;'>4. Lucro Estimado:</b><br>
                        <code style='background-color: #334155; padding: 4px 8px; border-radius: 4px; color: #e2e8f0;'>
                        Faturamento Bruto - Total de Custos
                        </code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_detalhes3:
                st.subheader("Composição dos Custos")
                custos_df = pd.DataFrame({
                    'Item': ['Impostos', 'Funcionário', 'Contadora'],
                    'Valor': [imposto_simples, custo_funcionario, custo_contadora]
                })
                
                graf_composicao = alt.Chart(custos_df).mark_arc().encode(
                    theta='Valor',
                    color='Item',
                    tooltip=['Item', alt.Tooltip('Valor', format='$.2f')]
                ).properties(
                    width=600,
                    height=500
                )
                st.altair_chart(graf_composicao, use_container_width=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                        padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #60a5fa;'>
                <h2 style='color: white; margin: 0; text-align: center;'>📑 Relatório</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Gerar Relatório PDF", type="primary", use_container_width=True):
                with st.spinner("Gerando relatório..."):
                    pdf_buffer = create_pdf_report(
                        df, vendas, total_vendas, imposto_simples, custo_funcionario, 
                        custo_contadora, total_custos, lucro_estimado, CONFIG["logo_path"]
                    )
                    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                    pdf_display = f'<a href="data:application/pdf;base64,{b64_pdf}" download="relatorio_clips_burger.pdf" style="display: inline-block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 15px 30px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 16px;">📥 Clique aqui para baixar o Relatório PDF</a>'
                    st.markdown(f'<div style="text-align: center; margin: 20px 0;">{pdf_display}</div>', unsafe_allow_html=True)
                    st.success("✅ Relatório gerado com sucesso!")
            
        except Exception as e:
            st.error(f"❌ Ocorreu um erro ao processar o arquivo: {str(e)}")
            st.exception(e)
    else:
        st.info("ℹ️ Aguardando upload do arquivo de transações.")

elif escolha_menu == "🧩 Análise com Arquivo":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid #60a5fa;'>
        <h2 style='color: white; margin: 0; text-align: center;'>🧩 Análise de Combinações por Arquivo</h2>
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
        
        if st.button("🔎 Analisar Combinação", type="primary", use_container_width=True):
            with st.spinner("Calculando a melhor combinação com 2 Combos..."):
                dados = gerar_dados_geneticos_two_combos(
                    valor_selecionado, 
                    population_size, 
                    generations
                )
                st.session_state.resultado_arquivo = dados
        
        if st.session_state.resultado_arquivo:
            st.divider()
            renderizar_resultados(st.session_state.resultado_arquivo)
        
    else:
        st.info("ℹ️ Faça o upload de dados na aba 'Resumo das Vendas' primeiro.")

elif escolha_menu == "💸 Calculadora PIX":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid #60a5fa;'>
        <h2 style='color: white; margin: 0; text-align: center;'>💸 Calculadora Rápida (PIX/Manual)</h2>
        <p style='color: #dbeafe; text-align: center; margin: 10px 0 0 0;'>
            Digite um valor e descubra quantos Combos 1 + Combos 2 + Cebolas formam esse total
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_action = st.columns([0.4, 0.6])
    
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
        if st.button("🚀 Calcular Combinação", type="primary", use_container_width=True):
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
        st.divider()
        renderizar_resultados(st.session_state.resultado_pix)

# Rodapé
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 14px; padding: 20px 0;'>
        © 2025 Clips Burger - Sistema de Gestão de Combos | Desenvolvido com ❤️ e Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)
