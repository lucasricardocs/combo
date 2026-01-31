# =========================================================
# CLIPS BURGER — SISTEMA FINAL
# Arquivo único | Design final | GA preservado
# =========================================================

import streamlit as st
import random
import base64
from PIL import Image
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Clips Burger | Otimizador",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_PATH = "logo.png"

# =========================================================
# CSS FINAL (PRODUÇÃO)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;900&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(180deg, #0c0c0c 0%, #171717 100%);
}

.header {
    display:flex;
    align-items:center;
    gap:20px;
    margin-bottom:32px;
}

.header img {
    width:90px;
}

.section {
    background:#161616;
    border-radius:20px;
    padding:28px;
    margin-bottom:24px;
    box-shadow:0 10px 30px rgba(0,0,0,.35);
}

.card {
    background:#202020;
    border-radius:18px;
    padding:22px;
    box-shadow:0 6px 20px rgba(0,0,0,.4);
}

.total {
    font-size:52px;
    font-weight:900;
}

.subtle {
    color:#9ca3af;
}

.stButton>button {
    background: linear-gradient(135deg, #FF6B35, #F7931E);
    border-radius:14px;
    padding:14px 30px;
    font-weight:700;
    border:none;
}

.stButton>button:hover {
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# UTIL
# =========================================================
def img_base64(path):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def moeda(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =========================================================
# HEADER
# =========================================================
logo64 = img_base64(LOGO_PATH)

st.markdown(
    f"""
    <div class="header">
        {f'<img src="data:image/png;base64,{logo64}">' if logo64 else ''}
        <div>
            <h1>🍔 Clips Burger</h1>
            <p class="subtle">Otimizador inteligente de combos</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎯 Parâmetros")
    alvo = st.number_input("Valor alvo (R$)", min_value=1.0, step=1.0)

    st.markdown("---")
    st.markdown("### ℹ️ Regras")
    st.markdown("- Combos respeitam sanduíche = refri")
    st.markdown("- Cebola usada como ajuste fino")
    st.markdown("- Algoritmo genético otimiza a diferença")

# =========================================================
# 🔒 ALGORITMO GENÉTICO — COLE O SEU AQUI
# =========================================================
"""
⚠️ IMPORTANTE:
Cole AQUI exatamente o seu algoritmo genético original,
sem alterar NENHUMA linha.

Exemplo:
- create_individual_two_combos
- evaluate_fitness_two_combos
- genetic_algorithm_two_combos
- buscar_combinacao_two_combos
"""

# def buscar_combinacao_two_combos(alvo):
#     ...
#     return resultado

# =========================================================
# MAIN
# =========================================================
if alvo > 0:
    if st.button("🧬 Calcular melhor combinação"):
        # 👉 chamada do SEU algoritmo
        resultado = buscar_combinacao_two_combos(alvo)

        # 👉 total calculado exatamente como no seu código
        total = resultado["total"]

        st.markdown("## Resultado")

        c1, c2, c3 = st.columns(3)

        c1.markdown(f"""
        <div class="card">
            <p>🍔 Combo JBC</p>
            <h2>{resultado["jbc"]}x</h2>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="card">
            <p>🧀 Combo Double</p>
            <h2>{resultado["double"]}x</h2>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="card">
            <p>🧅 Cebola extra</p>
            <h2>{resultado["cebola"]}x</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section" style="text-align:center;">
            <p class="subtle">Valor total</p>
            <div class="total">{moeda(total)}</div>
            <p class="subtle">Meta: {moeda(alvo)}</p>
        </div>
        """, unsafe_allow_html=True)
