# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from supabase import create_client
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
import plotly.io as pio

# Configuración de página
st.set_page_config(
    page_title="Gold Trading AI Advisor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cliente Supabase
@st.cache_resource
def init_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-buy {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .recommendation-sell {
        background-color: #F44336;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .recommendation-hold {
        background-color: #FF9800;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FFD700;
    }
    .profit-card {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
    .loss-card {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #F44336;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def trigger_analysis(prediction_horizon, investment_amount, risk_tolerance):
    """Disparar análisis en n8n"""
    n8n_webhook_url = st.secrets["N8N_WEBHOOK_URL"]
    payload = {
        "prediction_horizon": prediction_horizon,
        "investment_amount": investment_amount,
        "risk_tolerance": risk_tolerance
    }
    try:
        response = requests.post(n8n_webhook_url, json=payload, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def fetch_historical_recommendations():
    """Obtener recomendaciones históricas de Supabase"""
    supabase = init_supabase()
    
    try:
        response = supabase.table('gold_trading_recommendations')\
                         .select('*')\
                         .order('processed_at', desc=True)\
                         .limit(50)\
                         .execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def calculate_investment_projection(current_price, predicted_price, investment_amount):
    """Calcular proyección de inversión"""
    # Calcular onzas de oro que se pueden comprar
    ounces = investment_amount / current_price
    
    # Valor futuro proyectado
    future_value = ounces * predicted_price
    
    # Ganancia/pérdida
    profit_loss = future_value - investment_amount
    profit_loss_percent = (profit_loss / investment_amount) * 100
    
    return {
        'ounces': ounces,
        'current_value': investment_amount,
        'future_value': future_value,
        'profit_loss': profit_loss,
        'profit_loss_percent': profit_loss_percent
    }

def create_price_chart(chart_data, current_price, predicted_price, prediction_horizon_str="7 days"):
    """Crear gráfico interactivo de precios"""
    if not chart_data:
        return None

    def _parse_days(s):
        if isinstance(s, str):
            digits = "".join([c for c in s if c.isdigit()])
            if digits:
                return int(digits)
        return 7

    horizon_days = _parse_days(prediction_horizon_str)

    df = pd.DataFrame(chart_data)
    df['date'] = pd.to_datetime(df['date'])

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Evolución del Precio del Oro', 'Análisis Técnico'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    # Gráfico principal de precios
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['price'],
                  name='Precio Actual', line=dict(color='#FFD700', width=3)),
        row=1, col=1
    )

    # Media móvil
    if 'moving_average' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['moving_average'],
                      name='Media Móvil (5 días)', line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )

    # Precio predicho
    last_date = df['date'].max()
    predicted_date = last_date + timedelta(days=horizon_days)

    fig.add_trace(
        go.Scatter(x=[last_date, predicted_date],
                  y=[current_price, predicted_price],
                  name=f'Predicción IA ({horizon_days}d)',
                  line=dict(color='#4ECDC4', width=2, dash='dot')),
        row=1, col=1
    )

    # Volumen/indicadores
    price_changes = df['price'].diff().fillna(0)
    fig.add_trace(
        go.Bar(x=df['date'], y=price_changes,
               name='Cambio Diario',
               marker_color=['#4CAF50' if x >= 0 else '#F44336' for x in price_changes]),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white',
        title_x=0.5
    )

    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cambio USD", row=2, col=1)

    return fig

def create_volatility_chart(chart_data):
    """Crear gráfico de volatilidad"""
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calcular volatilidad rodante
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['volatility'],
        name='Volatilidad (5 días)',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig.update_layout(
        title='Análisis de Volatilidad del Mercado',
        xaxis_title='Fecha',
        yaxis_title='Volatilidad (%)',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_rsi_chart(chart_data):
    """Crear gráfico de RSI (Relative Strength Index)"""
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calcular RSI simplificado
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rsi'],
        name='RSI',
        line=dict(color='#9C27B0', width=2)
    ))
    
    # Líneas de referencia
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobreventa")
    
    fig.update_layout(
        title='Índice de Fuerza Relativa (RSI)',
        xaxis_title='Fecha',
        yaxis_title='RSI',
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_investment_projection_chart(projection):
    """Crear gráfico de proyección de inversión"""
    fig = go.Figure()
    
    categories = ['Inversión Inicial', 'Valor Proyectado']
    values = [projection['current_value'], projection['future_value']]
    colors = ['#2196F3', '#4CAF50' if projection['profit_loss'] > 0 else '#F44336']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f"${v:,.2f}" for v in values],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Proyección de Inversión',
        yaxis_title='Valor (USD)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_price_distribution_chart(chart_data):
    """Crear histograma de distribución de precios"""
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['price'],
        nbinsx=20,
        name='Distribución de Precios',
        marker_color='#FFD700'
    ))
    
    fig.update_layout(
        title='Distribución de Precios Históricos',
        xaxis_title='Precio (USD)',
        yaxis_title='Frecuencia',
        template='plotly_white',
        height=400
    )
    
    return fig

def generate_pdf_report(result, projection):
    """Generar reporte PDF del análisis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#FFD700'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12
    )
    
    # Título
    story.append(Paragraph("🥇 Gold Trading AI Advisor", title_style))
    story.append(Paragraph(f"Reporte de Análisis - {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recomendación principal
    recommendation = result.get('ai_recommendation', 'HOLD')
    rec_color = {
        'BUY': colors.green,
        'SELL': colors.red,
        'HOLD': colors.orange
    }.get(recommendation, colors.orange)
    
    rec_data = [[f"RECOMENDACIÓN: {recommendation}"]]
    rec_table = Table(rec_data, colWidths=[6*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), rec_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 20))
    
    # Métricas principales
    story.append(Paragraph("Métricas del Mercado", heading_style))
    
    metrics_data = [
        ['Métrica', 'Valor'],
        ['Precio Actual', f"${result.get('current_price', 0):.2f}"],
        ['Precio Predicho', f"${result.get('predicted_price', 0):.2f}"],
        ['Cambio Esperado', f"{result.get('price_change_percent', 0):.2f}%"],
        ['Confianza IA', f"{result.get('confidence_score', 0) * 100:.1f}%"],
        ['Nivel de Riesgo', result.get('risk_level', 'MEDIUM')],
        ['Volatilidad', f"{result.get('volatility', 0) * 100:.1f}%"],
        ['Horizonte Predicción', result.get('prediction_horizon', 'N/A')]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Proyección de inversión
    story.append(Paragraph("Proyección de Inversión", heading_style))
    
    profit_color = colors.green if projection['profit_loss'] > 0 else colors.red
    
    investment_data = [
        ['Concepto', 'Valor'],
        ['Inversión Inicial', f"${projection['current_value']:,.2f}"],
        ['Onzas de Oro', f"{projection['ounces']:.4f} oz"],
        ['Valor Proyectado', f"${projection['future_value']:,.2f}"],
        ['Ganancia/Pérdida', f"${projection['profit_loss']:,.2f}"],
        ['Rentabilidad', f"{projection['profit_loss_percent']:.2f}%"]
    ]
    
    investment_table = Table(investment_data, colWidths=[3*inch, 3*inch])
    investment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('BACKGROUND', (0, 4), (-1, -1), profit_color),
        ('TEXTCOLOR', (0, 4), (-1, -1), colors.whitesmoke),
        ('FONTNAME', (0, 4), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(investment_table)
    story.append(Spacer(1, 20))
    
    # Explicación
    story.append(Paragraph("Explicación del Análisis", heading_style))
    
    explanation = {
        "BUY": "El modelo detecta una tendencia alcista fuerte con alta confianza. Los indicadores técnicos y el análisis de patrones sugieren potencial de crecimiento.",
        "SELL": "Se identifica una tendencia bajista con indicadores de sobrecompra. El modelo recomienda considerar toma de ganancias o protección.",
        "HOLD": "El mercado muestra lateralidad o señales mixtas. Se recomienda esperar señales más claras antes de tomar posición."
    }.get(recommendation, "Análisis en proceso...")
    
    story.append(Paragraph(explanation, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Descargo de Responsabilidad", heading_style))
    story.append(Paragraph(
        "Este análisis es generado por inteligencia artificial y tiene fines informativos únicamente. "
        "No constituye asesoramiento financiero. Las inversiones en commodities conllevan riesgos significativos. "
        "Consulte siempre con un asesor financiero certificado antes de tomar decisiones de inversión.",
        styles['Normal']
    ))
    
    # Construir PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_recommendation(result):
    """Mostrar recomendación con estilo"""
    recommendation = result.get('ai_recommendation', 'HOLD')
    confidence = result.get('confidence_score', 0) * 100
    current_price = result.get('current_price', 0)
    predicted_price = result.get('predicted_price', 0)
    
    st.markdown(f"## 🎯 Recomendación: {recommendation}")
    
    if recommendation == "BUY":
        st.markdown('<div class="recommendation-buy">📈 COMPRAR ORO</div>', unsafe_allow_html=True)
    elif recommendation == "SELL":
        st.markdown('<div class="recommendation-sell">📉 VENDER ORO</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-hold">⚖️ MANTENER POSICIÓN</div>', unsafe_allow_html=True)
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Precio Actual", 
            f"${current_price:.2f}", 
            f"{result.get('price_change_percent', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "Confianza IA", 
            f"{confidence:.1f}%"
        )
    
    with col3:
        st.metric(
            "Precio Predicho", 
            f"${predicted_price:.2f}",
            f"{(predicted_price - current_price):.2f} USD"
        )
    
    with col4:
        risk_level = result.get('risk_level', 'MEDIUM')
        risk_color = {
            'LOW': '🟢',
            'MEDIUM': '🟡', 
            'HIGH': '🔴'
        }.get(risk_level, '🟡')
        st.metric("Nivel de Riesgo", f"{risk_color} {risk_level}")

def display_investment_projection(projection):
    """Mostrar proyección de inversión"""
    st.markdown("### 💰 Proyección de tu Inversión")
    
    is_profit = projection['profit_loss'] > 0
    card_class = "profit-card" if is_profit else "loss-card"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0;">{"📈 Ganancia Proyectada" if is_profit else "📉 Pérdida Proyectada"}</h3>
            <h2 style="margin:0;">${abs(projection['profit_loss']):,.2f}</h2>
            <p style="font-size:1.2rem; margin:0;">({projection['profit_loss_percent']:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Detalles de la Inversión</h4>
            <p><b>Inversión Inicial:</b> ${projection['current_value']:,.2f}</p>
            <p><b>Onzas de Oro:</b> {projection['ounces']:.4f} oz</p>
            <p><b>Valor Proyectado:</b> ${projection['future_value']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🥇 Gold Trading AI Advisor</h1>', unsafe_allow_html=True)
    st.markdown("### Sistema de Trading Inteligente con Redes Neuronales Profundas")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración de Análisis")

        horizon_label_to_value = {
            "1 día": "1d",
            "2 días": "2d",
            "3 días": "3d"
        }
        horizon_label = st.selectbox(
            "Horizonte de predicción:",
            list(horizon_label_to_value.keys()),
            index=1,
            help="Selecciona a qué distancia en el futuro quieres predecir"
        )
        prediction_horizon = horizon_label_to_value[horizon_label]

        investment_amount = st.number_input(
            "Monto de Inversión (USD):",
            min_value=100,
            max_value=1000000,
            value=5000,
            step=500,
            help="Cantidad que planeas invertir"
        )

        risk_tolerance = st.selectbox(
            "Tolerancia al Riesgo:",
            ["low", "medium", "high"],
            index=1,
            help="Tu perfil de riesgo como inversionista"
        )

        if st.button("🚀 Ejecutar Análisis con IA", type="primary", use_container_width=True):
            with st.spinner("🤖 Analizando tendencias con modelo híbrido CNN-LSTM..."):
                success, result = trigger_analysis(prediction_horizon, investment_amount, risk_tolerance)
                if success:
                    st.session_state.last_analysis = result
                    st.session_state.investment_amount = investment_amount
                    st.success("✅ Análisis completado!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
        
        st.markdown("---")
        st.header("📊 Datos Históricos")
        if st.button("Cargar Análisis Anteriores", use_container_width=True):
            with st.spinner("Cargando historial..."):
                st.session_state.historical_data = fetch_historical_recommendations()
    
    # Contenido principal
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Actual", "📊 Gráficos Avanzados", "📜 Histórico", "ℹ️ Información"])
    
    with tab1:
        if 'last_analysis' in st.session_state:
            result = st.session_state.last_analysis
            display_recommendation(result)
            
            # Calcular proyección de inversión
            investment_amount = st.session_state.get('investment_amount', result.get('investment_amount', 5000))
            projection = calculate_investment_projection(
                result.get('current_price', 0),
                result.get('predicted_price', 0),
                investment_amount
            )
            
            # Mostrar proyección
            display_investment_projection(projection)
            
            # Gráfico de proyección de inversión
            st.plotly_chart(create_investment_projection_chart(projection), use_container_width=True)
            
            # Botón para generar PDF
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📄 Generar Reporte PDF", type="secondary", use_container_width=True):
                    with st.spinner("Generando reporte..."):
                        pdf_buffer = generate_pdf_report(result, projection)
                        st.download_button(
                            label="⬇️ Descargar Reporte PDF",
                            data=pdf_buffer,
                            file_name=f"gold_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            
            # Gráfico de precios
            st.markdown("---")
            st.markdown("### 📊 Análisis de Tendencia")
            chart_fig = create_price_chart(
                result.get('chart_data', []),
                result.get('current_price', 0),
                result.get('predicted_price', 0),
                result.get('prediction_horizon', prediction_horizon)
            )
            
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
            
            # Análisis detallado
            st.markdown("### 🔍 Análisis Detallado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Métricas de Mercado")
                st.markdown(f"""
                <div class="metric-card">
                <b>Volatilidad:</b> {(result.get('volatility', 0) * 100):.1f}%<br/>
                <b>Puntos de Datos:</b> {result.get('data_points', 0)}<br/>
                <b>Timeframe:</b> {result.get('timeframe', 'N/A')}<br/>
                <b>Horizonte Predicción:</b> {result.get('prediction_horizon', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Análisis de Riesgo")
                risk_score = result.get('confidence_score', 0.5)
                
                st.markdown(f"""
                <div class="metric-card">
                <b>Inversión Analizada:</b> ${investment_amount:,.0f}<br/>
                <b>Score Confianza:</b> {(risk_score * 100):.1f}%<br/>
                <b>Riesgo Calculado:</b> {result.get('risk_level', 'MEDIUM')}<br/>
                <b>Recomendación:</b> {result.get('ai_recommendation', 'HOLD')}
                </div>
                """, unsafe_allow_html=True)
            
            # Explicación de la recomendación
            st.markdown("### 🤖 Explicación de la IA")
            
            recommendation = result.get('ai_recommendation', 'HOLD')
            explanation = {
                "BUY": "El modelo detecta una tendencia alcista fuerte con alta confianza. Los indicadores técnicos y el análisis de patrones sugieren potencial de crecimiento.",
                "SELL": "Se identifica una tendencia bajista con indicadores de sobrecompra. El modelo recomienda considerar toma de ganancias o protección.",
                "HOLD": "El mercado muestra lateralidad o señales mixtas. Se recomienda esperar señales más claras antes de tomar posición."
            }.get(recommendation, "Análisis en proceso...")
            
            st.info(f"**{explanation}**")
            
        else:
            st.info("👈 Configura los parámetros y ejecuta un análisis en el panel lateral")
    
    with tab2:
        st.header("📊 Análisis Gráfico Avanzado")
        
        if 'last_analysis' in st.session_state:
            result = st.session_state.last_analysis
            chart_data = result.get('chart_data', [])
            
            if chart_data:
                # Gráfico de volatilidad
                st.markdown("### 📉 Análisis de Volatilidad")
                volatility_fig = create_volatility_chart(chart_data)
                if volatility_fig:
                    st.plotly_chart(volatility_fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico RSI
                    st.markdown("### 📊 Índice de Fuerza Relativa (RSI)")
                    rsi_fig = create_rsi_chart(chart_data)
                    if rsi_fig:
                        st.plotly_chart(rsi_fig, use_container_width=True)
                
                with col2:
                    # Distribución de precios
                    st.markdown("### 📈 Distribución de Precios")
                    dist_fig = create_price_distribution_chart(chart_data)
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True)
                
                # Análisis de tendencias
                st.markdown("### 📐 Análisis de Tendencias")
                df = pd.DataFrame(chart_data)
                df['date'] = pd.to_datetime(df['date'])
                
                # Calcular tendencias
                df['returns'] = df['price'].pct_change()
                positive_days = len(df[df['returns'] > 0])
                negative_days = len(df[df['returns'] < 0])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Días Alcistas", positive_days, f"{(positive_days/len(df)*100):.1f}%")
                with col2:
                    st.metric("Días Bajistas", negative_days, f"{(negative_days/len(df)*100):.1f}%")
                with col3:
                    avg_return = df['returns'].mean() * 100
                    st.metric("Retorno Promedio Diario", f"{avg_return:.3f}%")
                with col4:
                    max_drawdown = (df['price'].min() / df['price'].max() - 1) * 100
                    st.metric("Máxima Caída", f"{max_drawdown:.2f}%")
                
                # Gráfico de retornos acumulados
                st.markdown("### 💹 Retornos Acumulados")
                df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['cumulative_returns'] * 100,
                    name='Retorno Acumulado',
                    line=dict(color='#2196F3', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.3)'
                ))
                
                fig.update_layout(
                    title='Evolución de Retornos Acumulados',
                    xaxis_title='Fecha',
                    yaxis_title='Retorno Acumulado (%)',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No hay datos de gráficos disponibles para análisis avanzado")
        else:
            st.info("👈 Ejecuta un análisis primero para ver los gráficos avanzados")
    
    with tab3:
        st.header("📈 Historial de Recomendaciones")
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            df = st.session_state.historical_data
            
            # Métricas históricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Análisis", len(df))
            with col2:
                buy_recommendations = len(df[df['ai_recommendation'] == 'BUY'])
                st.metric("Recomendaciones COMPRA", buy_recommendations)
            with col3:
                sell_recommendations = len(df[df['ai_recommendation'] == 'SELL'])
                st.metric("Recomendaciones VENTA", sell_recommendations)
            with col4:
                accuracy = len(df[df['confidence_score'] > 0.7]) / len(df) * 100
                st.metric("Alta Confianza", f"{accuracy:.1f}%")
            
            # Gráfico de evolución de recomendaciones
            df['processed_at'] = pd.to_datetime(df['processed_at'])
            
            fig = px.scatter(df, x='processed_at', y='current_price', 
                           color='ai_recommendation',
                           size='confidence_score',
                           title='Evolución de Recomendaciones y Precios',
                           color_discrete_map={
                               'BUY': '#4CAF50',
                               'SELL': '#F44336', 
                               'HOLD': '#FF9800'
                           },
                           hover_data=['predicted_price', 'risk_level'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribución de recomendaciones
            st.markdown("### 📊 Distribución de Recomendaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de pie
                rec_counts = df['ai_recommendation'].value_counts()
                fig_pie = px.pie(
                    values=rec_counts.values,
                    names=rec_counts.index,
                    title='Distribución de Recomendaciones',
                    color=rec_counts.index,
                    color_discrete_map={
                        'BUY': '#4CAF50',
                        'SELL': '#F44336',
                        'HOLD': '#FF9800'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Gráfico de niveles de riesgo
                risk_counts = df['risk_level'].value_counts()
                fig_risk = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title='Distribución de Niveles de Riesgo',
                    color=risk_counts.index,
                    color_discrete_map={
                        'LOW': '#4CAF50',
                        'MEDIUM': '#FF9800',
                        'HIGH': '#F44336'
                    }
                )
                fig_risk.update_layout(
                    xaxis_title='Nivel de Riesgo',
                    yaxis_title='Cantidad',
                    showlegend=False
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Análisis de precisión
            st.markdown("### 🎯 Análisis de Precisión")
            
            avg_confidence = df['confidence_score'].mean() * 100
            max_confidence = df['confidence_score'].max() * 100
            min_confidence = df['confidence_score'].min() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confianza Promedio", f"{avg_confidence:.1f}%")
            with col2:
                st.metric("Confianza Máxima", f"{max_confidence:.1f}%")
            with col3:
                st.metric("Confianza Mínima", f"{min_confidence:.1f}%")
            
            # Tabla detallada
            st.markdown("### 📋 Registro Detallado")
            display_columns = [
                'processed_at', 'ai_recommendation', 'current_price', 
                'predicted_price', 'confidence_score', 'risk_level', 
                'investment_amount'
            ]
            
            # Verificar qué columnas existen
            available_columns = [col for col in display_columns if col in df.columns]
            
            df_display = df[available_columns].copy()
            df_display['processed_at'] = df_display['processed_at'].dt.strftime('%Y-%m-%d %H:%M')
            if 'confidence_score' in df_display.columns:
                df_display['confidence_score'] = df_display['confidence_score'].apply(lambda x: f"{x*100:.1f}%")
            if 'current_price' in df_display.columns:
                df_display['current_price'] = df_display['current_price'].apply(lambda x: f"${x:.2f}")
            if 'predicted_price' in df_display.columns:
                df_display['predicted_price'] = df_display['predicted_price'].apply(lambda x: f"${x:.2f}")
            if 'investment_amount' in df_display.columns:
                df_display['investment_amount'] = df_display['investment_amount'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(
                df_display.sort_values('processed_at', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Ejecuta 'Cargar Análisis Anteriores' para ver el historial")
    
    with tab4:
        st.header("ℹ️ Acerca del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧠 Tecnologías Utilizadas
            
            **Backend (n8n):**
            - Automatización del flujo de datos
            - Integración con APIs de precios de oro
            - Comunicación con modelo de IA
            - Almacenamiento en Supabase
            
            **Inteligencia Artificial:**
            - Modelo híbrido CNN-LSTM
            - Análisis de series temporales
            - Redes neuronales profundas
            - Análisis técnico automatizado
            
            **Frontend (Streamlit):**
            - Visualización interactiva en tiempo real
            - Gráficos con Plotly
            - Dashboard responsive
            - Análisis histórico
            - Generación de reportes PDF
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Indicadores Técnicos
            
            **RSI (Relative Strength Index):**
            - Mide la fuerza de los movimientos de precio
            - Valores > 70: Sobrecompra
            - Valores < 30: Sobreventa
            
            **Volatilidad:**
            - Mide la variabilidad del precio
            - Mayor volatilidad = Mayor riesgo
            - Calculada sobre ventanas móviles
            
            **Media Móvil:**
            - Suaviza las fluctuaciones de precio
            - Identifica tendencias
            - Base para otros indicadores
            
            **Retornos Acumulados:**
            - Muestra la rentabilidad histórica
            - Ayuda a evaluar desempeño
            - Considera el efecto compuesto
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### ⚠️ Descargo de Responsabilidad
        
        Este sistema es una herramienta de análisis y **NO constituye asesoramiento financiero**. 
        
        **Consideraciones importantes:**
        - Las predicciones se basan en datos históricos y no garantizan resultados futuros
        - Las inversiones en commodities como el oro conllevan riesgos significativos
        - Los mercados son impredecibles y pueden cambiar rápidamente
        - Las proyecciones de ganancia/pérdida son estimaciones basadas en el modelo
        - La confianza del modelo no elimina el riesgo inherente de inversión
        
        **Recomendaciones:**
        - Diversifica tu portafolio de inversiones
        - No inviertas dinero que no puedas permitirte perder
        - Consulta con un asesor financiero certificado
        - Realiza tu propia investigación (DYOR)
        - Considera tu perfil de riesgo y objetivos financieros
        
        ### 🔧 Configuración Técnica
        
        El sistema requiere:
        - API Key de MetalPriceAPI
        - Instancia de n8n ejecutándose
        - Base de datos Supabase configurada
        - Servidor Python con modelo de IA entrenado
        - Conexión estable a internet
        
        ### 📞 Soporte
        
        Para reportar problemas o sugerencias:
        - Revisa la documentación técnica
        - Verifica la configuración de servicios externos
        - Asegúrate de que las API Keys sean válidas
        - Comprueba la conectividad de red
        """)
        
        st.markdown("---")
        
        # Footer con métricas del sistema
        st.markdown("### 📈 Estado del Sistema")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Versión", "2.0.0")
        with col2:
            st.metric("Modelo IA", "CNN-LSTM")
        with col3:
            st.metric("Framework", "Streamlit")
        with col4:
            st.metric("Estado", "🟢 Activo")

if __name__ == "__main__":
    main()