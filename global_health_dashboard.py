"""
Global Health Explorer Dashboard
Dimensionality Reduction Analysis (1990-2019)
Assessment 3 - Data Analytics and Visualisation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# Page configuration
st.set_page_config(
    page_title="Global Health Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');
    
    .main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); }
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #e8e8e8 !important; }
    
    .main-header {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(120deg, #E63946, #2A9D8F, #E9C46A);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header { color: #a0a0a0 !important; font-size: 1.1rem; margin-bottom: 2rem; }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .cluster-badge {
        display: inline-block; padding: 0.25rem 0.75rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .cluster-0 { background: #E63946; color: white; }
    .cluster-1 { background: #2A9D8F; color: white; }
    .cluster-2 { background: #E9C46A; color: #0f0f23; }
    .cluster-3 { background: #6A4C93; color: white; }
    
    div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ============== LOAD DATA ==============
@st.cache_data
def load_data():
    try:
        with open('dashboard_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ `dashboard_data.json` not found. Please run the export script in your notebook first.")
        st.code("""
# Add this to your notebook after dimensionality reduction:
# See export_dashboard_data.py for the full code
        """)
        st.stop()

data = load_data()

# Create DataFrame
df = pd.DataFrame({
    'Country': data['countries'],
    'Cluster': data['clusters'],
    'Cluster_Name': [data['cluster_labels'][str(c)] for c in data['clusters']]
})

# Add embeddings
for method in data['embeddings']:
    df[f'{method}_x'] = data['embeddings'][method]['x']
    df[f'{method}_y'] = data['embeddings'][method]['y']

# Add indicators
for indicator, values in data.get('indicators', {}).items():
    df[indicator] = values

# Cluster colors matching notebook
cluster_colors = {
    0: '#E63946',  # Developed - Red
    1: '#2A9D8F',  # Least Developed - Teal  
    2: '#E9C46A',  # Emerging - Yellow
    3: '#6A4C93'   # Developing - Purple
}

cluster_color_map = {
    'Developed Nations': '#E63946',
    'Least Developed': '#2A9D8F',
    'Emerging Economies': '#E9C46A',
    'Developing Nations': '#6A4C93'
}

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    
    # Get available methods
    methods = list(data['embeddings'].keys())
    method = st.radio("**Projection Method**", methods)
    
    st.markdown("---")
    
    # Color options
    color_options = ["Cluster"] + list(data.get('indicators', {}).keys())
    color_by = st.selectbox("**Color By**", color_options)
    
    st.markdown("---")
    
    selected_country = st.selectbox("**🔍 Find Country**", [""] + sorted(data['countries']))
    
    st.markdown("---")
    st.markdown("**Filter Clusters**")
    show_clusters = {int(i): st.checkbox(name, value=True, key=f"c{i}") 
                     for i, name in data['cluster_labels'].items()}
    
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size: 0.8rem; color: #888;'>
    <b>Countries:</b> {len(data['countries'])}<br>
    <b>Methods:</b> {', '.join(methods)}<br>
    <b>Data:</b> WHO, World Bank, UN
    </div>
    """, unsafe_allow_html=True)

# ============== MAIN CONTENT ==============
st.markdown('<h1 class="main-header">🌍 Global Health Explorer</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Dimensionality Reduction Analysis • {len(data["countries"])} Countries • 1990-2019</p>', unsafe_allow_html=True)

# Filter data
mask = df['Cluster'].isin([c for c, show in show_clusters.items() if show])
df_filtered = df[mask].copy()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### {method} Projection")
    
    x_col, y_col = f'{method}_x', f'{method}_y'
    
    if color_by == "Cluster":
        fig = px.scatter(
            df_filtered, x=x_col, y=y_col, 
            color='Cluster_Name',
            color_discrete_map=cluster_color_map,
            hover_name='Country',
            hover_data={x_col: False, y_col: False}
        )
    else:
        fig = px.scatter(
            df_filtered, x=x_col, y=y_col,
            color=color_by,
            color_continuous_scale='RdYlGn',
            hover_name='Country',
            hover_data={x_col: False, y_col: False}
        )
    
    # Highlight selected country
    if selected_country:
        cd = df[df['Country'] == selected_country]
        if len(cd) > 0:
            fig.add_trace(go.Scatter(
                x=cd[x_col], y=cd[y_col],
                mode='markers+text',
                marker=dict(size=20, color='white', line=dict(width=3, color='#ff6b6b')),
                text=[selected_country],
                textposition='top center',
                textfont=dict(size=14, color='white'),
                showlegend=False
            ))
    
    # Styling
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='rgba(0,0,0,0.3)')))
    
    # Axis labels
    x_label = f"{method} Dim 1"
    y_label = f"{method} Dim 2"
    if method == 'PCA' and 'variance_explained' in data:
        x_label += f" ({data['variance_explained']['PC1']}%)"
        y_label += f" ({data['variance_explained']['PC2']}%)"
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e8e8'),
        xaxis=dict(title=x_label, gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)'),
        yaxis=dict(title=y_label, gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)'),
        height=550,
        margin=dict(l=60, r=20, t=40, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Cluster Profiles")
    
    for cid in range(4):
        if show_clusters.get(cid, True):
            name = data['cluster_labels'][str(cid)]
            count = sum(1 for c in data['clusters'] if c == cid)
            color = cluster_colors[cid]
            
            avg_data = data.get('cluster_averages', {}).get(str(cid), {})
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="cluster-badge cluster-{cid}">{name}</span>
                <span style="color:#888;float:right;">{count} countries</span>
                <div style="margin-top:0.5rem;font-size:0.9rem;">
            """, unsafe_allow_html=True)
            
            for ind_name, ind_val in avg_data.items():
                if ind_val is not None:
                    if 'GDP' in ind_name:
                        display = f"${ind_val:,.0f}"
                    else:
                        display = f"{ind_val:.1f}"
                    st.markdown(f"""<span style="color:#888;">{ind_name}:</span> <span style="color:{color}">{display}</span><br>""", unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# Country detail
if selected_country:
    st.markdown("---")
    st.markdown(f"### 📍 {selected_country}")
    
    idx = data['countries'].index(selected_country)
    cid = data['clusters'][idx]
    
    cols = st.columns(3)
    cols[0].metric("Cluster", data['cluster_labels'][str(cid)])
    
    i = 1
    for ind_name, ind_values in data.get('indicators', {}).items():
        if i < len(cols):
            val = ind_values[idx]
            if val is not None:
                if 'GDP' in ind_name:
                    cols[i].metric(ind_name, f"${val:,.0f}")
                else:
                    cols[i].metric(ind_name, f"{val:.1f}")
            else:
                cols[i].metric(ind_name, "N/A")
            i += 1

# Method comparison
st.markdown("---")
st.markdown("### 🔬 Method Comparison")

method_cols = st.columns(len(methods))
descriptions = {
    'PCA': 'Linear, global variance',
    't-SNE': 'Non-linear, local focus',
    'UMAP': 'Local & global balance',
    'Isomap': 'Geodesic distances',
    'LLE': 'Local reconstruction'
}

for col, m in zip(method_cols, methods):
    with col:
        x_c, y_c = f'{m}_x', f'{m}_y'
        
        fig_small = px.scatter(
            df_filtered, x=x_c, y=y_c,
            color='Cluster',
            color_discrete_sequence=['#E63946', '#2A9D8F', '#E9C46A', '#6A4C93'],
            hover_name='Country',
            title=m
        )
        fig_small.update_traces(marker=dict(size=6))
        fig_small.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8e8e8', size=10),
            xaxis=dict(showticklabels=False, title='', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showticklabels=False, title='', gridcolor='rgba(255,255,255,0.05)'),
            height=200,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_small, use_container_width=True)
        st.caption(descriptions.get(m, ''))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <b>Assessment 3</b> • Data Analytics and Visualisation (RAI-7002) • 2024<br>
    Analysis using PCA, t-SNE, UMAP, Isomap, and LLE
</div>
""", unsafe_allow_html=True)
