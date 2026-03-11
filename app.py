import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gearbox Sales — Data Science",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #2563EB, #10B981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #6B7280; font-size: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.08); border-left: 4px solid;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #6B7280; margin: 0; }
    .kpi-delta { font-size: 0.8rem; font-weight: 600; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1F2937;
        border-bottom: 2px solid #E5E7EB; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #F0FDF4; border-left: 4px solid #10B981;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0;
        color: #065F46; font-size: 0.9rem;
    }
    .warning-box {
        background: #FFF7ED; border-left: 4px solid #F59E0B;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0;
        color: #92400E; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Market Region']  = df['Market Region'].str.strip()
    df['Date']           = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01')
    df['Total_sales']    = df['REMAN Gross sales'] + df['NEW Gross sales']
    df['Total_qty']      = df['REMAN Quantity'] + df['NEW Quantity']
    df['Prix_moy_REMAN'] = (df['REMAN Gross sales'] / df['REMAN Quantity']).round(2)
    df['Prix_moy_NEW']   = (df['NEW Gross sales']   / df['NEW Quantity']).round(2)
    df['Ratio_REMAN']    = (df['REMAN Gross sales'] / df['Total_sales'] * 100).round(2)
    df['CA_par_piece']   = (df['Total_sales'] / df['Total_qty']).round(2)
    df['Premium_index']  = (df['Prix_moy_NEW'] / df['Prix_moy_REMAN']).round(3)
    return df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Gearbox Sales Dashboard")
    st.markdown("---")

    # Chargement automatique du fichier inclus dans le repo
    DEFAULT_FILE = "Gearbox_data_Test POwer BI_07 25.xlsx"
    import os

    if os.path.exists(DEFAULT_FILE):
        df = load_data(DEFAULT_FILE)
        st.success(f"✅ {len(df)} lignes chargées automatiquement")
    else:
        # Fallback : upload manuel si le fichier n'est pas trouvé
        uploaded = st.file_uploader("📂 Charger les données (.xlsx)", type=['xlsx'])
        if uploaded:
            df = load_data(uploaded)
            st.success(f"✅ {len(df)} lignes chargées")
        else:
            st.info("⬆️ Upload le fichier Excel pour commencer")
            st.markdown("**Format attendu :**")
            st.markdown("- Year, Month\n- Market Region\n- Product type\n- REMAN / NEW Quantity & Gross sales")
            st.stop()

    st.markdown("---")
    st.markdown("### 🎛️ Filtres")

    regions = st.multiselect(
        "🌍 Régions", df['Market Region'].unique().tolist(),
        default=df['Market Region'].unique().tolist()
    )
    years = st.multiselect(
        "📅 Années", sorted(df['Year'].unique().tolist()),
        default=sorted(df['Year'].unique().tolist())
    )
    prod_types = st.multiselect(
        "🔩 Type produit", df['Product type'].unique().tolist(),
        default=df['Product type'].unique().tolist()
    )

    st.markdown("---")
    st.markdown("### 📋 Navigation")
    page = st.radio("", ["📊 Vue Générale", "📈 Temporel", "🤖 Modèle ML", "🔵 Clustering"])

# ── FILTER ────────────────────────────────────────────────────────────────────
df_f = df[
    df['Market Region'].isin(regions) &
    df['Year'].isin(years) &
    df['Product type'].isin(prod_types)
].copy()

if df_f.empty:
    st.warning("⚠️ Aucune donnée avec ces filtres.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔧 Gearbox Sales — Analyse Data Science</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ventes de boîtes de vitesses IVECO/ZF | REMAN vs NEW | France · Italie · Allemagne | 2023–2024</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Vue Générale":

    # KPIs
    ca_total  = df_f['Total_sales'].sum()
    ca_reman  = df_f['REMAN Gross sales'].sum()
    ca_new    = df_f['NEW Gross sales'].sum()
    qty_total = df_f['Total_qty'].sum()
    nb_refs   = df_f['Part Number'].nunique()
    premium   = df_f['Premium_index'].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "CA Total", f"{ca_total/1e6:.2f}M€", "#2563EB"),
        (c2, "CA REMAN", f"{ca_reman/1e3:.0f}k€", "#10B981"),
        (c3, "CA NEW",   f"{ca_new/1e3:.0f}k€",   "#F59E0B"),
        (c4, "Unités vendues", f"{qty_total:,}", "#EF4444"),
        (c5, "Références", f"{nb_refs}", "#8B5CF6"),
        (c6, "Premium NEW/REMAN", f"{premium:.2f}x", "#06B6D4"),
    ]
    for col, label, value, color in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{color}">
                <p class="kpi-label">{label}</p>
                <p class="kpi-value" style="color:{color}">{value}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">📊 Répartition du CA</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        ca_region = df_f.groupby('Market Region')[['REMAN Gross sales','NEW Gross sales']].sum().reset_index()
        ca_region_m = ca_region.melt(id_vars='Market Region', var_name='Type', value_name='CA')
        fig = px.bar(ca_region_m, x='Market Region', y='CA', color='Type',
                     title="CA par région — REMAN vs NEW",
                     color_discrete_map={'REMAN Gross sales':'#2563EB','NEW Gross sales':'#10B981'},
                     labels={'CA':'Chiffre d\'affaires (€)'},
                     barmode='group', template='plotly_white')
        fig.update_layout(legend_title_text='Type', height=380)
        fig.update_yaxes(tickformat=',.0f', ticksuffix='€')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            values=[ca_reman, ca_new],
            names=['REMAN','NEW'],
            title="Mix REMAN / NEW (CA global)",
            color_discrete_sequence=['#2563EB','#10B981'],
            hole=0.45,
            template='plotly_white'
        )
        fig2.update_traces(textposition='outside', textinfo='percent+label',
                           pull=[0.03, 0.03])
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        ca_prod = df_f.groupby('Part Description')['Total_sales'].sum().sort_values(ascending=True).reset_index()
        fig3 = px.bar(ca_prod, x='Total_sales', y='Part Description',
                      orientation='h', title="CA par type de pièce",
                      color='Total_sales', color_continuous_scale='Blues',
                      template='plotly_white',
                      labels={'Total_sales':'CA (€)', 'Part Description':'Pièce'})
        fig3.update_xaxes(tickformat=',.0f', ticksuffix='€')
        fig3.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(df_f, x='Market Region', y='Prix_moy_REMAN',
                      color='Market Region',
                      title="Distribution du prix unitaire REMAN par région",
                      template='plotly_white',
                      color_discrete_sequence=['#2563EB','#10B981','#F59E0B'],
                      labels={'Prix_moy_REMAN':'Prix unitaire (€)'})
        fig4.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-title">💡 Insights clés</p>', unsafe_allow_html=True)
    top_reg = df_f.groupby('Market Region')['Total_sales'].sum().idxmax()
    st.markdown(f'<div class="insight-box">🥇 <b>1er marché :</b> {top_reg} — concentre le plus gros CA sur la période sélectionnée</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">💰 <b>Premium NEW/REMAN :</b> {premium:.2f}x — la pièce neuve coûte en moyenne {(premium-1)*100:.0f}% plus cher que le reconditionné</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="warning-box">⚠️ <b>Levier identifié :</b> Le segment REMAN est sous-exploité en Allemagne — potentiel de croissance significatif</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TEMPOREL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Temporel":

    st.markdown('<p class="section-title">📈 Évolution temporelle du CA</p>', unsafe_allow_html=True)

    monthly = df_f.groupby('Date').agg(
        CA_REMAN=('REMAN Gross sales','sum'),
        CA_NEW=('NEW Gross sales','sum'),
        Total=('Total_sales','sum'),
        Qty=('Total_qty','sum')
    ).reset_index().sort_values('Date')
    monthly['MM3']       = monthly['Total'].rolling(3, center=True).mean()
    monthly['CA_cumul']  = monthly['Total'].cumsum()
    monthly['Mois_label']= monthly['Date'].dt.strftime('%b %Y')

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("CA mensuel REMAN + NEW",
                                        "Tendance + Moyenne Mobile 3 mois",
                                        "CA Cumulé",
                                        "Quantités vendues / mois"),
                        vertical_spacing=0.14, horizontal_spacing=0.08)

    fig.add_trace(go.Bar(x=monthly['Mois_label'], y=monthly['CA_REMAN'],
                         name='REMAN', marker_color='#2563EB', opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly['Mois_label'], y=monthly['CA_NEW'],
                         name='NEW', marker_color='#10B981', opacity=0.85), row=1, col=1)

    fig.add_trace(go.Scatter(x=monthly['Mois_label'], y=monthly['Total'],
                             mode='lines+markers', name='CA mensuel',
                             line=dict(color='#2563EB', width=1.5),
                             marker=dict(size=6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=monthly['Mois_label'], y=monthly['MM3'],
                             mode='lines', name='MM 3 mois',
                             line=dict(color='#EF4444', width=2.5, dash='dash')), row=1, col=2)

    fig.add_trace(go.Scatter(x=monthly['Mois_label'], y=monthly['CA_cumul'],
                             mode='lines', fill='tozeroy', name='Cumulé',
                             line=dict(color='#8B5CF6', width=2),
                             fillcolor='rgba(139,92,246,0.1)'), row=2, col=1)

    fig.add_trace(go.Bar(x=monthly['Mois_label'], y=monthly['Qty'],
                         name='Quantités', marker_color='#F59E0B', opacity=0.85), row=2, col=2)

    fig.update_layout(height=650, template='plotly_white', barmode='stack',
                      legend=dict(orientation='h', y=-0.08),
                      title_text="Analyse temporelle — Ventes Gearbox 2023/2024",
                      title_font_size=14)
    st.plotly_chart(fig, use_container_width=True)

    # 2023 vs 2024
    st.markdown('<p class="section-title">📊 Comparaison 2023 vs 2024</p>', unsafe_allow_html=True)

    df_23 = df_f[df_f['Year']==2023].groupby('Month')['Total_sales'].sum()
    df_24 = df_f[df_f['Year']==2024].groupby('Month')['Total_sales'].sum()
    common = sorted(set(df_23.index) & set(df_24.index))
    mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']

    if common:
        growth = [(df_24[m]-df_23[m])/df_23[m]*100 for m in common]
        col1, col2 = st.columns(2)
        with col1:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(x=[mois_labels[m-1] for m in common],
                                     y=[df_23[m] for m in common],
                                     name='2023', marker_color='#2563EB', opacity=0.8))
            fig_cmp.add_trace(go.Bar(x=[mois_labels[m-1] for m in common],
                                     y=[df_24[m] for m in common],
                                     name='2024', marker_color='#EF4444', opacity=0.8))
            fig_cmp.update_layout(barmode='group', title='CA mensuel 2023 vs 2024',
                                  template='plotly_white', height=360)
            st.plotly_chart(fig_cmp, use_container_width=True)

        with col2:
            colors_g = ['#10B981' if g >= 0 else '#EF4444' for g in growth]
            fig_g = go.Figure(go.Bar(
                x=[mois_labels[m-1] for m in common], y=growth,
                marker_color=colors_g, text=[f"{g:+.1f}%" for g in growth],
                textposition='outside'
            ))
            fig_g.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_g.update_layout(title='Croissance 2024 vs 2023 (%)',
                                template='plotly_white', height=360)
            st.plotly_chart(fig_g, use_container_width=True)

        ca_23 = df_f[df_f['Year']==2023]['Total_sales'].sum()
        ca_24 = df_f[df_f['Year']==2024]['Total_sales'].sum()
        proj  = ca_24 / len(df_24) * 12
        st.markdown(f'<div class="insight-box">📈 <b>Projection 2024 annualisée :</b> {proj:,.0f}€ '
                    f'({(proj-ca_23)/ca_23*100:+.1f}% vs CA 2023 de {ca_23:,.0f}€)</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Sélectionne les deux années dans les filtres pour afficher la comparaison.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODÈLE ML
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modèle ML":

    st.markdown('<p class="section-title">🤖 Modèle de prédiction du CA</p>', unsafe_allow_html=True)

    @st.cache_resource
    def train_model(data_hash):
        df_m = df.copy()
        le_r = LabelEncoder(); le_p = LabelEncoder(); le_d = LabelEncoder()
        df_m['Region_enc']   = le_r.fit_transform(df_m['Market Region'])
        df_m['ProdType_enc'] = le_p.fit_transform(df_m['Product type'])
        df_m['Desc_enc']     = le_d.fit_transform(df_m['Part Description'])
        df_m['Mois_sin']     = np.sin(2*np.pi*df_m['Month']/12)
        df_m['Mois_cos']     = np.cos(2*np.pi*df_m['Month']/12)
        df_m['Is_2024']      = (df_m['Year']==2024).astype(int)
        df_m['Prix_gap']     = df_m['Prix_moy_NEW'] - df_m['Prix_moy_REMAN']
        df_m['Qty_ratio']    = df_m['REMAN Quantity'] / (df_m['Total_qty']+1e-6)
        df_m = df_m.sort_values('Date').reset_index(drop=True)

        FEATS = ['Month','Mois_sin','Mois_cos','Is_2024','Region_enc','ProdType_enc',
                 'Desc_enc','REMAN Quantity','NEW Quantity','Total_qty',
                 'Prix_moy_REMAN','Prix_moy_NEW','Prix_gap','Ratio_REMAN','Qty_ratio','Premium_index']
        X = df_m[FEATS]; y = df_m['Total_sales']
        split = int(len(X)*0.8)
        X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
        model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return model, X_te, y_te, y_pred, FEATS, le_r, le_p, le_d

    model, X_te, y_te, y_pred, FEATS, le_r, le_p, le_d = train_model(len(df))

    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card" style="border-color:#2563EB"><p class="kpi-label">MAE (Erreur moyenne)</p><p class="kpi-value" style="color:#2563EB">{mae:,.0f}€</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card" style="border-color:#10B981"><p class="kpi-label">R² Score</p><p class="kpi-value" style="color:#10B981">{r2:.3f}</p></div>', unsafe_allow_html=True)
    with c3:
        baseline = mean_absolute_error(y_te, np.full(len(y_te), y_te.mean()))
        amelio = (baseline-mae)/baseline*100
        st.markdown(f'<div class="kpi-card" style="border-color:#F59E0B"><p class="kpi-label">Amélioration vs baseline</p><p class="kpi-value" style="color:#F59E0B">{amelio:.1f}%</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_sc = px.scatter(x=y_te, y=y_pred, labels={'x':'CA Réel (€)','y':'CA Prédit (€)'},
                            title="Réel vs Prédit — Random Forest",
                            template='plotly_white', opacity=0.7,
                            color_discrete_sequence=['#2563EB'])
        fig_sc.add_shape(type='line', x0=y_te.min(), y0=y_te.min(),
                         x1=y_te.max(), y1=y_te.max(),
                         line=dict(color='red', dash='dash', width=1.5))
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        residuals = y_te.values - y_pred
        fig_res = px.histogram(x=residuals, nbins=25,
                               title="Distribution des résidus",
                               labels={'x':'Erreur (€)','y':'Fréquence'},
                               template='plotly_white',
                               color_discrete_sequence=['#10B981'])
        fig_res.add_vline(x=0, line_dash='dash', line_color='red')
        fig_res.add_vline(x=residuals.mean(), line_dash='dot',
                          line_color='orange',
                          annotation_text=f"Moy: {residuals.mean():.0f}€")
        fig_res.update_layout(height=380)
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance
    imp = pd.DataFrame({'Feature': FEATS, 'Importance': model.feature_importances_})\
            .sort_values('Importance', ascending=True)
    fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance — Random Forest",
                     color='Importance', color_continuous_scale='Blues',
                     template='plotly_white')
    fig_imp.update_layout(height=480, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Simulateur
    st.markdown('<p class="section-title">🎯 Simulateur de CA</p>', unsafe_allow_html=True)
    st.markdown("Renseigne les paramètres d'une transaction pour estimer le CA :")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_region   = st.selectbox("🌍 Région", df['Market Region'].unique())
        sim_prodtype = st.selectbox("🔩 Type", df['Product type'].unique())
        sim_month    = st.slider("📅 Mois", 1, 12, 6)
    with s2:
        sim_reman_qty = st.number_input("📦 Quantité REMAN", 1, 20, 3)
        sim_new_qty   = st.number_input("📦 Quantité NEW",   1, 10, 1)
        sim_year      = st.selectbox("📅 Année", [2023, 2024])
    with s3:
        sim_prix_reman = st.number_input("💰 Prix unit. REMAN (€)", 500, 15000, 2500)
        sim_prix_new   = st.number_input("💰 Prix unit. NEW (€)",   500, 20000, 3500)

    if st.button("🚀 Estimer le CA", use_container_width=True):
        total_qty   = sim_reman_qty + sim_new_qty
        ratio_reman = sim_reman_qty / total_qty
        prix_gap    = sim_prix_new - sim_prix_reman
        premium_idx = sim_prix_new / sim_prix_reman
        desc_enc    = 0

        try:
            region_enc   = le_r.transform([sim_region])[0]
            prodtype_enc = le_p.transform([sim_prodtype])[0]
        except:
            region_enc = 0; prodtype_enc = 0

        X_sim = pd.DataFrame([{
            'Month': sim_month, 'Mois_sin': np.sin(2*np.pi*sim_month/12),
            'Mois_cos': np.cos(2*np.pi*sim_month/12), 'Is_2024': int(sim_year==2024),
            'Region_enc': region_enc, 'ProdType_enc': prodtype_enc, 'Desc_enc': desc_enc,
            'REMAN Quantity': sim_reman_qty, 'NEW Quantity': sim_new_qty,
            'Total_qty': total_qty, 'Prix_moy_REMAN': sim_prix_reman,
            'Prix_moy_NEW': sim_prix_new, 'Prix_gap': prix_gap,
            'Ratio_REMAN': ratio_reman*100, 'Qty_ratio': ratio_reman,
            'Premium_index': premium_idx
        }])

        pred_ca = model.predict(X_sim)[0]
        ca_calc = sim_reman_qty * sim_prix_reman + sim_new_qty * sim_prix_new

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🤖 CA estimé (modèle ML)", f"{pred_ca:,.0f}€")
        with col_b:
            st.metric("🧮 CA calculé (prix × qté)", f"{ca_calc:,.0f}€",
                      delta=f"{pred_ca-ca_calc:+,.0f}€ vs ML")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":

    st.markdown('<p class="section-title">🔵 Segmentation des références produits</p>', unsafe_allow_html=True)

    prod_agg = df_f.groupby('Part Number').agg(
        CA_total       = ('Total_sales','sum'),
        CA_REMAN       = ('REMAN Gross sales','sum'),
        CA_NEW         = ('NEW Gross sales','sum'),
        Qty_total      = ('Total_qty','sum'),
        Prix_moy_REMAN = ('Prix_moy_REMAN','mean'),
        Prix_moy_NEW   = ('Prix_moy_NEW','mean'),
        Premium_index  = ('Premium_index','mean'),
        Ratio_REMAN    = ('Ratio_REMAN','mean'),
        Nb_trans       = ('Total_sales','count')
    ).reset_index()

    cl_feats = ['CA_total','Qty_total','Prix_moy_REMAN','Prix_moy_NEW',
                'Ratio_REMAN','Premium_index','Nb_trans']
    X_cl = prod_agg[cl_feats].fillna(0)
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_cl)

    k = st.slider("Nombre de segments (k)", 2, min(8, len(prod_agg)-1), 3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    prod_agg['Cluster'] = km.fit_predict(X_sc)

    sorted_idx = prod_agg.groupby('Cluster')['CA_total'].mean().sort_values().index.tolist()
    segment_names = ['🔵 Faible rotation','🟡 Standard','🟠 Dynamique','🔴 Top ventes',
                     '⚫ Ultra-premium','🟢 Volume','🟣 Spécial','🩵 Niche']
    labels_map = {sorted_idx[i]: segment_names[i] for i in range(k)}
    prod_agg['Segment'] = prod_agg['Cluster'].map(labels_map)

    col1, col2 = st.columns(2)
    with col1:
        ca_seg = prod_agg.groupby('Segment')['CA_total'].sum().sort_values(ascending=False).reset_index()
        fig_seg = px.bar(ca_seg, x='Segment', y='CA_total',
                         title="CA total par segment",
                         color='Segment', template='plotly_white',
                         labels={'CA_total':'CA (€)'},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_seg.update_yaxes(tickformat=',.0f', ticksuffix='€')
        fig_seg.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        fig_bubble = px.scatter(
            prod_agg, x='Qty_total', y='CA_total',
            color='Segment', size='Nb_trans',
            hover_data=['Part Number','Prix_moy_REMAN','Prix_moy_NEW'],
            title="Références — Volume vs CA (taille = nb transactions)",
            template='plotly_white',
            labels={'Qty_total':'Quantités totales','CA_total':'CA total (€)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bubble.update_yaxes(tickformat=',.0f', ticksuffix='€')
        fig_bubble.update_layout(height=380)
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown('<p class="section-title">📋 Profil des segments</p>', unsafe_allow_html=True)
    profile = prod_agg.groupby('Segment')[cl_feats].mean().round(0)
    profile.columns = ['CA moy (€)','Qté moy','Prix REMAN (€)','Prix NEW (€)',
                       'Ratio REMAN (%)','Premium index','Nb transactions']
    st.dataframe(profile.style.background_gradient(cmap='Blues', subset=['CA moy (€)']),
                 use_container_width=True)

    st.markdown('<p class="section-title">🔍 Détail des références par segment</p>', unsafe_allow_html=True)
    seg_select = st.selectbox("Choisir un segment", sorted(prod_agg['Segment'].unique()))
    refs = prod_agg[prod_agg['Segment']==seg_select][
        ['Part Number','CA_total','Qty_total','Prix_moy_REMAN','Prix_moy_NEW','Nb_trans']
    ].sort_values('CA_total', ascending=False).reset_index(drop=True)
    refs.columns = ['Référence','CA Total (€)','Quantités','Prix REMAN (€)','Prix NEW (€)','Transactions']
    st.dataframe(refs.style.format({'CA Total (€)':'{:,.0f}','Prix REMAN (€)':'{:,.0f}','Prix NEW (€)':'{:,.0f}'}),
                 use_container_width=True)

    top_ref = refs.iloc[0]['Référence'] if len(refs) > 0 else "N/A"
    st.markdown(f'<div class="insight-box">🔑 <b>Référence phare de ce segment :</b> {top_ref} — à prioriser dans la stratégie stock et commerciale</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#9CA3AF; font-size:0.8rem;'>"
    "🔧 Gearbox Sales Dashboard · Projet Data Science · 2025"
    "</center>",
    unsafe_allow_html=True
)
