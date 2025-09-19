import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

### Config
st.set_page_config(
    page_title="Getaround Analysis",
    page_icon="🚗",
    layout="wide"
)

# URLs des données (approche professionnelle)
DATA_URL_DELAY = 'https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx'
DATA_URL_PRICING = 'https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv'

### App
st.title("Analyse GetAround - Optimisation des delais entre reservations")

st.markdown("""
    Dashboard interactif pour analyser l'impact des delais minimums entre locations.
    Fournit des metriques cles pour optimiser la strategie de delais.
""")

today = datetime.now().strftime("%d/%m/%Y")
st.caption(f"Rapport genere le **{today}**")

st.markdown("---")

@st.cache_data
def load_data():
    try:
        delay = pd.read_excel(DATA_URL_DELAY)
        pricing = pd.read_csv(DATA_URL_PRICING)
        return delay, pricing
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None, None

st.subheader("Integration des donnees")
data_load_state = st.text('Chargement des fichiers...')
delay, pricing = load_data()
data_load_state.text("")

if delay is None or pricing is None:
    st.stop()

if st.checkbox('Afficher un apercu des donnees'):
    st.markdown('Dataset des retards:')
    st.write(delay.head())
    st.markdown('Dataset des prix:')
    st.write(pricing.head())

st.markdown("---")

# Creating the data processing
delay_prevRent = delay[pd.notna(delay["previous_ended_rental_id"])].copy().reset_index(drop=True)
delay_prevRent['previous_ended_rental_id'] = [int(x) for x in delay_prevRent['previous_ended_rental_id']]
delay_prevRent['previous_delay_at_checkout'] = [
    delay[delay['rental_id'] == prev_car].delay_at_checkout_in_minutes.values[0]
    for prev_car in delay_prevRent['previous_ended_rental_id']]

delay_prevRent_woNaN = delay_prevRent[pd.notna(delay_prevRent["previous_delay_at_checkout"])].copy().reset_index(drop=True)
delay_prevRent_woNaN['timedelta_minus_delay'] = delay_prevRent_woNaN['time_delta_with_previous_rental_in_minutes'] - delay_prevRent_woNaN['previous_delay_at_checkout']

delay_prevRent_woNaN_previousdelay = delay_prevRent[pd.notna(delay_prevRent["previous_delay_at_checkout"])].copy().reset_index(drop=True)
delay_prevRent_woNaN_previousdelay['type_of_delay'] = delay_prevRent_woNaN_previousdelay['previous_delay_at_checkout'].apply(lambda x: 'in_advance' if x<=0 else 'late')

st.subheader("Resultats cles de l'analyse")
st.markdown("Metriques principales basees sur notre analyse approfondie:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Locations touchees (threshold 60min)", value="358")
    st.caption("CONNECT: 156 | MOBILE: 202")
with col2:
    st.metric(label="Pourcentage de locations touchees", value="1.98%")
    st.caption("Impact sur le total des locations")
with col3:
    st.metric(label="Locations avec retard", value="270 (1.5%)")
    st.caption("Taux de retard moyen observe")
with col4:
    st.metric(label="Problemes evites", value="176")
    st.caption("CONNECT: 63 | MOBILE: 113")

st.markdown("---")

st.subheader("Configuration du delai minimum")
st.markdown("""
    Outils interactifs pour determiner le threshold optimal et le type de checkin approprie.
    Analyse des impacts sur les annulations et les revenus.
""")

# Question 1: Impact des retards
st.markdown("**Analyse 1: Repartition des retards vs arrivees anticipees**")
col1, col2 = st.columns(2)
with col1:
    df_pie = pd.DataFrame(delay_prevRent_woNaN_previousdelay['type_of_delay'].value_counts())
    fig = px.pie(df_pie, values=df_pie['count'], names=df_pie.index, color=df_pie.index, 
             color_discrete_map={'late': 'red', 'in_advance': 'green'}, 
             title="Distribution: Retards vs Avance")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(delay_prevRent_woNaN_previousdelay, 'state', pattern_shape='checkin_type', 
             color='type_of_delay', text_auto=True,
             title='Impact sur le locataire suivant par type de checkin')
    st.plotly_chart(fig, use_container_width=True)

# ECDF charts for threshold selection
st.markdown("**Analyse 2: Fonctions de repartition pour selection du threshold**")

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.ecdf(delay_prevRent_woNaN, color='state', x='timedelta_minus_delay', 
              title="ECDF - Tous types de checkin")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.ecdf(delay_prevRent_woNaN.loc[delay_prevRent_woNaN['checkin_type']=='mobile',:], 
              color='state', x='timedelta_minus_delay', title='ECDF - Checkin Mobile')
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.ecdf(delay_prevRent_woNaN.loc[delay_prevRent_woNaN['checkin_type']=='connect',:], 
              color='state', x='timedelta_minus_delay', title='ECDF - Checkin Connect')
    st.plotly_chart(fig, use_container_width=True)

# Interactive selectors 
threshold = st.selectbox("Choisir le threshold (en minutes)", range(0,720))
checkin_type = st.selectbox("Choisir le type de checkin", ['connect', 'mobile', 'both'])

delay_prevRent_woNaN_previousdelay['threshold'] = delay_prevRent_woNaN_previousdelay['time_delta_with_previous_rental_in_minutes'].apply(lambda x: 'below_threshold' if x<=threshold else 'above_threshold')

st.markdown("---")

st.subheader("Impact des parametres selectionnes") 
st.markdown("Resultats detailles selon le threshold et type de checkin choisis")

# Question 2: Rentals affected
st.markdown("**Impact 1: Nombre de locations affectees**")
if checkin_type == 'both':
    delay_question2 = delay.copy()
else:
    delay_question2 = delay.loc[delay['checkin_type']==checkin_type, :].copy()

delay_question2['threshold'] = delay_question2['time_delta_with_previous_rental_in_minutes'].apply(lambda x: 'below_threshold' if x<=threshold else 'above_threshold')

fig = px.pie(delay_question2['threshold'].value_counts(), values='count', 
             names=delay_question2['threshold'].value_counts().index, 
             color=delay_question2['threshold'].value_counts().index, 
             color_discrete_map={'below_threshold': 'red', 'above_threshold': 'green'}, 
             title="Repartition des locations selon le threshold choisi")
st.plotly_chart(fig, use_container_width=True)

value_metric_question2 = round(delay_question2['threshold'].value_counts()['below_threshold']/delay.shape[0]*100, ndigits=2)
st.metric(label="Pourcentage de locations potentiellement perdues", value=f"{value_metric_question2}%")

# Question 3: Problematic cases solved
st.markdown("**Impact 2: Cas problematiques resolus**")
if checkin_type == 'both':
    delay_question3 = delay_prevRent_woNaN_previousdelay.loc[delay_prevRent_woNaN_previousdelay['type_of_delay']=='late', :]
else:
    delay_question3 = delay_prevRent_woNaN_previousdelay.loc[delay_prevRent_woNaN_previousdelay['type_of_delay']=='late', :].loc[delay_prevRent_woNaN_previousdelay['checkin_type']==checkin_type, :]

fig = px.histogram(delay_question3, 'state', pattern_shape='checkin_type', color='threshold', text_auto=True,  
             title='Repartition des retards et impact sur les annulations')
st.plotly_chart(fig, use_container_width=True)

df_groupby = delay_question3.groupby(['state', 'threshold'], as_index=False).size()
df = df_groupby.loc[df_groupby['state']=='canceled',:]
if len(df.loc[df['threshold']=='below_threshold', 'size']) > 0:
    number_solved = df.loc[df['threshold']=='below_threshold', 'size'].values[0]
    total_number = delay_question3.groupby('state').count().loc['canceled', 'threshold']
    value_metric_question3 = round(number_solved/total_number*100, ndigits=2)
else:
    value_metric_question3 = 0

st.metric(label="Pourcentage d'annulations evitees", value=f"{value_metric_question3}%")

# Question 4: Revenue impact
st.markdown("**Impact 3: Analyse des revenus**")
fig = px.histogram(pricing, 'rental_price_per_day', color='has_getaround_connect', 
                   title="Distribution des prix actuels par type de service")
fig.add_vline(x=np.mean(pricing.loc[pricing['has_getaround_connect']==False, 'rental_price_per_day']), 
              line_dash='dash', line_color='blue')
fig.add_vline(x=np.mean(pricing.loc[pricing['has_getaround_connect']==True, 'rental_price_per_day']), 
              line_dash='dash', line_color='lightblue')
st.plotly_chart(fig, use_container_width=True)

# Revenue calculations 
current_prices = sum(pricing['rental_price_per_day'])
mean_price_mobile = np.mean(pricing.loc[pricing['has_getaround_connect']==False, 'rental_price_per_day'])
mean_price_connect = np.mean(pricing.loc[pricing['has_getaround_connect']==True, 'rental_price_per_day'])

if checkin_type == 'both':
    affected_percentage = value_metric_question2 / 100
    estimated_loss = current_prices * affected_percentage * 0.15
elif checkin_type == 'connect':
    affected_percentage = value_metric_question2 / 100
    estimated_loss = sum(pricing.loc[pricing['has_getaround_connect']==True, 'rental_price_per_day']) * affected_percentage * 0.12
else:
    affected_percentage = value_metric_question2 / 100
    estimated_loss = sum(pricing.loc[pricing['has_getaround_connect']==False, 'rental_price_per_day']) * affected_percentage * 0.18

percent_loss = (estimated_loss / current_prices) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Locations CONNECT impactees", value="156")
with col2:
    st.metric(label="Locations MOBILE impactees", value="202")

col3, col4 = st.columns(2)
with col3:
    st.metric(label="Problemes CONNECT evites", value="63")
with col4:
    st.metric(label="Problemes MOBILE evites", value="113")
    
st.markdown("---")

# Sidebar 
st.sidebar.header("GetAround Analysis Dashboard")
st.sidebar.markdown("""
    **Navigation:**
    * [Donnees sources](#integration-des-donnees)
    * [Resultats cles](#resultats-cles-de-l-analyse)  
    * [Configuration delai](#configuration-du-delai-minimum)
    * [Impacts parametres](#impact-des-parametres-selectionnes)
""")
st.sidebar.markdown("---")
st.sidebar.write("Analyse réalisée pour optimiser la stratégie de délais")