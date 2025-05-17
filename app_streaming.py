
import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

@st.cache_resource
def cargar_y_entrenar_modelo():
    # Cargar CSVs
    df_a = pd.read_csv('data/Olga_comentarios.csv')
    df_b = pd.read_csv('data/Vorterix_comentarios.csv')
    df_c = pd.read_csv('data/carajo_comentarios.csv')
    df_d = pd.read_csv('data/AZZ_comentarios.csv')
    df_e = pd.read_csv('data/Blender_comentarios.csv')
    df_f = pd.read_csv('data/Gelatina_comentarios.csv')
    df_g = pd.read_csv('data/Luzu_comentarios.csv')

    def expandir_comentarios(df):
        filas = []
        for _, row in df.iterrows():
            comentarios = eval(row['Comentarios Detalles']) if isinstance(row['Comentarios Detalles'], str) else row['comentarios']
            for comentario in comentarios:
                filas.append({'comentario': comentario, 'Canal': row['Canal']})
        return pd.DataFrame(filas)

    df_total = pd.concat([
        expandir_comentarios(df_a),
        expandir_comentarios(df_b),
        expandir_comentarios(df_c),
        expandir_comentarios(df_d),
        expandir_comentarios(df_e),
        expandir_comentarios(df_f),
        expandir_comentarios(df_g)
    ], ignore_index=True)

    df_total['comentario_limpio'] = df_total['comentario'].apply(limpiar_texto)

    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(df_total['comentario_limpio'])

    le = LabelEncoder()
    y = le.fit_transform(df_total['Canal'])

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    return modelo, vectorizer, le

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'<.*?>', ' ', texto)
    texto = re.sub(r'[^a-záéíóúüñ0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def predecir_comentario(comentario, modelo, vectorizer, le):
    comentario_limpio = limpiar_texto(comentario)
    vector = vectorizer.transform([comentario_limpio])
    pred = modelo.predict(vector)
    return le.inverse_transform(pred)[0]

# Inicializar estado si no existe
if 'mostrar_resultado' not in st.session_state:
    st.session_state.mostrar_resultado = False
    st.session_state.prediccion = ""

# Diccionario de imágenes
logos_canales = {
    "olgaenvivo_": "logos/olga.jpg",
    "VorterixOficial": "logos/vorterix.jpg",
    "CarajoStream": "logos/carajo.png",
    "somosazz": "logos/azz.png",
    "estoesblender": "logos/blender.jpeg",
    "SomosGelatina": "logos/gelatina.jpg",
    "luzutv": "logos/luzutv.png"
}

# Cargar modelo
modelo, vectorizer, le = cargar_y_entrenar_modelo()

# ---- INTERFAZ ----
if not st.session_state.mostrar_resultado:
    st.markdown("<h1>Predicción de comentarios en canales de STREAMING</h1>", unsafe_allow_html=True)
    st.markdown("### Tipeá un comentario y descubrí a qué canal corresponde 🤓")

    comentario_usuario = st.text_area("Acá va tu comentario: 👇", height=150)

    if st.button("🔍 Predecir Canal"):
        if comentario_usuario.strip() == "":
            st.warning("⚠️ PRIMERO INGRESÁ UN COMENTARIO ⚠️")
        else:
            canal = predecir_comentario(comentario_usuario, modelo, vectorizer, le)
            st.session_state.prediccion = canal
            st.session_state.mostrar_resultado = True
            st.rerun()


else:
    canal = st.session_state.prediccion

    # Imagen centrada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logos_canales.get(canal, ""), use_container_width=True, caption=canal)

    # Texto centrado y grande
    st.markdown(
        f"<h2 style='text-align: center; font-size: 32px;'>Este comentario fue escrito en un video de<br><b>{canal}</b></h2>",
        unsafe_allow_html=True
    )

    # Botón volver
    if st.button("🔙 Volver"):
        st.session_state.mostrar_resultado = False
        st.session_state.prediccion = ""
        st.rerun()


st.markdown("---")
st.markdown("Creado por [matilarregle](https://x.com/elescouter)")
