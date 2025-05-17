import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# ------------------------ FUNCIONES AUXILIARES ------------------------

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'<.*?>', ' ', texto)
    texto = re.sub(r'[^a-záéíóúüñ0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def cargar_csv(nombre_archivo):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(base_dir, 'data', nombre_archivo)
    return pd.read_csv(ruta)

def expandir_comentarios(df):
    filas = []
    for _, row in df.iterrows():
        comentarios = eval(row['Comentarios Detalles']) if isinstance(row['Comentarios Detalles'], str) else row['comentarios']
        for comentario in comentarios:
            filas.append({'comentario': comentario, 'Canal': row['Canal']})
    return pd.DataFrame(filas)

@st.cache_resource
def cargar_y_entrenar_modelo():
    archivos = [
        'Olga_comentarios.csv',
        'Vorterix_comentarios.csv',
        'carajo_comentarios.csv',
        'AZZ_comentarios.csv',
        'Blender_comentarios.csv',
        'Gelatina_comentarios.csv',
        'Luzu_comentarios.csv'
    ]
    df_list = [expandir_comentarios(cargar_csv(a)) for a in archivos]
    df_total = pd.concat(df_list, ignore_index=True)

    df_total['comentario_limpio'] = df_total['comentario'].apply(limpiar_texto)

    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(df_total['comentario_limpio'])

    le = LabelEncoder()
    y = le.fit_transform(df_total['Canal'])

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    return modelo, vectorizer, le

def predecir_comentario(comentario, modelo, vectorizer, le):
    comentario_limpio = limpiar_texto(comentario)
    vector = vectorizer.transform([comentario_limpio])
    pred = modelo.predict(vector)
    return le.inverse_transform(pred)[0]

def obtener_ruta_logo(nombre_archivo):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'logos', nombre_archivo)

logos_canales = {
    "olgaenvivo_": obtener_ruta_logo("olga.jpg"),
    "VorterixOficial": obtener_ruta_logo("vorterix.jpg"),
    "CarajoStream": obtener_ruta_logo("carajo.png"),
    "somosazz": obtener_ruta_logo("azz.png"),
    "estoesblender": obtener_ruta_logo("blender.jpeg"),
    "SomosGelatina": obtener_ruta_logo("gelatina.jpg"),
    "luzutv": obtener_ruta_logo("luzutv.png")
}

# ------------------------ INTERFAZ STREAMLIT ------------------------

st.markdown("<h1>Predicción de comentarios en canales de STREAMING</h1>", unsafe_allow_html=True)
st.markdown("### Tipeá un comentario y descubrí a qué canal corresponde 🤓")

comentario_usuario = st.text_area("Acá va tu comentario: 👇", height=150)

modelo, vectorizer, le = cargar_y_entrenar_modelo()

if st.button("🔍 Predecir Canal"):
    if comentario_usuario.strip() == "":
        st.warning("⚠️ PRIMERO INGRESÁ UN COMENTARIO ⚠️")
    else:
        canal = predecir_comentario(comentario_usuario, modelo, vectorizer, le)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logos_canales.get(canal, ""), use_container_width=True, caption=canal)

        st.markdown(
            f"<h2 style='text-align: center; font-size: 32px;'>Este comentario fue escrito en un video de<br><b>{canal}</b></h2>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown("Creado por [matilarregle](https://x.com/elescouter)")
