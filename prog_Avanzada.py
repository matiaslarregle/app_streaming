import streamlit as st
import pandas as pd
import re
import os
import numpy as np
from abc import ABC, abstractmethod  # Abstracción
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class ClasificadorBase(ABC):
    @abstractmethod
    def cargar_y_entrenar_modelo(self):
        pass

    @abstractmethod
    def predecir_comentario(self, comentario):
        pass

class ClasificadorComentarios(ClasificadorBase):
    def __init__(self, data_folder='data'):
        self._data_folder = data_folder  # Encapsulado con "_"
        self._vectorizer = TfidfVectorizer(max_features=50000)
        self._le = LabelEncoder()
        self._modelo = None

    def _limpiar_texto(self, texto):  # Método "privado"
        texto = str(texto).lower()
        texto = re.sub(r'<.*?>', ' ', texto)
        texto = re.sub(r'[^a-záéíóúüñ0-9\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def _cargar_csv(self, nombre_archivo):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(base_dir, self._data_folder, nombre_archivo)
        return pd.read_csv(ruta)

    def _expandir_comentarios(self, df):
        filas = []
        for _, row in df.iterrows():
            comentarios = eval(row['Comentarios Detalles']) if isinstance(row['Comentarios Detalles'], str) else row['comentarios']
            for comentario in comentarios:
                filas.append({'comentario': comentario, 'Canal': row['Canal']})
        return pd.DataFrame(filas)

    @st.cache_resource
    def cargar_y_entrenar_modelo(self):
        archivos = [
            'Olga_comentarios.csv',
            'Vorterix_comentarios.csv',
            'carajo_comentarios.csv',
            'AZZ_comentarios.csv',
            'Blender_comentarios.csv',
            'Gelatina_comentarios.csv',
            'Luzu_comentarios.csv'
        ]
        df_list = [self._expandir_comentarios(self._cargar_csv(a)) for a in archivos]
        df_total = pd.concat(df_list, ignore_index=True)

        df_total['comentario_limpio'] = df_total['comentario'].apply(self._limpiar_texto)

        X = self._vectorizer.fit_transform(df_total['comentario_limpio'])
        y = self._le.fit_transform(df_total['Canal'])

        X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        X_train = X_train.toarray()
        num_classes = len(self._le.classes_)

        self._modelo = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        self._modelo.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

        self._modelo.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
        return self._modelo, self._vectorizer, self._le

    def predecir_comentario(self, comentario):  # Polimorfismo
        comentario_limpio = self._limpiar_texto(comentario)
        vector = self._vectorizer.transform([comentario_limpio]).toarray()
        pred = self._modelo.predict(vector, verbose=0)
        return self._le.inverse_transform([np.argmax(pred)])[0]

class ClasificadorDummy(ClasificadorBase):  # Otro ejemplo de polimorfismo
    def cargar_y_entrenar_modelo(self):
        return None, None, None

    def predecir_comentario(self, comentario):
        return "Canal Desconocido"

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

st.markdown("<h1>Predicción de comentarios en canales de STREAMING</h1>", unsafe_allow_html=True)
st.markdown("### Tipeá un comentario y descubrí a qué canal corresponde 🤓")

comentario_usuario = st.text_area("Acá va tu comentario: 👇", height=150)

clasificador = ClasificadorComentarios()
modelo, vectorizer, le = clasificador.cargar_y_entrenar_modelo()

if st.button("🔍 Predecir Canal"):
    if comentario_usuario.strip() == "":
        st.warning("⚠️ PRIMERO INGRESÁ UN COMENTARIO ⚠️")
    else:
        canal = clasificador.predecir_comentario(comentario_usuario)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logos_canales.get(canal, ""), use_container_width=True, caption=canal)

        st.markdown(
            f"<h2 style='text-align: center; font-size: 32px;'>Este comentario fue escrito en un video de<br><b>{canal}</b></h2>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown("Creado por [matilarregle](https://x.com/elescouter)")
