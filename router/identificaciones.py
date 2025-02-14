from fastapi import APIRouter
from database import get_data

from sentence_transformers import SentenceTransformer

import re

import unicodedata

import faiss
import numpy as np
import pandas as pd




data = get_data()

# Regex para mantener solo letras y números, excluyendo apóstrofes
_alnum_regex = re.compile(r"(?ui)\W")

def preprocess(s):
    # 1. Normalizar caracteres acentuados a su versión base (ej. á → a, ñ → n)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    
    # 2. Eliminar caracteres no alfanuméricos (incluye los apóstrofes)
    string_out = _alnum_regex.sub(" ", s)
    
    # 3. Retornar en minúsculas y sin espacios extras
    return string_out.strip().lower()

# Pruebas

texts =  data.apply(lambda row: preprocess(f"{row['nombre']}{row['identificacion']}"), axis=1).tolist()
ids = data['id'].tolist()


# Modelo de embeddings local
model = SentenceTransformer('dccuchile/bert-base-spanish-wwm-cased')

embeddings = model.encode(texts)

# Crear índice y añadir datos
dimension = 768
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype='float32'))

def searchFromIndex(query: str, threshold: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generar embedding para la consulta del usuario
    query_embedding = model.encode([preprocess(query)])

    # Buscar en FAISS
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=10)
    # Mostrar resultados
    similarity_tuple = []  # Lista para almacenar los resultados
    results = []
    for i, idx in enumerate(indices[0]):
        # Obtener ID único
        result_id = ids[idx]  
        # Texto original
        loc = data.iloc[idx]

        result_text = loc['nombre'] + " " + loc['identificacion']
        similarity = 1 - distances[0][i] / max(distances[0])  
        # Filtrar resultados por umbral de similitud
        if similarity >= threshold:
            similarity_tuple.append({"ID": result_id, "Registro": result_text, "Similaridad": f"{similarity:.2%}"})
            results.append({"id": loc['id'], "identificacion": loc['identificacion'], "nombre": loc['nombre']})


    return pd.DataFrame(results), pd.DataFrame(similarity_tuple)




router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])


@router.get("/")
def read_identificaciones(search: str = None):
    if search:
        result = searchFromIndex(search)
        return result[0].to_dict(orient="records")
    else:
        return data.to_dict(orient="records")
    
@router.get("/{id}")
def read_identificacion(id: int):
    data = data[data['id'] == id]
    return data.to_dict(orient="records")

