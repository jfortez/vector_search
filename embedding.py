from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from util import normalize

from database import get_data

model = SentenceTransformer('dccuchile/bert-base-spanish-wwm-cased')

data = get_data()



texts =  data.apply(lambda row: normalize(f"{row['nombre']}{row['identificacion']}"), axis=1).tolist()
ids = data['id'].tolist()


# Modelo de embeddings local

print("creating embedding")
embeddings = model.encode(texts)

# Crear índice y añadir datos
dimension = 768
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype='float32'))

def searchFromIndex(query: str, threshold: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generar embedding para la consulta del usuario
    query_embedding = model.encode([normalize(query)])

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

