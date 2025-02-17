import faiss
import torch
import numpy as np
import os
import pickle
from transformers import AutoModel, AutoTokenizer
from search import get_identificacion  # Obtiene datos desde SQL Server

# 🔹 Configuración del modelo
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME)

INDEX_PATH = "faiss_index.bin"
NAMES_PATH = "names.pkl"

# 🔹 Eliminar índice FAISS si existe


def reset_index():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(NAMES_PATH):
        os.remove(NAMES_PATH)

# 🔹 Generar embeddings normalizados


def generate_embeddings(texts):
    tokens = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(
            **tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 🔹 Normalización L2 para similitud coseno
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 🔹 Crear índice FAISS optimizado para similitud coseno


def create_index(df):
    reset_index()  # 🔄 Elimina el índice antes de crearlo
    print("🔄 Creando nuevo índice FAISS...")

    names = df["nombre"].tolist()
    embeddings = generate_embeddings(names)

    # 🔹 FAISS IndexFlatIP (similaridad coseno con normalización previa)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Guardar índice y nombres
    faiss.write_index(index, INDEX_PATH)
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)

    return index, names

# 🔹 Búsqueda con IA usando FAISS y similitud coseno


def ai_search(query, index, names, top_k=5, threshold=0.6):
    query_embedding = generate_embeddings([query])

    # 🔹 FAISS devuelve similitud coseno directamente
    similarities, indices = index.search(query_embedding, top_k)

    results = [(names[idx], similarities[0][i])
               for i, idx in enumerate(indices[0]) if similarities[0][i] > threshold]
    return sorted(results, key=lambda x: x[1], reverse=True) if results else None


# 🔹 Prueba con múltiples consultas
if __name__ == "__main__":
    data = get_identificacion()
    print("📊 Resultados de Identificación")
    print(data)
    print("---------------------------------")

    index, names = create_index(data)

    # Lista de nombres mal escritos para testear
    search_queries = ["J. Prze", "I. Peres", "J. Perez",
                      "J Prez", "Peres", "J Qeres", "Z. Erep", "Peerez", "J Peéz"]

    for query in search_queries:
        results = ai_search(query, index, names)
        if results:
            print(f"✅ Query: '{query}' → Matches: {results}")
        else:
            print(f"❌ Query: '{query}' → No matches found.")
