import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




from rapidfuzz import process
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np
import time
from database import get_data

# Cargar el modelo de embeddings (puedes cambiarlo por otro de Hugging Face)
modelo = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(modelo)
modelo_embedding = AutoModel.from_pretrained(modelo)

# Función para obtener embeddings de un texto
def get_embedding(texto):
    tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    # with torch.no_grad():
    #     # Forzar el uso de CPU
    #     modelo_embedding.to('cpu')
    #     tokens = {k: v.to('cpu') for k, v in tokens.items()}
    #     embedding = modelo_embedding(**tokens).last_hidden_state.mean(dim=1).numpy()
    # return embedding
    with torch.no_grad():
        embedding = modelo_embedding(**tokens).last_hidden_state.mean(dim=1).numpy()
    return embedding

# Crear y almacenar los embeddings en FAISS
def create_index(df):
    names = df["nombre"].tolist()
    embeddings = np.vstack([get_embedding(name) for name in names])
    
    indice = faiss.IndexFlatL2(embeddings.shape[1])
    indice.add(embeddings)
    return indice, names

# Buscar el nombre más similar con IA
def search_with_ai(nombre_buscar, indice, nombres):
    emb_query = get_embedding(nombre_buscar)
    D, I = indice.search(emb_query, 1)  # 1 resultado más cercano
    
    if D[0][0] < 10:  # Umbral de distancia
        return nombres[I[0][0]]
    return None

def search_name(nombre_buscar, df):
    names = df["nombre"].tolist()
    matches = process.extract(nombre_buscar, names, limit=None, score_cutoff=60)  # Cambiado a extract para múltiples coincidencias
    
    if matches:
        return matches  # Retorna una lista de coincidencias (nombre, puntuación, índice)
    return None



def print_fuzzy_search(data, input):
    start_time = time.time()
    resultado = search_name(input, data)
    search_time = time.time() - start_time
    
    if resultado:     
        # Mostrar todas las coincidencias y el número de coincidencias
        nombres_encontrados = [match[0] for match in resultado]
        num_coincidencias = len(resultado)
        print(f"✅ Número de coincidencias: {num_coincidencias} | ⏱️ {search_time:.4f}s")
    else:
        print(f"❌ No se encontró coincidencia | ⏱️ {search_time:.4f}s")

def print_ai_search(data, input):
    start_time = time.time()
    indice, nombres = create_index(data)
    result = search_with_ai(input, indice, nombres)
    search_time = time.time() - start_time
    
    if result:
        print(f"✅ Nombre encontrado con IA: {result} | ⏱️ {search_time:.4f}s")
    else:
        print(f"❌ No se encontró coincidencia | ⏱️ {search_time:.4f}s")
    
    

if __name__ == "__main__":
    data = get_data()
    inputs = ["J Peres", "J. Peres", "J Perez", "I. Prez"]
    print("RESULT FROM TABLE Identificacion:")
    print(data)
    print("---------------------------")
    
    # Measure fuzzy search time
    print("FUZZY SEARCH:")
    fuzzy_start_time = time.time()
    for index, input in enumerate(inputs,start=1):
        print(f"Input #{index}: {input}")
        print_fuzzy_search(data,input)
    total_fuzzy_time = time.time() - fuzzy_start_time
    
    print("---------------------------")
    
    # Measure AI search time
    print("AI SEARCH:")
    ai_start_time = time.time()
    for index, input in enumerate(inputs,start=1):
        print(f"Input #{index}: {input}")
        print_ai_search(data,input)
    total_ai_time = time.time() - ai_start_time
    
    print("\n---------------------------")
    print(f"Total fuzzy search time: {total_fuzzy_time:.4f} seconds")
    print(f"Total AI search time: {total_ai_time:.4f} seconds")
    print(f"Time difference: {abs(total_fuzzy_time - total_ai_time):.4f} seconds")




