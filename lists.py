import requests
import json
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss
from util import normalize
import pandas as pd


class Post(BaseModel):
    userId: int
    id: int
    title: str
    body: str

    class Config:
        """Configuración de Pydantic para la clase Post."""

        from_attributes = True  # Permite crear instancias desde atributos


class ListManager:
    """Clase para manejar listas de Posts obtenidas de una API y guardarlas/cargarlas desde un archivo."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Inicializa el ListManager.

        Args:
            file_path (str, optional): Ruta del archivo donde se guardará/cargará la lista.
                                     Si es None, no se guarda en disco.
        """
        self.file_path = Path(file_path) if file_path else None
        self.data: List[Post] = []
        self._init_list()
        if self.file_path and not self.file_path.exists():
            self.save_list()

    def _init_list(self) -> None:
        """Obtiene datos de la API o archivo y los asigna a self.data como lista de Posts."""
        if self.file_path and self.file_path.exists():
            self.load_list()
            return

        try:
            response = requests.get("https://jsonplaceholder.typicode.com/posts")
            response.raise_for_status()
            raw_data = response.json()
            # Convertir cada diccionario en un objeto Post
            self.data = [Post(**item) for item in raw_data]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error al obtener datos de la API: {e}")

    def save_list(self) -> None:
        """Guarda los datos en el archivo especificado en formato JSON."""
        if not self.file_path:
            raise ValueError("No se especificó una ruta de archivo para guardar")

        try:
            # Convertir los objetos Post a diccionarios para guardar
            data_to_save = [post.model_dump() for post in self.data]
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise Exception(f"Error al guardar el archivo: {e}")

    def load_list(self) -> None:
        """Carga los datos desde el archivo y los convierte en lista de Posts."""
        if not self.file_path:
            raise ValueError("No se especificó una ruta de archivo para cargar")

        if not self.file_path.exists():
            raise FileNotFoundError(f"El archivo {self.file_path} no existe")

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # Convertir cada diccionario cargado en un objeto Post
                self.data = [Post(**item) for item in raw_data]
        except (IOError, json.JSONDecodeError) as e:
            raise Exception(f"Error al cargar el archivo: {e}")

    def get_data(self) -> List[Post]:
        """Devuelve los datos actuales como lista de Posts."""
        return self.data


class ListEmbedding:
    def __init__(self, path: Optional[str] = "embeddings"):
        self.path = path
        data_manager = ListManager("lists.json")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

        self.data = data_manager.get_data()
        self._create_embedding()

    def _create_embedding(self):
        if self.path and Path(self.path).exists():
            self.load_embeddings(self.path)
            return

        sentences = [
            {post.id: normalize((post.title or "") + (post.body or ""))}
            for post in self.data
        ]

        print(f"[INIT:create] Creating Embeddings for {len(self.data)} records")
        self.embeddings = self.model.encode(sentences).astype("float32")

        self.index.add(self.embeddings)

        if not Path(self.path).exists():
            os.makedirs(self.path)
        self.save_embedding()

    def load_embeddings(self, npy_files_dir):
        """
        Load embeddings from .npy files in the specified directory.

        Args:
            npy_files_dir (str): Directory containing .npy embedding files
        """

        # Get all .npy files
        emb_files = [f for f in os.listdir(npy_files_dir) if f.endswith(".npy")]

        # Sort files numerically based on the number in the filename
        emb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Initialize an empty list to store the embeddings
        embeddings = []
        for file in emb_files:
            emb = np.load(os.path.join(npy_files_dir, file))
            embeddings.append(emb)

        self.embeddings = np.vstack(embeddings).astype("float32")

        print(
            f"[INIT:load] Loaded {self.embeddings.shape[0]} embeddings with dim {self.embeddings.shape[1]}"
        )

        self.index.add(self.embeddings)
        print("[INIT:load] Index added to the index.")

    def save_embedding(self):
        split = 256
        file_count = 0
        embeddings = self.embeddings
        for i in range(0, embeddings.shape[0], split):
            end = i + split
            if end > embeddings.shape[0] + 1:
                end = embeddings.shape[0] + 1
            file_count = "0" + str(file_count) if file_count < 0 else str(file_count)
            with open(f"{self.path}/embeddings_{file_count}.npy", "wb") as fp:
                np.save(fp, embeddings[i:end, :])
            print(f"embeddings_{file_count}.npy | {i} -> {end}")
            file_count = int(file_count) + 1

    def search(self, query: str, threshold: float = 0.1, k: int = 10):
        """
        Busca en el índice FAISS los k elementos más similares a la consulta 'query'.
        """
        query_embedding = self.model.encode([normalize(query)]).astype("float32")
        distances, faiss_ids = self.index.search(query_embedding, k)

        max_distance = distances[0].max() if distances[0].max() != 0 else 1

        result_data = []

        for dist, idx_id in zip(distances[0], faiss_ids[0]):
            if idx_id == -1:
                continue

            similarity = 1 - dist / max_distance

            if similarity >= threshold:
                current_row = self.data[idx_id]
                row = {
                    "id": current_row.id,
                    "title": current_row.title,
                    "body": current_row.body,
                    "similarity": f"{similarity:.2%}",
                }
                result_data.append(row)

        return pd.DataFrame(result_data)


def main():

    manager = ListEmbedding()
    print(manager.embeddings.shape)
    query = "odio"
    result = manager.search(query)
    print(result)


if __name__ == "__main__":
    main()
