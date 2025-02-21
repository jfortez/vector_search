import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from util import normalize


class Post(BaseModel):
    """Modelo de datos para un post."""

    userId: int
    id: int
    title: str
    body: str

    class Config:
        from_attributes = True


class DataStorage:
    """Maneja la persistencia de datos de posts en disco."""

    def __init__(self, file_path: Optional[str] = "lists.json"):
        self.file_path = Path(file_path)
        self.posts: List[Post] = self._load_or_fetch_data()

    def _fetch_from_api(self) -> List[Post]:
        """Obtiene datos desde la API si no existen en disco."""
        try:
            response = requests.get("https://jsonplaceholder.typicode.com/posts")
            response.raise_for_status()
            return [Post(**item) for item in response.json()]
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching API data: {e}")

    def _load_or_fetch_data(self) -> List[Post]:
        """Carga datos desde archivo o los obtiene de la API."""
        if self.file_path.exists():
            return self._load_from_file()
        posts = self._fetch_from_api()
        self._save_to_file(posts)
        return posts

    def _save_to_file(self, posts: List[Post]) -> None:
        """Guarda los posts en un archivo JSON."""
        try:
            data = [post.model_dump() for post in posts]
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise RuntimeError(f"Error saving file: {e}")

    def _load_from_file(self) -> List[Post]:
        """Carga posts desde un archivo JSON."""
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                return [Post(**item) for item in json.load(f)]
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Error loading file: {e}")


class EmbeddingManager:
    """Gestiona la creación, carga y búsqueda de embeddings."""

    def __init__(self, embeddings_dir: Optional[str] = "embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

        self.data_storage = DataStorage()
        self.posts = self.data_storage.posts
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> np.ndarray:
        """Inicializa embeddings cargándolos o creándolos."""
        if self.embeddings_dir.exists() and any(self.embeddings_dir.glob("*.npy")):
            return self._load_embeddings()
        return self._create_and_save_embeddings()

    def _create_and_save_embeddings(self) -> np.ndarray:
        """Crea embeddings a partir de los posts y los guarda."""
        sentences = [normalize(f"{post.title}{post.body}") for post in self.posts]
        print(f"[INIT] Creating embeddings for {len(self.posts)} records")

        embeddings = self.model.encode(sentences, show_progress_bar=True).astype(
            "float32"
        )
        self.index.add(embeddings)
        self._save_embeddings(embeddings)
        return embeddings

    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Guarda los embeddings en archivos .npy por lotes."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        batch_size = 256

        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            file_idx = f"{i // batch_size:03d}"
            file_path = self.embeddings_dir / f"embeddings_{file_idx}.npy"
            np.save(file_path, batch)
            print(f"Saved {file_path.name} | {i} -> {i + len(batch)}")

    def _load_embeddings(self) -> np.ndarray:
        """Carga embeddings desde archivos .npy en orden numérico correcto."""
        npy_files = list(self.embeddings_dir.glob("embeddings_*.npy"))
        if not npy_files:
            raise FileNotFoundError("No embedding files found")

        # Extraer el número del nombre del archivo y ordenar numéricamente
        npy_files.sort(key=lambda x: int(x.stem.split("_")[1]))

        embeddings = np.vstack([np.load(file) for file in npy_files]).astype("float32")
        print(
            f"[INIT] Loaded {embeddings.shape[0]} embeddings with dim {embeddings.shape[1]}"
        )
        self.index.add(embeddings)
        return embeddings

    def search(self, query: str, threshold: float = 0.1, k: int = 10) -> pd.DataFrame:
        """Busca los k posts más similares a la consulta."""
        query_embedding = self.model.encode([normalize(query)]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        max_distance = max(distances[0].max(), 1e-10)  # Evitar división por cero
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            similarity = 1 - dist / max_distance
            if similarity >= threshold:
                post = self.posts[idx]
                results.append(
                    {
                        "id": post.id,
                        "title": post.title,
                        "body": post.body,
                        "similarity": f"{similarity:.2%}",
                    }
                )

        return pd.DataFrame(results)


def main():
    """Función principal para demostrar el uso del EmbeddingManager."""
    embedding_manager = EmbeddingManager()

    query = "odio"
    results = embedding_manager.search(query)
    print("\nSearch results:")
    print(results)


if __name__ == "__main__":
    main()
