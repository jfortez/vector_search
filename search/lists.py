import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from models.list import BaseList
import pickle
from util import normalize


class DataStorage:
    """Maneja la persistencia de datos de Listas en disco de forma optimizada."""

    _cache = None  # Cache en memoria para datos crudos

    def __init__(self, file_path: Optional[str] = "lists.pkl"):
        self.file_path = Path(file_path)
        self.url = "http://172.16.11.132:4000/api/getProcessedData"
        start_time = time.time()
        self.storage: List[Dict[str, Any]] = self._load_or_fetch_data()
        print(f"[DataStorage] Inicialización total: {time.time() - start_time:.2f}s")

    def _fetch_from_api(self) -> List[Dict[str, Any]]:
        """Obtiene datos desde la API de forma rápida."""
        start_time = time.time()
        try:
            # Optimizamos la solicitud con conexión persistente y timeout
            session = requests.Session()
            response = session.get(self.url, timeout=5)
            response.raise_for_status()
            data = response.json()  # Datos crudos como lista de dicts
            print(
                f"[DataStorage] Fetch API: {time.time() - start_time:.2f}s ({len(data)} registros)"
            )
            return data
        except requests.RequestException as e:
            raise RuntimeError(
                f"[DataStorage] Error en API tras {time.time() - start_time:.2f}s: {str(e)}"
            )

    def _load_or_fetch_data(self) -> List[Dict[str, Any]]:
        """Carga datos desde archivo o API."""
        start_time = time.time()

        # Usar cache en memoria si existe
        if DataStorage._cache is not None:
            print(f"[DataStorage] Usando cache en memoria")
            return DataStorage._cache

        if self.file_path.exists():
            data = self._load_from_file()
        else:
            data = self._fetch_from_api()
            self._save_to_file(data)

        # Guardar en cache
        DataStorage._cache = data
        print(f"[DataStorage] _load_or_fetch_data: {time.time() - start_time:.2f}s")
        return data

    def _save_to_file(self, source: List[Dict[str, Any]]) -> None:
        """Guarda los datos en disco usando pickle."""
        start_time = time.time()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("wb") as f:
            pickle.dump(source, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[DataStorage] Guardado: {time.time() - start_time:.2f}s ({len(source)} registros)"
        )

    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Carga los datos desde disco usando pickle."""
        start_time = time.time()
        with self.file_path.open("rb") as f:
            data = pickle.load(f)
        print(
            f"[DataStorage] Carga desde disco: {time.time() - start_time:.2f}s ({len(data)} registros)"
        )
        return data

    def get_lists(self) -> List[BaseList]:
        """Convierte los datos crudos a BaseList bajo demanda."""
        start_time = time.time()
        result = [BaseList(**item) for item in self.storage]
        print(f"[DataStorage] Conversión a BaseList: {time.time() - start_time:.2f}s")
        return result


class EmbeddingManager:
    """Gestiona la creación, carga y búsqueda de embeddings."""

    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.index_file = self.embeddings_dir / "faiss_index.bin"
        self.embeddings_file = self.embeddings_dir / "embeddings.npy"

        print("🤖 [EmbeddingManager] Cargando modelo de embeddings...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.data_storage = DataStorage()
        self.data = self.data_storage.storage
        print(
            f"🧠 [EmbeddingManager] Inicializando embeddings para {len(self.data)} registros..."
        )

        # Inicializar índice FAISS
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> np.ndarray:
        """Inicializa embeddings cargándolos o creándolos."""
        if self.index_file.exists() and self.embeddings_file.exists():
            return self._load_embeddings_and_index()
        return self._create_and_save_embeddings()

    def _load_npy_file(self, file_path: Path) -> np.ndarray:
        """Función auxiliar para cargar un archivo .npy (usada en paralelo si necesario)."""
        return np.load(file_path)

    def _load_embeddings_and_index(self) -> np.ndarray:
        """Carga embeddings e índice FAISS desde disco."""

        print(
            f"📂 [EmbeddingManager] Cargando embeddings desde {self.embeddings_file}..."
        )
        embeddings = np.load(self.embeddings_file).astype("float32")

        print(f"📂 [EmbeddingManager] Cargando índice FAISS desde {self.index_file}...")
        self.index = faiss.read_index(str(self.index_file))

        print(
            f"✅ [EmbeddingManager] Cargados {embeddings.shape[0]} embeddings y índice"
        )

        return embeddings

    def _create_and_save_embeddings(self) -> np.ndarray:
        """Crea embeddings a partir de las listas y los guarda."""
        sentences = [
            normalize(f"{element.NombreCompleto} {element.Fuente}")
            for element in self.data
        ]
        print(
            f"✨ [EmbeddingManager] Generando embeddings para {len(sentences)} registros"
        )
        embeddings = self.model.encode(sentences, show_progress_bar=True).astype(
            "float32"
        )

        print("📈 [EmbeddingManager] Indexando embeddings en FAISS...")
        self.index.add(embeddings)

        # Guardar embeddings y índice
        print(
            f"💾 [EmbeddingManager] Guardando embeddings en {self.embeddings_file}..."
        )
        np.save(self.embeddings_file, embeddings)

        print(f"💾 [EmbeddingManager] Guardando índice en {self.index_file}...")
        faiss.write_index(self.index, str(self.index_file))

        print("✅ [EmbeddingManager] Embeddings e índice guardados exitosamente")
        return embeddings

    def search(self, query: str, threshold: float = 0.1, k: int = 10) -> pd.DataFrame:
        """Busca los k elementos más similares a la consulta."""
        print(
            f"🔍 [EmbeddingManager] Buscando '{query}' con threshold={threshold}, k={k}"
        )
        query_embedding = self.model.encode([normalize(query)]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        max_distance = max(distances[0].max(), 1e-10)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = 1 - dist / max_distance
            if similarity >= threshold:
                row = self.data[idx]
                results.append(
                    {"idx": idx, **row.model_dump(), "similarity": f"{similarity:.2%}"}
                )

        print(f"✅ [EmbeddingManager] Encontrados {len(results)} resultados relevantes")
        return pd.DataFrame(results)


def main():
    """Función principal para demostrar el uso del EmbeddingManager."""
    print("🚀 Iniciando demo de EmbeddingManager...")
    embedding_manager = EmbeddingManager()
    query = "BORIS YAKOVLEVICH LIVSHITS"
    print(f"\n🔎 Realizando búsqueda para: '{query}'")
    results = embedding_manager.search(query)
    print("\n📊 Resultados de la búsqueda:")
    print(results)


if __name__ == "__main__":
    # main()
    print("CALL MAIN")
