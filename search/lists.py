import time
from pathlib import Path
from typing import List, Optional
from enum import Enum
import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from models.list import BaseList, ListItem
import pickle
from util import normalize


# Define an Enum for model selection
class EmbeddingModelType(Enum):
    MINI_LM = "all-MiniLM-L6-v2"  # Default model for general use, 384 dimensions
    MULTI_QA = (
        "multi-qa-MiniLM-L6-cos-v1"  # Better for semantic similarity, 384 dimensions
    )
    PARAPHRASE = "paraphrase-MiniLM-L6-v2"  # Good for paraphrases and name variations, 384 dimensions


class DataStorage:
    """Maneja la persistencia de datos de Listas en disco de forma optimizada."""

    _cache = None  # Cache en memoria para datos crudos

    def __init__(self, file_path: Optional[str] = "lists.pkl"):
        self.file_path = Path(file_path)
        self.url = "http://172.16.11.132:4000/api/getProcessedData"
        self.storage: List[ListItem] = self._load_or_fetch_data()

    def _fetch_from_api(self) -> List[ListItem]:
        """Obtiene datos desde la API de forma rápida."""
        start_time = time.time()
        try:
            session = requests.Session()
            response = session.get(self.url)
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

    def _load_or_fetch_data(self) -> List[ListItem]:
        """Carga datos desde archivo o API."""
        if DataStorage._cache is not None:
            print(f"[DataStorage] Usando cache en memoria")
            return DataStorage._cache

        if self.file_path.exists():
            data = self._load_from_file()
        else:
            data = self._fetch_from_api()
            self._save_to_file(data)

        DataStorage._cache = data
        return data

    def _save_to_file(self, source: List[ListItem]) -> None:
        """Guarda los datos en disco usando pickle."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("wb") as f:
            pickle.dump(source, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_file(self) -> List[ListItem]:
        """Carga los datos desde disco usando pickle."""
        with self.file_path.open("rb") as f:
            data = pickle.load(f)
        return data

    def get_lists(self) -> List[BaseList]:
        """Convierte los datos crudos a BaseList bajo demanda."""
        result = [BaseList(**item) for item in self.storage]
        return result


class EmbeddingManager:
    """Gestiona la creación, carga y búsqueda de embeddings con soporte para múltiples modelos."""

    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        model_type: EmbeddingModelType = EmbeddingModelType.MINI_LM,
    ):
        # Sanitizar el nombre del modelo para usarlo en rutas (reemplazar "/" por "_")
        self.model_name = model_type.value.replace("/", "_")
        self.embeddings_dir = Path(embeddings_dir) / self.model_name
        self.index_file = self.embeddings_dir / "faiss_index.bin"
        self.embeddings_file = self.embeddings_dir / "embeddings.npy"

        # Inicializar el modelo seleccionado
        self.model = SentenceTransformer(model_type.value)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.data_storage = DataStorage()
        self.data = self.data_storage.storage
        print(
            f"🧠 [EmbeddingManager] Inicializando embeddings para {len(self.data)} registros con modelo {model_type.value} | dimension {self.dimension}..."
        )

        # Inicializar índice FAISS para la dimensionalidad del modelo
        self.index = faiss.IndexFlatL2(
            self.dimension
        )  # Usamos L2 para consistencia; puedes cambiar a IP si usas cosine
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> np.ndarray:
        """Inicializa embeddings cargándolos o creándolos para el modelo actual."""
        # Crear directorio si no existe
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        if self.index_file.exists() and self.embeddings_file.exists():
            return self._load_embeddings_and_index()
        return self._create_and_save_embeddings()

    def _load_npy_file(self, file_path: Path) -> np.ndarray:
        """Función auxiliar para cargar un archivo .npy."""
        return np.load(file_path)

    def _load_embeddings_and_index(self) -> np.ndarray:
        """Carga embeddings e índice FAISS desde disco para el modelo actual."""
        print(
            f"📂 [EmbeddingManager] Cargando embeddings desde {self.embeddings_file}..."
        )
        embeddings = np.load(self.embeddings_file).astype("float32")

        print(f"📂 [EmbeddingManager] Cargando índice FAISS desde {self.index_file}...")
        self.index = faiss.read_index(str(self.index_file))

        print(
            f"✅ [EmbeddingManager] Cargados {embeddings.shape[0]} embeddings y índice para el modelo {self.model_name}"
        )
        return embeddings

    def _create_and_save_embeddings(self) -> np.ndarray:
        """Crea embeddings a partir de las listas y los guarda para el modelo actual."""
        # Enhanced name normalization: handle permutations and variations

        sentences = [
            self._normalize_name(
                f"{element["NombreCompleto"]} {element["Identificacion"] if element["Identificacion"] is not None else ""}"
            )
            for element in self.data
        ]
        print(
            f"✨ [EmbeddingManager] Generando embeddings para {len(sentences)} registros con modelo {self.model_name}"
        )
        embeddings = self.model.encode(sentences, show_progress_bar=True).astype(
            "float32"
        )

        print("📈 [EmbeddingManager] Indexando embeddings en FAISS...")
        self.index.add(embeddings)

        # Guardar embeddings y índice en el directorio del modelo
        print(
            f"💾 [EmbeddingManager] Guardando embeddings en {self.embeddings_file}..."
        )
        np.save(self.embeddings_file, embeddings)

        print(f"💾 [EmbeddingManager] Guardando índice en {self.index_file}...")
        faiss.write_index(self.index, str(self.index_file))

        print(
            f"✅ [EmbeddingManager] Embeddings e índice guardados exitosamente para {self.model_name}"
        )
        return embeddings

    def _normalize_name(self, name: str) -> str:
        """Normaliza nombres para manejar variaciones como orden diferente de palabras."""
        return normalize(name)

    def search(self, query: str, threshold: float = 0.5, k: int = 10) -> pd.DataFrame:
        """Busca los k elementos más similares a la consulta con un umbral más alto."""
        # Normalize the query similarly to names
        normalized_query = self._normalize_name(query)
        query_embedding = self.model.encode([normalized_query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        # Use L2 distance to compute similarity (0 to 1 scale)
        max_distance = max(distances[0].max(), 1e-10)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Convert L2 distance to a similarity score (0 to 1)
            similarity = 1 - (dist / max_distance) if max_distance > 0 else 0
            if similarity >= threshold:
                row = self.data[idx]
                results.append({**row, "Similarity": f"{similarity:.2%}"})

        return pd.DataFrame(results)
