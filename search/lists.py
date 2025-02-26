import time
from pathlib import Path
from typing import List, Optional
from enum import Enum
import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from scipy.spatial.distance import canberra
from models.list import BaseList, ListItem
import pickle
from util import normalize


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


# Define an Enum for model selection
class EmbeddingModelType(Enum):
    MINI_LM = "all-MiniLM-L6-v2"  # Default model for general use, 384 dimensions
    MULTI_QA = (
        "multi-qa-MiniLM-L6-cos-v1"  # Better for semantic similarity, 384 dimensions
    )
    PARAPHRASE = "paraphrase-MiniLM-L6-v2"  # Good for paraphrases and name variations, 384 dimensions


# Enum para los tipos de índices FAISS soportados
# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
class FaissIndexType(Enum):
    FLAT_L2 = "flat_l2"  # Búsqueda exacta con distancia L2, alta precisión, alto uso de memoria
    PQ = "pq"  # Cuantización de producto, búsquedas exactas con compresión
    IVFFLAT = "ivfflat"  # Índice invertido con distancias exactas en listas, rápido para grandes datos
    IVFPQ = (
        "ivfpq"  # Índice invertido con cuantización de producto, eficiente en memoria
    )
    HNSWFLAT = "hnswflat"  # Gráfico HNSW, eficiente para alta dimensión, no requiere entrenamiento


class EmbeddingGenerator:
    """Maneja la generación y carga de embeddings usando SentenceTransformer."""

    def __init__(
        self,
        model_type: EmbeddingModelType,
        data: List[dict],
        embeddings_dir: Path,
        print_times: bool = False,
    ):
        """
        Inicializa el generador de embeddings.

        Args:
            model_type (EmbeddingModelType): Modelo SentenceTransformer a usar.
            data (List[dict]): Lista de datos crudos para generar embeddings.
            embeddings_dir (Path): Directorio donde se almacenan los embeddings.
            print_times (bool): Si True, imprime tiempos de ejecución.
        """
        self.model = SentenceTransformer(model_type.value)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.data = data
        self.embeddings_file = embeddings_dir / "embeddings.npy"
        self.print_times = print_times
        self.model_name = model_type.value.replace("/", "_")

    def _normalize_name(self, name: str) -> str:
        """Normaliza nombres para manejar variaciones."""
        return normalize(name)

    def _generate_embeddings(self) -> np.ndarray:
        """Genera embeddings a partir de los datos."""
        start_time = time.time()
        sentences = [
            self._normalize_name(
                f"{element['NombreCompleto']} {element['Identificacion'] if element['Identificacion'] is not None else ''}"
            )
            for element in self.data
        ]
        print(
            f"✨ [EmbeddingGenerator] Generando embeddings para {len(sentences)} registros "
            f"con modelo {self.model_name}"
        )
        embeddings = self.model.encode(sentences, show_progress_bar=True).astype(
            "float32"
        )
        if self.print_times:
            print(
                f"⏱ [EmbeddingGenerator] Generación de embeddings: {time.time() - start_time:.2f}s"
            )
        return embeddings

    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Guarda los embeddings en disco."""
        start_time = time.time()
        print(
            f"💾 [EmbeddingGenerator] Guardando embeddings en {self.embeddings_file}..."
        )
        np.save(self.embeddings_file, embeddings)
        if self.print_times:
            print(
                f"⏱ [EmbeddingGenerator] Guardado de embeddings: {time.time() - start_time:.2f}s"
            )

    def _load_embeddings(self) -> np.ndarray:
        """Carga embeddings desde disco."""
        start_time = time.time()
        embeddings = np.load(self.embeddings_file).astype("float32")
        if self.print_times:
            print(
                f"⏱ [EmbeddingGenerator] Carga de embeddings: {time.time() - start_time:.2f}s"
            )
        return embeddings

    def get_embeddings(self) -> np.ndarray:
        """Obtiene embeddings, cargándolos si existen o generándolos si no."""
        if self.embeddings_file.exists():
            embeddings = self._load_embeddings()
            if embeddings.shape[1] != self.dimension or embeddings.shape[0] != len(
                self.data
            ):
                print(
                    "⚠️ [EmbeddingGenerator] Dimensiones no coinciden. Regenerando embeddings..."
                )
                embeddings = self._generate_embeddings()
                self._save_embeddings(embeddings)
        else:
            embeddings = self._generate_embeddings()
            self._save_embeddings(embeddings)
        return embeddings


class FaissIndexManager:
    """Gestiona la creación, carga y búsqueda en índices FAISS."""

    def __init__(
        self,
        dimension: int,
        index_type: FaissIndexType,
        embeddings_dir: Path,
        embeddings: np.ndarray,
        print_times: bool = False,
    ):
        """
        Inicializa el administrador de índices FAISS.

        Args:
            dimension (int): Dimensión de los embeddings.
            index_type (FaissIndexType): Tipo de índice FAISS a utilizar.
            embeddings_dir (Path): Directorio donde se almacenan los índices.
            embeddings (np.ndarray): Embeddings para indexar.
            print_times (bool): Si True, imprime tiempos de ejecución.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index_file = embeddings_dir / f"faiss_index_{index_type.value}.bin"
        self.embeddings = embeddings
        self.print_times = print_times
        self.index = None
        self._initialize_index()

    def _create_index(self) -> faiss.Index:
        """Crea un índice FAISS según el tipo especificado."""
        if self.index_type == FaissIndexType.FLAT_L2:
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == FaissIndexType.PQ:
            return faiss.IndexPQ(self.dimension, 16, 8)  # 16 subcuantizadores, 8 bits
        elif self.index_type == FaissIndexType.IVFFLAT:
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 10000)  # 10000 listas
        elif self.index_type == FaissIndexType.IVFPQ:
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFPQ(
                quantizer, self.dimension, 10000, 16, 8
            )  # 10000 listas, 16 subvectores, 8 bits
        elif self.index_type == FaissIndexType.HNSWFLAT:
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 vecinos por nodo
            index.hnsw.efConstruction = 200  # Construcción rápida del gráfico
            return index
        else:
            raise ValueError(f"Tipo de índice no soportado: {self.index_type}")

    def _initialize_index(self) -> None:
        """Inicializa el índice FAISS, cargándolo si existe o creándolo si no."""
        if self.index_file.exists():
            self._load_index()
        else:
            self._create_and_save_index()

    def _load_index(self) -> None:
        """Carga el índice FAISS desde disco."""
        start_time = time.time()
        print(f"📂 [FaissIndexManager] Cargando índice desde {self.index_file}...")
        self.index = faiss.read_index(str(self.index_file))
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Carga de índice: {time.time() - start_time:.2f}s"
            )

    def _create_and_save_index(self) -> None:
        """Crea el índice FAISS, lo entrena si es necesario y lo guarda."""
        start_time = time.time()
        self.index = self._create_index()
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Creación de índice: {time.time() - start_time:.2f}s"
            )

        requires_training = self.index_type in {
            FaissIndexType.PQ,
            FaissIndexType.IVFFLAT,
            FaissIndexType.IVFPQ,
        }
        if requires_training and not self.index.is_trained:
            start_time = time.time()
            self.index.train(self.embeddings)
            if self.print_times:
                print(
                    f"⏱ [FaissIndexManager] Entrenamiento de índice: {time.time() - start_time:.2f}s"
                )

        start_time = time.time()
        print(
            f"📈 [FaissIndexManager] Añadiendo {len(self.embeddings)} embeddings al índice..."
        )
        self.index.add(self.embeddings)
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Añadir embeddings: {time.time() - start_time:.2f}s"
            )

        start_time = time.time()
        print(f"💾 [FaissIndexManager] Guardando índice en {self.index_file}...")
        faiss.write_index(self.index, str(self.index_file))
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Guardado de índice: {time.time() - start_time:.2f}s"
            )

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Realiza una búsqueda en el índice FAISS."""
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]


class SearchType(Enum):
    DEFAULT = "default"  # Usa el cálculo original basado en distancia L2
    COSINE = "coseno"  # Usa similitud coseno
    CAMBERRA = "camberra"  # Usa similitud Canberra


class EmbeddingManager:
    """Integra la generación de embeddings y la gestión de índices FAISS para búsqueda semántica."""

    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        model_type: EmbeddingModelType = EmbeddingModelType.MINI_LM,
        index_type: FaissIndexType = FaissIndexType.FLAT_L2,
        print_times: bool = False,
    ):
        self.model_name = model_type.value.replace("/", "_")
        self.embeddings_dir = Path(embeddings_dir) / self.model_name
        self.print_times = print_times
        self.index_type = index_type

        # Cargar datos
        self.data_storage = DataStorage()
        self.data = self.data_storage.storage
        print(
            f"🧠 [EmbeddingManager] Inicializando para {len(self.data)} registros "
            f"con modelo {model_type.value} | índice {index_type.value}..."
        )

        # Inicializar capas
        self.embedding_generator = EmbeddingGenerator(
            model_type, self.data, self.embeddings_dir, print_times
        )
        self.dimension = self.embedding_generator.dimension
        self.embeddings = self.embedding_generator.get_embeddings()
        self.faiss_manager = FaissIndexManager(
            self.dimension,
            index_type,
            self.embeddings_dir,
            self.embeddings,
            print_times,
        )

    def _normalize_name(self, name: str) -> str:
        """Normaliza nombres para manejar variaciones."""
        return normalize(name)

    def search(
        self,
        query: str,
        threshold: float = 0.1,
        k: int = 10,
        type: SearchType = SearchType.DEFAULT,
    ) -> pd.DataFrame:
        """Busca los k elementos más similares a la consulta usando el método especificado."""
        start_time = time.time()
        normalized_query = self._normalize_name(query)
        query_embedding = self.embedding_generator.model.encode(
            [normalized_query], show_progress_bar=False
        ).astype("float32")
        # Búsqueda inicial con FAISS para obtener k candidatos
        distances, indices = self.faiss_manager.search(query_embedding, k)
        if type == SearchType.DEFAULT:
            # Cálculo original basado en distancia L2
            max_distance = max(distances.max(), 1e-10)
            results = []
            for dist, idx in zip(distances, indices):
                if idx == -1:
                    continue
                similarity = 1 - (dist / max_distance) if max_distance > 0 else 0
                if similarity >= threshold:
                    row = self.data[idx]
                    results.append({**row, "Similarity": f"{similarity:.2%}"})
        elif type == SearchType.COSINE:
            # Cálculo basado en similitud coseno
            candidate_embeddings = self.embeddings[indices]
            similarities = cos_sim(query_embedding, candidate_embeddings)[0]
            results = []
            for sim, idx in zip(similarities, indices):
                if idx == -1:
                    continue
                if sim >= threshold:
                    row = self.data[idx]
                    results.append({**row, "Similarity": f"{sim:.2%}"})
        elif type == SearchType.CAMBERRA:
            candidate_embeddings = self.embeddings[indices]
            results = []
            for idx, cand_emb in zip(indices, candidate_embeddings):
                if idx == -1:
                    continue
                dist = canberra(query_embedding[0], cand_emb)  # Distancia Canberra
                similarity = 1 / (1 + dist)  # Convertir a similitud (0-1)
                if similarity >= threshold:
                    row = self.data[idx]
                    results.append({**row, "Similarity": f"{similarity:.2%}"})

        if self.print_times:
            print(
                f"⏱ [EmbeddingManager] Búsqueda total ({query}, tipo: {type.value}): {time.time() - start_time:.2f}s"
            )
        return pd.DataFrame(results)


if __name__ == "__main__":
    em = EmbeddingManager(
        model_type=EmbeddingModelType.PARAPHRASE,
        index_type=FaissIndexType.HNSWFLAT,
    )
    inputs = ["burbano paul cirstan"]
    for input in inputs:
        print(f"Searching for: {input}")
        result1 = em.search(input, type=SearchType.COSINE)
        result2 = em.search(input, type=SearchType.DEFAULT)
        result3 = em.search(
            input,
            type=SearchType.CAMBERRA,
        )
        print("Cosine Similarity")
        print(result1)
        print("")
        print("Default")
        print(result2)
        print("")
        print("Camberra")
        print(result3)
        print("-" * 50)
