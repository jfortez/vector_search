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
import re


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
        """Guarda los datos en disco usando pickle, creando la carpeta si no existe."""
        self.file_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Crea la carpeta si no existe

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


class EmbeddingModelType(Enum):
    MINI_LM = "all-MiniLM-L6-v2"  # 384 dimensiones, ligero
    MULTI_QA = "multi-qa-MiniLM-L6-cos-v1"  # 384 dimensiones, bueno para similitud
    PARAPHRASE = "paraphrase-MiniLM-L6-v2"  # 384 dimensiones, bueno para variaciones
    MP_NET = "all-mpnet-base-v2"  # 768 dimensiones, robusto para errores ortográficos
    MULTI_QA_MPNET = "multi-qa-mpnet-base-dot-v1"  # 768 dimensiones, optimizado para QA


class FaissIndexType(Enum):
    FLAT_L2 = "flat_l2"  # Búsqueda exacta con L2
    PQ = "pq"  # Cuantización de producto
    IVFFLAT = (
        "ivfflat"  # Índice invertido con Inner Product, rápido para datasets grandes
    )
    IVFPQ = "ivfpq"  # Índice invertido con cuantización
    HNSWFLAT = "hnswflat"  # Gráfico HNSW, eficiente para alta dimensión
    FLAT_IP = "flat_ip"  # Búsqueda exacta con Inner Product


class EmbeddingGenerator:
    """Maneja la generación y carga de embeddings usando SentenceTransformer."""

    def __init__(
        self,
        model_type: EmbeddingModelType,
        data: List[dict],
        embeddings_dir: Path,
        print_times: bool = False,
        field: str = "NombreCompleto",
    ):
        self.model = SentenceTransformer(model_type.value)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.data = data
        self.embeddings_file = embeddings_dir / f"{field}_embeddings.npy"
        self.print_times = print_times
        self.model_name = model_type.value.replace("/", "_")
        self.field = field

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para manejar variaciones."""
        return normalize(text)

    def _generate_embeddings(self) -> np.ndarray:
        """Genera embeddings a partir de los datos."""
        start_time = time.time()
        sentences = [
            (
                self._normalize_text(item[self.field])
                if item[self.field] is not None
                else ""
            )
            for item in self.data
        ]
        print(
            f"✨ [EmbeddingGenerator] Generando embeddings para {len(sentences)} registros "
            f"con modelo {self.model_name} (campo: {self.field})"
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
        """Guarda los embeddings en disco, creando la carpeta si no existe."""
        start_time = time.time()
        self.embeddings_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # Crea la carpeta si no existe
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
                    f"⚠️ [EmbeddingGenerator] Dimensiones no coinciden para {self.field}. Regenerando embeddings..."
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
        field: str = "NombreCompleto",
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index_file = embeddings_dir / f"faiss_index_{field}_{index_type.value}.bin"
        self.embeddings = embeddings
        self.print_times = print_times
        self.field = field
        self.index = None
        self._initialize_index()

    def _create_index(self) -> faiss.Index:
        """Crea un índice FAISS según el tipo especificado."""
        if self.index_type == FaissIndexType.FLAT_L2:
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == FaissIndexType.PQ:
            return faiss.IndexPQ(self.dimension, 16, 8)
        elif self.index_type == FaissIndexType.IVFFLAT:
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(
                quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT
            )
        elif self.index_type == FaissIndexType.IVFPQ:
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFPQ(quantizer, self.dimension, 100, 16, 8)
        elif self.index_type == FaissIndexType.HNSWFLAT:
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            return index
        elif self.index_type == FaissIndexType.FLAT_IP:
            return faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Tipo de índice no soportado: {self.index_type}")

    def _initialize_index(self) -> None:
        """Inicializa el índice FAISS, cargándolo si existe o creándolo si no."""
        if self.index_file.exists():
            self._load_index()
        else:
            self._create_and_save_index()

    def _load_index(self) -> None:
        start_time = time.time()
        print(f"📂 [FaissIndexManager] Cargando índice desde {self.index_file}...")
        self.index = faiss.read_index(str(self.index_file))
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Carga de índice: {time.time() - start_time:.2f}s"
            )

    def _create_and_save_index(self) -> None:
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
            f"📈 [FaissIndexManager] Añadiendo {len(self.embeddings)} embeddings al índice ({self.field})..."
        )
        self.index.add(self.embeddings)
        if self.print_times:
            print(
                f"⏱ [FaissIndexManager] Añadir embeddings: {time.time() - start_time:.2f}s"
            )

        start_time = time.time()
        self.index_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # Crea la carpeta si no existe
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
    DEFAULT = "default"
    COSINE = "coseno"
    CANBERRA = "canberra"


class EmbeddingManager:
    """Integra la generación de embeddings y la gestión de índices FAISS para búsqueda semántica."""

    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        model_type: EmbeddingModelType = EmbeddingModelType.MULTI_QA_MPNET,
        index_type: FaissIndexType = FaissIndexType.FLAT_IP,
        print_times: bool = False,
        limit_data: Optional[int] = None,
    ):
        self.model_name = model_type.value.replace("/", "_")
        self.embeddings_dir = Path(embeddings_dir) / self.model_name
        self.print_times = print_times
        self.index_type = index_type

        # Cargar datos
        self.data_storage = DataStorage()
        self.data = self.data_storage.storage
        if limit_data:
            self.data = self.data[:limit_data]

        print(
            f"🧠 [EmbeddingManager] Inicializando para {len(self.data)} registros "
            f"con modelo {model_type.value} | índice {index_type.value}..."
        )

        # Inicializar generadores de embeddings
        self.name_generator = EmbeddingGenerator(
            model_type,
            self.data,
            self.embeddings_dir,
            print_times,
            field="NombreCompleto",
        )
        self.id_generator = EmbeddingGenerator(
            model_type,
            self.data,
            self.embeddings_dir,
            print_times,
            field="Identificacion",
        )
        self.dimension = self.name_generator.dimension

        # Obtener embeddings
        self.name_embeddings = self.name_generator.get_embeddings()
        self.id_embeddings = self.id_generator.get_embeddings()

        # Normalizar embeddings
        faiss.normalize_L2(self.name_embeddings)
        faiss.normalize_L2(self.id_embeddings)

        # Inicializar gestores de índices FAISS
        self.name_faiss_manager = FaissIndexManager(
            self.dimension,
            index_type,
            self.embeddings_dir,
            self.name_embeddings,
            print_times,
            field="NombreCompleto",
        )
        self.id_faiss_manager = FaissIndexManager(
            self.dimension,
            index_type,
            self.embeddings_dir,
            self.id_embeddings,
            print_times,
            field="Identificacion",
        )

    def _normalize_name(self, name: str) -> str:
        """Normaliza nombres para manejar variaciones."""
        return normalize(name)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Genera y normaliza el embedding para una consulta."""
        embedding = self.name_generator.model.encode(
            [query], show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def _search_by_id(self, query: str, k: int, threshold: float) -> List[dict]:
        """Busca coincidencias por ID, exactas y parciales."""
        query_embedding = self._get_query_embedding(query)
        distances, indices = self.id_faiss_manager.search(query_embedding, k * 2)

        # Búsqueda parcial por ID
        partial_id = query[-4:] if len(query) >= 4 else query
        partial_embedding = self._get_query_embedding(partial_id)
        partial_distances, partial_indices = self.id_faiss_manager.search(
            partial_embedding, k * 2
        )

        # Combinar resultados
        results = []
        seen_indices = set()
        for score, idx in sorted(
            zip(distances, indices), key=lambda x: x[0], reverse=True
        ):
            if idx == -1 or idx in seen_indices:
                continue
            row = self.data[idx]
            adjusted_score = 1.0 if row["Identificacion"] == query else score
            if adjusted_score >= threshold:
                results.append(
                    {**row, "Similarity": f"{adjusted_score:.2%}", "type": "exact"}
                )
                seen_indices.add(idx)
                if len(results) >= k:
                    break

        if len(results) < k:
            for score, idx in sorted(
                zip(partial_distances, partial_indices),
                key=lambda x: x[0],
                reverse=True,
            ):
                if idx == -1 or idx in seen_indices:
                    continue
                row = self.data[idx]
                if (
                    row["Identificacion"] is not None
                    and partial_id in row["Identificacion"]
                ):
                    adjusted_score = min(1.0, score + 0.2)
                else:
                    adjusted_score = score
                if adjusted_score >= threshold - 0.1:
                    results.append(
                        {
                            **row,
                            "Similarity": f"{adjusted_score:.2%}",
                            "type": "partial",
                        }
                    )
                    seen_indices.add(idx)
                    if len(results) >= k:
                        break

        return results[:k]

    def _search_combined(
        self, name_part: str, id_part: str, k: int, threshold: float
    ) -> List[dict]:
        """Busca coincidencias combinadas de nombre e ID."""
        name_embedding = self._get_query_embedding(name_part)
        name_distances, name_indices = self.name_faiss_manager.search(
            name_embedding, k * 2
        )

        id_embedding = self._get_query_embedding(id_part)
        id_distances, id_indices = self.id_faiss_manager.search(id_embedding, k * 2)

        name_results = {
            idx: score for score, idx in zip(name_distances, name_indices) if idx != -1
        }
        id_results = {
            idx: score for score, idx in zip(id_distances, id_indices) if idx != -1
        }

        results = []
        seen_indices = set()
        for idx in set(name_results.keys()) & set(id_results.keys()):
            combined_score = (name_results[idx] + id_results[idx]) / 2
            row = self.data[idx]
            name_match = name_part in normalize(row["NombreCompleto"])
            id_match = (
                row["Identificacion"] is not None and id_part in row["Identificacion"]
            )
            if name_match and id_match:
                combined_score = min(1.0, combined_score + 0.3)
            if combined_score >= threshold:
                results.append(
                    {**row, "Similarity": f"{combined_score:.2%}", "type": "combined"}
                )
                seen_indices.add(idx)

        if len(results) < k:
            for idx, score in sorted(
                name_results.items(), key=lambda x: x[1], reverse=True
            ):
                if idx not in seen_indices and score >= threshold:
                    row = self.data[idx]
                    results.append(
                        {**row, "Similarity": f"{score:.2%}", "type": "name"}
                    )
                    if len(results) >= k:
                        break

        return results[:k]

    def _search_by_name(
        self, query: str, k: int, threshold: float, search_type: SearchType
    ) -> List[dict]:
        """Busca coincidencias por nombre según el tipo de búsqueda."""
        query_embedding = self._get_query_embedding(query)
        distances, indices = self.name_faiss_manager.search(query_embedding, k)

        if search_type == SearchType.DEFAULT:
            max_distance = max(distances.max(), 1e-10) if distances.size > 0 else 1e-10
            results = [
                {
                    **self.data[idx],
                    "Similarity": (
                        f"{(1 - (dist / max_distance) if max_distance > 0 else 0):.2%}"
                        if query != normalize(self.data[idx]["NombreCompleto"])
                        else "100.00%"
                    ),
                }
                for dist, idx in zip(distances, indices)
                if idx != -1
                and (1 - (dist / max_distance) if max_distance > 0 else 0) >= threshold
            ]
        elif search_type == SearchType.COSINE:
            similarities = cos_sim(query_embedding, self.name_embeddings[indices])[0]
            results = [
                {
                    **self.data[idx],
                    "Similarity": (
                        "100.00%"
                        if query == normalize(self.data[idx]["NombreCompleto"])
                        else f"{sim.item():.2%}"
                    ),
                }
                for sim, idx in zip(similarities, indices)
                if idx != -1 and sim >= threshold
            ]
        elif search_type == SearchType.CANBERRA:
            candidate_embeddings = self.name_embeddings[indices]
            results = [
                {
                    **self.data[idx],
                    "Similarity": (
                        "100.00%"
                        if query == normalize(self.data[idx]["NombreCompleto"])
                        else f"{(1 / (1 + canberra(query_embedding[0], cand_emb))):.2%}"
                    ),
                }
                for idx, cand_emb in zip(indices, candidate_embeddings)
                if idx != -1
                and (1 / (1 + canberra(query_embedding[0], cand_emb))) >= threshold
            ]

        return results[:k]

    def search(
        self,
        query: str,
        threshold: float = 0.5,
        k: int = 5,
        type: SearchType = SearchType.COSINE,
    ) -> pd.DataFrame:
        """Busca los k elementos más similares a la consulta usando el método especificado."""
        start_time = time.time()
        normalized_query = self._normalize_name(query)

        # Determinar si la consulta es ID, nombre o combinación
        id_part = "".join(re.findall(r"\d+", query))
        name_part = re.sub(r"\d+", "", query).strip()

        if query.isdigit():
            results = self._search_by_id(query, k, threshold)
        elif id_part:
            results = self._search_combined(name_part, id_part, k, threshold)
        else:
            results = self._search_by_name(normalized_query, k, threshold, type)

        if self.print_times:
            print(
                f"⏱ [EmbeddingManager] Búsqueda total ({query}, tipo: {type.value}): {time.time() - start_time:.2f}s"
            )
        return pd.DataFrame(results)


# Pruebas con las queries del usuario
if __name__ == "__main__":
    em = EmbeddingManager(
        model_type=EmbeddingModelType.MULTI_QA_MPNET,  # Usando multi-qa-mpnet-base-dot-v1
        index_type=FaissIndexType.FLAT_IP,
        print_times=True,
    )
    inputs = [
        "0201688686",
        "carlos torres",
        "pedro",
        "8686",
        "juan crlos",
        "juan carlos",
        "torres 36300",
        "diego fernando andrade torre",
    ]
    for input in inputs:
        print(f"Searching for: {input}")
        result_cos = em.search(input, type=SearchType.COSINE, k=5, threshold=0.1)
        print("Cosine Similarity:")
        print(result_cos)
        print("-" * 50)
