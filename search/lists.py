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


data = [
    {
        "CredencialId": 11429,
        "Identificacion": "0100036300",
        "NombreCompleto": "CARLOS ENRIQUE TORRES MARTINEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11430,
        "Identificacion": "0100236793",
        "NombreCompleto": "MARIA CRISTINA ARIAS PAREDES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11431,
        "Identificacion": "0100275502",
        "NombreCompleto": "JOSE FERNANDO GARCIA VELEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11432,
        "Identificacion": "0100290261",
        "NombreCompleto": "MANUEL JESUS MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11433,
        "Identificacion": "0100550607",
        "NombreCompleto": "ROSA ELVIRA GUAMAN DURAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11434,
        "Identificacion": "0100767177",
        "NombreCompleto": "CARLOS ALFONSO LOPEZ IDROVO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11435,
        "Identificacion": "0100961457",
        "NombreCompleto": "MANUEL VICTOR MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11436,
        "Identificacion": "0101056091",
        "NombreCompleto": "LUIS ALBERTO MARTINEZ LASSO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11437,
        "Identificacion": "0101083244",
        "NombreCompleto": "LUIS ALBERTO ZAMBRANO SARMIENTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11438,
        "Identificacion": "0101139921",
        "NombreCompleto": "MIGUEL ANGEL VELEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11439,
        "Identificacion": "0101173078",
        "NombreCompleto": "ROSA NORMA LOPEZ LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11440,
        "Identificacion": "0101179273",
        "NombreCompleto": "MANUEL CABRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11441,
        "Identificacion": "0101196038",
        "NombreCompleto": "LUIS JAIME GUAMAN NAULA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11442,
        "Identificacion": "0101208122",
        "NombreCompleto": "CARLOS HUMBERTO LOPEZ SARMIENTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11443,
        "Identificacion": "0101218410",
        "NombreCompleto": "JUAN CARLOS GONZALEZ VINTIMILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11444,
        "Identificacion": "0101222503",
        "NombreCompleto": "JORGE EDISON GARCIA PARRA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11445,
        "Identificacion": "0101285674",
        "NombreCompleto": "LUIS ALBERTO LEMA ORTIZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11446,
        "Identificacion": "0101316008",
        "NombreCompleto": "LUIS ALBERTO PALACIOS RIVERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11447,
        "Identificacion": "0101386043",
        "NombreCompleto": "LUIS ARIOLFO SANCHEZ FLORES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11448,
        "Identificacion": "0101409332",
        "NombreCompleto": "MARIA ROSARIO VIVAR CAMPOVERDE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11449,
        "Identificacion": "0101685394",
        "NombreCompleto": "JOSE HERNANDO VELEZ AUCAPIÑA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11450,
        "Identificacion": "0101715597",
        "NombreCompleto": "JORGE EUGENIO LOPEZ JARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11451,
        "Identificacion": "0101891109",
        "NombreCompleto": "LILIA DEL CARMEN CABRERA ROJAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11452,
        "Identificacion": "0101941631",
        "NombreCompleto": "LUIS PATRICIO AREVALO BARRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11453,
        "Identificacion": "0101971018",
        "NombreCompleto": "JUAN CARLOS LOPEZ LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11454,
        "Identificacion": "0102030749",
        "NombreCompleto": "ROSA MARIA MOSQUERA MOSQUERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11455,
        "Identificacion": "0102041076",
        "NombreCompleto": "VICTOR IVAN CABRERA CABRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11456,
        "Identificacion": "0102155025",
        "NombreCompleto": "JOSE RICARDO SERRANO SALGADO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11457,
        "Identificacion": "0102176526",
        "NombreCompleto": "LUIS ALBERTO MENDEZ VINTIMILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11458,
        "Identificacion": "0102206539",
        "NombreCompleto": "DIEGO IVAN GARCIA CARDENAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11459,
        "Identificacion": "0102252517",
        "NombreCompleto": "LUIS ARTURO GUAMAN GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11460,
        "Identificacion": "0102353034",
        "NombreCompleto": "CARLOS RODRIGO MENDEZ PEREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11461,
        "Identificacion": "0102378395",
        "NombreCompleto": "FREDDY HERNAN ESPINOZA CALLE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11462,
        "Identificacion": "0102383981",
        "NombreCompleto": "MANUEL ALEJANDRO MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11463,
        "Identificacion": "0102503216",
        "NombreCompleto": "DIEGO FERNANDO ANDRADE TORRES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11464,
        "Identificacion": "0102581212",
        "NombreCompleto": "JUAN CARLOS LOPEZ PATIÑO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11465,
        "Identificacion": "0102651064",
        "NombreCompleto": "FATIMA IRENE SANCHEZ TAPIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11466,
        "Identificacion": "0102676434",
        "NombreCompleto": "JUAN CARLOS LOPEZ QUIZHPI",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11467,
        "Identificacion": "0102708492",
        "NombreCompleto": "MONICA ESPERANZA GONZALEZ GONZALEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11468,
        "Identificacion": "0102708492-",
        "NombreCompleto": "MONICA ESPERANZA GONZALEZ GONZALEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11469,
        "Identificacion": "0102710019",
        "NombreCompleto": "CARLOS ALBERTO ROJAS PACURUCU",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11470,
        "Identificacion": "0102814548",
        "NombreCompleto": "ESTEBAN LEONARDO CALLE CALLE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11471,
        "Identificacion": "0102946993",
        "NombreCompleto": "LUIS ENRIQUE GUAMAN SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11472,
        "Identificacion": "0103020855",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11473,
        "Identificacion": "0103488896",
        "NombreCompleto": "CARLOS ARMANDO PEREZ BRITO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11474,
        "Identificacion": "0103747093",
        "NombreCompleto": "JORGE EDUARDO CABRERA PEREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11475,
        "Identificacion": "0103830527",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ CARPIO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11476,
        "Identificacion": "0103923546",
        "NombreCompleto": "MARCO ANTONIO ANDRADE FLORES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11477,
        "Identificacion": "0103975645",
        "NombreCompleto": "JOSE RIGOBERTO CABRERA LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11478,
        "Identificacion": "0104001110",
        "NombreCompleto": "JOSE ANTONIO CALDERON BARRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11479,
        "Identificacion": "0104028527",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ UZHCA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11480,
        "Identificacion": "0104031265",
        "NombreCompleto": "CARLOS FERNANDO GOMEZ CRESPO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11481,
        "Identificacion": "0104164363",
        "NombreCompleto": "JORGE LUIS GARCIA TAPIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11482,
        "Identificacion": "0104200803",
        "NombreCompleto": "JORGE ENRIQUE VARGAS SICHA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11483,
        "Identificacion": "0104279518",
        "NombreCompleto": "ANA LUCIA GUAMAN GUERRERO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11484,
        "Identificacion": "0104545686",
        "NombreCompleto": "CARLOS ALBERTO LOPEZ VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11485,
        "Identificacion": "0104545686",
        "NombreCompleto": "LOPEZ VERA LOPEZ VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11486,
        "Identificacion": "0104627930",
        "NombreCompleto": "MARIA HORTENCIA JIMENEZ JIMENEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11487,
        "Identificacion": "0104628656",
        "NombreCompleto": "JUAN JOSE DELGADO ORAMAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11488,
        "Identificacion": "0104648282",
        "NombreCompleto": "JOSE VICENTE ZAMBRANO TENESACA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11489,
        "Identificacion": "0104968367",
        "NombreCompleto": "JOSE LUIS CABRERA LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11490,
        "Identificacion": "0105090807",
        "NombreCompleto": "JOSE LORENZO SALINAS SALINAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11491,
        "Identificacion": "0105094163",
        "NombreCompleto": "MARIA ELISABETH RAMOS BRITO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11492,
        "Identificacion": "0105362222",
        "NombreCompleto": "JAVIER EDUARDO SANCHEZ SAMANIEGO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11493,
        "Identificacion": "0105376313",
        "NombreCompleto": "MIGUEL ANGEL RODRIGUEZ RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11494,
        "Identificacion": "0150609428",
        "NombreCompleto": "JOSE LUIS ROBLES JARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11495,
        "Identificacion": "0200049401",
        "NombreCompleto": "LUIS ALFONSO RAMIREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11496,
        "Identificacion": "0200267862",
        "NombreCompleto": "LUIS ALBERTO GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11497,
        "Identificacion": "0200386837",
        "NombreCompleto": "FLOR MARIA GARCIA CARRILLO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11498,
        "Identificacion": "0200408938",
        "NombreCompleto": "JORGE ALBERTO GARCIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11499,
        "Identificacion": "0200426831",
        "NombreCompleto": "LUIS ALBERTO LARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11500,
        "Identificacion": "0200437721",
        "NombreCompleto": "LUIS ALFREDO RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11501,
        "Identificacion": "0200510428",
        "NombreCompleto": "SAUL WASHINGTON ESCOBAR ORELLANA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11502,
        "Identificacion": "0200575470",
        "NombreCompleto": "LUIS ALBERTO RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11503,
        "Identificacion": "0200642759",
        "NombreCompleto": "LUIS VICENTE SOLANO ANGULO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11504,
        "Identificacion": "0200671154",
        "NombreCompleto": "PEDRO ALBERTO VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11505,
        "Identificacion": "0200680916",
        "NombreCompleto": "MIGUEL ANGEL SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11506,
        "Identificacion": "0200764603",
        "NombreCompleto": "CARLOS ALBERTO ZAPATA SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11507,
        "Identificacion": "0200785806",
        "NombreCompleto": "LAURA BEATRIZ GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11508,
        "Identificacion": "0200788768",
        "NombreCompleto": "LUIS BOLIVAR VERGARA ORTIZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11509,
        "Identificacion": "0200944296",
        "NombreCompleto": "ANGEL VICENTE HURTADO REA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11510,
        "Identificacion": "0200947141",
        "NombreCompleto": "JORGE LUIS RODRIGUEZ VASCONEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11511,
        "Identificacion": "0200959179",
        "NombreCompleto": "JOSE SALVADOR JIMENEZ MODUMBA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11512,
        "Identificacion": "0200972636",
        "NombreCompleto": "ANGEL OSWALDO BONILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11513,
        "Identificacion": "0200980290",
        "NombreCompleto": "JUAN CARLOS GONZALEZ LOMBEIDA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11514,
        "Identificacion": "0200980308",
        "NombreCompleto": "GUILLERMO EDUARDO RODRIGUEZ PAREDES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11515,
        "Identificacion": "0201067956",
        "NombreCompleto": "MARIA LUCILA RAMOS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11516,
        "Identificacion": "0201155264",
        "NombreCompleto": "CARLOS ALBERTO ZAPATA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11517,
        "Identificacion": "0201161072",
        "NombreCompleto": "LUIS ENRIQUE SALAZAR LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11518,
        "Identificacion": "0201164712",
        "NombreCompleto": "VICTOR HUGO MENDOZA VERDESOTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11519,
        "Identificacion": "0201216694",
        "NombreCompleto": "FRANKLIN EDUARDO GARCIA GALLEGOS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11520,
        "Identificacion": "0201229440",
        "NombreCompleto": "MARIA TRANSITO HERRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11521,
        "Identificacion": "0201317443",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ VELASCO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11522,
        "Identificacion": "0201563384",
        "NombreCompleto": "MARCO ANTONIO GARCIA GARCIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11523,
        "Identificacion": "0201576949",
        "NombreCompleto": "JUAN CARLOS MUÑOZ SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11524,
        "Identificacion": "0201605607",
        "NombreCompleto": "JOSE LUIS SANCHEZ MOYA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11525,
        "Identificacion": "0201607256",
        "NombreCompleto": "JOSE LUIS RUIZ SOTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11526,
        "Identificacion": "0201610292",
        "NombreCompleto": "WALTER HERNAN GRANIZO HERRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11527,
        "Identificacion": "0201644994",
        "NombreCompleto": "JOSE LUIS SALAZAR CAJILEMA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11528,
        "Identificacion": "0201688686",
        "NombreCompleto": "JUAN CARLOS BRAVO GAIBOR",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
]
if __name__ == "__main__":

    model = SentenceTransformer(EmbeddingModelType.PARAPHRASE.value)
    d = model.get_sentence_embedding_dimension()
    index = faiss.IndexHNSWFlat(d, 32)  # 32 vecinos por nodo
    index.hnsw.efConstruction = 200  # Construcción rápida del gráfico
    sentences = [
        f"{normalize(item["NombreCompleto"])} {item["Identificacion"]}" for item in data
    ]

    embeddings = model.encode(sentences, show_progress_bar=True).astype("float32")
    print(f"Embeddings para {len(sentences)} elementos")

    query = "juan carlos"
    query_embedding = model.encode([query], show_progress_bar=False).astype("float32")
    index.add(embeddings)
    distances, indices = index.search(query_embedding, 10)
    d = distances[0]
    i = indices[0]

    max_d = max(d.max(), 1e-10)
    results = []
    for dist, idx in zip(d, i):
        similarity = 1 - (dist / max_d) if max_d > 0 else 0
        if idx == -1:
            continue
        result = sentences[idx]
        results.append(
            {"result": result, "similarity": similarity, "distance": f"{dist:.2f}"}
        )

    df = pd.DataFrame(results)
    print(f"query: {query}")
    print(df)
# em = EmbeddingManager(
#     model_type=EmbeddingModelType.PARAPHRASE,
#     index_type=FaissIndexType.HNSWFLAT,
# )
# inputs = ["burbano paul cirstan"]
# for input in inputs:
#     print(f"Searching for: {input}")
#     result1 = em.search(input, type=SearchType.COSINE)
#     result2 = em.search(input, type=SearchType.DEFAULT)
#     result3 = em.search(
#         input,
#         type=SearchType.CAMBERRA,
#     )
#     print("Cosine Similarity")
#     print(result1)
#     print("")
#     print("Default")
#     print(result2)
#     print("")
#     print("Camberra")
#     print(result3)
#     print("-" * 50)
