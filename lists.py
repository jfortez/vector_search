import json
import threading
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from models.list import BaseList

from util import normalize


class DataStorage:
    """Maneja la persistencia de datos de Listas en disco."""

    def __init__(self, file_path: Optional[str] = "lists.json"):
        self.file_path = Path(file_path)
        self.url = "http://172.16.11.132:4000/api/getProcessedData"
        print(f"📦 [DataStorage] Inicializando almacenamiento en {self.file_path}...")
        self.storage: List[BaseList] = self._load_or_fetch_data()

    def _fetch_from_api(self) -> List[BaseList]:
        """Obtiene datos desde la API si no existen en disco."""
        print(f"🌐 [DataStorage] Solicitando datos desde {self.url}...")

        loading = True
        start_time = time.time()

        def show_loading():
            animation = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            idx = 0
            while loading:
                elapsed = time.time() - start_time
                print(
                    f"\r⏳ Esperando API {animation[idx % len(animation)]} ({elapsed:.1f}s)",
                    end="",
                )
                idx += 1
                time.sleep(0.1)
            print("\r" + " " * 50 + "\r", end="")  # Limpia la línea

        loading_thread = threading.Thread(target=show_loading)
        loading_thread.start()

        try:
            response = requests.get(self.url)
            response.raise_for_status()
            loading = False
            loading_thread.join()

            elapsed_time = time.time() - start_time
            print(f"✅ [DataStorage] Datos recibidos en {elapsed_time:.2f}s")

            data = response.json()
            print(f"📋 [DataStorage] Procesando {len(data)} registros desde API")
            return [
                BaseList(**item)
                for item in tqdm(data, desc="Convirtiendo datos", unit="reg")
            ]
        except requests.RequestException as e:
            loading = False
            loading_thread.join()
            elapsed_time = time.time() - start_time
            raise RuntimeError(
                f"❌ [DataStorage] Error tras {elapsed_time:.2f}s: {str(e)}"
            )

    def _load_or_fetch_data(self) -> List[BaseList]:
        """Carga datos desde archivo o los obtiene de la API."""
        if self.file_path.exists():
            return self._load_from_file()
        data = self._fetch_from_api()
        self._save_to_file(data)
        return data

    def _save_to_file(self, source: List[BaseList]) -> None:
        """Guarda las Listas en un archivo JSON."""
        print(
            f"💾 [DataStorage] Guardando {len(source)} registros en {self.file_path}..."
        )
        try:
            data = [
                element.model_dump()
                for element in tqdm(source, desc="Serializando", unit="reg")
            ]

            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            total_size = len(json_str.encode("utf-8"))

            with self.file_path.open("w", encoding="utf-8") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Escribiendo JSON"
                ) as pbar:
                    f.write(json_str)
                    pbar.update(total_size)

            print(f"✅ [DataStorage] Datos guardados exitosamente en {self.file_path}")
        except IOError as e:
            raise RuntimeError(f"❌ [DataStorage] Error al guardar: {str(e)}")

    def _load_from_file(self) -> List[BaseList]:
        """Carga las Listas desde un archivo JSON."""
        print(f"📂 [DataStorage] Cargando datos desde {self.file_path}...")
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                print(
                    f"📋 [DataStorage] Procesando {len(data)} registros desde archivo"
                )
                return [
                    BaseList(**item)
                    for item in tqdm(data, desc="Deserializando", unit="reg")
                ]
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"❌ [DataStorage] Error al cargar: {str(e)}")


class EmbeddingManager:
    """Gestiona la creación, carga y búsqueda de embeddings."""

    def __init__(self, embeddings_dir: Optional[str] = "embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        print("🤖 [EmbeddingManager] Cargando modelo de embeddings...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

        self.data_storage = DataStorage()
        self.data = self.data_storage.storage
        print(
            f"🧠 [EmbeddingManager] Inicializando embeddings para {len(self.data)} registros..."
        )
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> np.ndarray:
        """Inicializa embeddings cargándolos o creándolos."""
        if self.embeddings_dir.exists() and any(self.embeddings_dir.glob("*.npy")):
            return self._load_embeddings()
        return self._create_and_save_embeddings()

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
        self._save_embeddings(embeddings)
        return embeddings

    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Guarda los embeddings en archivos .npy por lotes."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        batch_size = 256
        total_batches = (len(embeddings) + batch_size - 1) // batch_size

        print(
            f"💾 [EmbeddingManager] Guardando {len(embeddings)} embeddings en {total_batches} batches..."
        )
        for i in tqdm(
            range(0, len(embeddings), batch_size),
            desc="Guardando batches",
            unit="batch",
        ):
            batch = embeddings[i : i + batch_size]
            file_idx = f"{i // batch_size:03d}"
            file_path = self.embeddings_dir / f"embeddings_{file_idx}.npy"
            np.save(file_path, batch)

    def _load_embeddings(self) -> np.ndarray:
        """Carga embeddings desde archivos .npy en orden numérico correcto."""
        npy_files = sorted(
            self.embeddings_dir.glob("embeddings_*.npy"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        if not npy_files:
            raise FileNotFoundError(
                "❌ [EmbeddingManager] No se encontraron archivos de embeddings"
            )

        print(
            f"📂 [EmbeddingManager] Cargando {len(npy_files)} archivos de embeddings..."
        )
        embeddings_list = [
            np.load(file)
            for file in tqdm(npy_files, desc="Cargando batches", unit="file")
        ]
        embeddings = np.vstack(embeddings_list).astype("float32")
        print(
            f"📈 [EmbeddingManager] Indexando {embeddings.shape[0]} embeddings en FAISS..."
        )
        self.index.add(embeddings)
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
    main()
