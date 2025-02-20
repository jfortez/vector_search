import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComplexData:
    """Clase para representar datos complejos"""

    id: int
    name: str
    vector: np.ndarray
    metadata: Dict


class FAISSSearchEngine:
    """Motor de búsqueda usando FAISS con almacenamiento en array"""

    def __init__(self, dimension: int = 128):
        """
        Inicializa el motor de búsqueda

        Args:
            dimension (int): Dimensión de los vectores
        """
        self.dimension = dimension
        self.data_store: List[ComplexData] = []

        # Inicializar índice FAISS (usamos IndexFlatL2 para simplicidad)
        self.index = faiss.IndexFlatL2(dimension)
        self.next_id = 0

    def populate_sample_data(self, num_samples: int = 100):
        """Pobla el almacenamiento con datos de ejemplo"""
        try:
            for i in range(num_samples):
                # Generar datos de ejemplo
                name = f"Item_{i}"
                vector = np.random.random(self.dimension).astype(np.float32)
                metadata = {"category": f"cat_{i % 5}", "value": i}

                # Crear objeto y añadir al almacenamiento
                data_item = ComplexData(self.next_id, name, vector, metadata)
                self.data_store.append(data_item)

                # Agregar al índice FAISS
                self.index.add(vector.reshape(1, -1))
                self.next_id += 1

            logger.info(f"Poblados {num_samples} registros exitosamente")

        except Exception as e:
            logger.error(f"Error al poblar datos: {e}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[ComplexData]:
        """
        Realiza búsqueda de k-vecinos más cercanos

        Args:
            query_vector (np.ndarray): Vector de consulta
            k (int): Número de resultados a devolver

        Returns:
            List[ComplexData]: Lista de resultados ordenados por similitud
        """
        if query_vector.shape[-1] != self.dimension:
            raise ValueError(
                f"El vector de consulta debe tener dimensión {self.dimension}"
            )

        # Realizar búsqueda en FAISS
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)

        # Obtener resultados completos
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.data_store):
                result = self.data_store[idx]
                setattr(result, "distance", float(distance))
                results.append(result)

        return results

    def add_new_item(self, name: str, vector: np.ndarray, metadata: Dict) -> int:
        """Añade un nuevo item al índice y almacenamiento"""
        try:
            if vector.shape[-1] != self.dimension:
                raise ValueError(f"El vector debe tener dimensión {self.dimension}")

            # Crear nuevo item
            new_item = ComplexData(self.next_id, name, vector, metadata)
            self.data_store.append(new_item)

            # Añadir al índice FAISS
            self.index.add(vector.reshape(1, -1))
            self.next_id += 1

            return new_item.id

        except Exception as e:
            logger.error(f"Error al añadir item: {e}")
            raise

    def get_item(self, item_id: int) -> Optional[ComplexData]:
        """Obtiene un item por su ID"""
        for item in self.data_store:
            if item.id == item_id:
                return item
        return None

    def remove_item(self, item_id: int) -> bool:
        """Elimina un item por su ID"""
        try:
            for i, item in enumerate(self.data_store):
                if item.id == item_id:
                    # FAISS no permite eliminación directa, necesitamos reconstruir el índice
                    self.data_store.pop(i)
                    self._rebuild_index()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error al eliminar item: {e}")
            raise

    def _rebuild_index(self):
        """Reconstruye el índice FAISS desde el almacenamiento"""
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.data_store:
            vectors = np.array([item.vector for item in self.data_store])
            self.index.add(vectors)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del motor de búsqueda
    search_engine = FAISSSearchEngine(dimension=128)

    # Poblar con datos de ejemplo
    search_engine.populate_sample_data(1000)

    # Ejemplo de búsqueda
    query_vector = np.random.random(128).astype(np.float32)
    results = search_engine.search(query_vector, k=5)

    print(query_vector)

    # Mostrar resultados
    for result in results:
        print(f"ID: {result.id}, Name: {result.name}, Distance: {result.distance}")
        print(f"Metadata: {result.metadata}")
        print("---")

    # Ejemplo de añadir nuevo item
    new_vector = np.random.random(128).astype(np.float32)
    new_id = search_engine.add_new_item(
        "New_Item", new_vector, {"category": "test", "value": 999}
    )
    print(f"Nuevo item añadido con ID: {new_id}")
