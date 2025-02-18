import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from util import normalize
from database import get_identificacion


class FaissIndexManager:
    """
    Clase para administrar un índice FAISS que permite
    insertar, eliminar, buscar y actualizar datos.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo y crea el índice FAISS. Carga datos iniciales
        desde la base de datos y construye los embeddings.
        """
        # Cargamos el modelo y obtenemos la dimensión directamente de él
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Empleamos un IndexIDMap para poder manejar IDs de forma nativa en FAISS
        # (así evitamos tener que manejar manualmente la lista 'ids').
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

        # Cargamos datos iniciales
        self.data = get_identificacion().copy()

        # Generamos embeddings iniciales
        embeddings = self._create_embeddings(self.data)
        ids = self.data["id"].astype(np.int64).values  # FAISS requiere IDs tipo int64

        # Insertamos en FAISS
        self.index.add_with_ids(embeddings, ids)

        # Mantendremos en memoria una copia de los embeddings para facilitar
        # la depuración y la actualización.
        self.embeddings_in_memory = embeddings

        print(f"[INIT] Modelo: {model_name} | Dimensión: {self.dimension}")
        print(f"[INIT] Datos cargados: {len(self.data)} registros.")
        print(f"[INIT] Total elementos en el índice FAISS: {self.get_index_length()}")

    def _create_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Crea embeddings para un DataFrame utilizando SentenceTransformer.
        """
        texts = df.apply(
            lambda row: normalize(f"{row['nombre']}{row['identificacion']}"), axis=1
        ).tolist()

        # Generamos embeddings y nos aseguramos de que sean float32
        return self.model.encode(texts).astype("float32")

    def search(self, query: str, threshold: float = 0.1, k: int = 10):
        """
        Busca en el índice FAISS los k elementos más similares a la consulta 'query'.
        Devuelve un DataFrame con los resultados y otro con la similitud formateada.
        """
        # Generar embedding de la consulta
        query_embedding = self.model.encode([normalize(query)]).astype("float32")

        # Buscar en FAISS
        distances, faiss_ids = self.index.search(query_embedding, k)

        # Para evitar división por cero
        max_distance = distances[0].max() if distances[0].max() != 0 else 1

        # Preparar resultados
        similarity_results = []
        result_data = []

        for dist, idx_id in zip(distances[0], faiss_ids[0]):
            # FAISS puede devolver -1 si no encuentra suficientes resultados
            if idx_id == -1:
                continue

            # Recuperamos la fila correspondiente al ID
            registro = self.data.loc[self.data["id"] == idx_id]
            if registro.empty:
                continue

            # Solo hay una fila por ID, así que tomamos el primero
            registro = registro.iloc[0]

            # Cálculo de similitud
            similarity = 1 - dist / max_distance
            if similarity >= threshold:
                result_text = f"{registro['nombre']} {registro['identificacion']}"
                similarity_results.append(
                    {
                        "ID": int(idx_id),
                        "Registro": result_text,
                        "Similaridad": f"{similarity:.2%}",
                    }
                )
                result_data.append(
                    {
                        "id": registro["id"],
                        "identificacion": registro["identificacion"],
                        "nombre": registro["nombre"],
                    }
                )

        df_data = pd.DataFrame(result_data)
        df_sim = pd.DataFrame(similarity_results)
        return df_data, df_sim

    def insert_data(self, new_data: pd.DataFrame):
        """
        Inserta nuevos registros en el índice FAISS y en el DataFrame interno.
        """
        # Convertimos a copia para evitar problemas si 'new_data' se modifica fuera
        new_data = new_data.copy()

        # Creamos embeddings
        new_embeddings = self._create_embeddings(new_data)
        new_ids = new_data["id"].astype(np.int64).values

        # Añadir al índice
        self.index.add_with_ids(new_embeddings, new_ids)

        # Actualizamos la copia en memoria
        self.embeddings_in_memory = np.concatenate(
            [self.embeddings_in_memory, new_embeddings], axis=0
        )

        # Añadimos al DataFrame interno
        self.data = pd.concat([self.data, new_data], ignore_index=True)

        print(f"[INSERT] Insertados {len(new_data)} nuevos registros.")
        print(f"[INSERT] Total elementos en el índice FAISS: {self.get_index_length()}")

    def delete_data(self, delete_ids: list):
        """
        Elimina registros del índice FAISS y del DataFrame interno
        según los IDs proporcionados.
        """
        if not delete_ids:
            print("[DELETE] Lista de IDs vacía, no se elimina nada.")
            return

        # Convertir a np.int64 para FAISS
        delete_ids = np.array(delete_ids, dtype=np.int64)

        # Verificar cuáles de esos IDs existen realmente en 'self.data'
        existing_ids = self.data[self.data["id"].isin(delete_ids)]["id"].unique()
        if len(existing_ids) == 0:
            print("[DELETE] Ninguno de los IDs existe en la base de datos.")
            return

        # Eliminamos del índice
        # Nota: IndexIDMap con IndexFlatL2 no soporta remove_ids.
        # Se reconstruye el índice sin esos IDs.
        self._rebuild_index_excluding(existing_ids)

        # Eliminamos del DataFrame
        self.data = self.data[~self.data["id"].isin(existing_ids)].reset_index(
            drop=True
        )

        print(f"[DELETE] Eliminados {len(existing_ids)} registros.")
        print(f"[DELETE] Total elementos en el índice FAISS: {self.get_index_length()}")

    def _rebuild_index_excluding(self, exclude_ids: np.ndarray):
        """
        Reconstruye el índice excluyendo una lista de IDs.
        (Para IndexFlatL2 sin remove_ids nativo)
        """
        exclude_ids = set(exclude_ids)
        # Filtramos el DataFrame para quedarnos con los que NO están en exclude_ids
        df_remaining = self.data[~self.data["id"].isin(exclude_ids)]

        # Regeneramos embeddings_in_memory
        embeddings = self._create_embeddings(df_remaining)
        ids = df_remaining["id"].astype(np.int64).values

        # Reseteamos y volvemos a construir
        self.index.reset()
        self.index.add_with_ids(embeddings, ids)

        # Actualizamos self.embeddings_in_memory
        self.embeddings_in_memory = embeddings

    def update_data(self, updated_data: pd.DataFrame):
        """
        Actualiza uno o más registros en la base de datos e índice FAISS.
        """
        updated_data = updated_data.copy()

        # Verificamos qué IDs existen en la base
        existing_ids = self.data[self.data["id"].isin(updated_data["id"])].copy()
        if existing_ids.empty:
            print("[UPDATE] Ninguno de los IDs existe en la base de datos.")
            return

        # Mezclamos la info para actualizar 'nombre' e 'identificacion'
        # en self.data
        self.data.set_index("id", inplace=True)
        updated_data.set_index("id", inplace=True)

        for idx in updated_data.index:
            if idx not in self.data.index:
                print(f"[UPDATE] ID {idx} no existe en la base, se ignora.")
                continue

            self.data.loc[idx, "nombre"] = updated_data.loc[idx, "nombre"]
            self.data.loc[idx, "identificacion"] = updated_data.loc[
                idx, "identificacion"
            ]

        # Volvemos a resetear el índice de self.data
        self.data.reset_index(inplace=True)

        # Para IndexFlatL2, no hay método directo de actualización.
        # Volvemos a reconstruir completamente el índice.
        self._rebuild_index()

        print(f"[UPDATE] Registros actualizados: {len(updated_data)}.")
        print(f"[UPDATE] Total elementos en el índice FAISS: {self.get_index_length()}")

    def _rebuild_index(self):
        """
        Reconstruye el índice FAISS desde cero usando self.data.
        """
        embeddings = self._create_embeddings(self.data)
        ids = self.data["id"].astype(np.int64).values

        self.index.reset()
        self.index.add_with_ids(embeddings, ids)

        # Actualizamos la copia en memoria
        self.embeddings_in_memory = embeddings

    def debug_last_embedding(self):
        """
        Muestra el último embedding en memoria y su ID correspondiente,
        para depuración.
        """
        if self.get_index_length() == 0:
            print("[DEBUG] El índice está vacío.")
            return

        print(f"[DEBUG] Total en el índice: {self.get_index_length()}")

        stored_ids = self._get_all_ids()
        if len(stored_ids) == 0:
            print("[DEBUG] No hay IDs en el índice.")
            return

        last_id = stored_ids[-1]
        last_embedding = self.index.reconstruct(last_id)
        print(f"[DEBUG] Último ID en el índice: {last_id}")
        print(f"[DEBUG] Último embedding:\n{last_embedding}")

    def _get_all_ids(self) -> np.ndarray:
        """
        Retorna un array de todos los IDs almacenados en el índice FAISS.
        """
        # 'downcast_index' convierte nuestro IndexIDMap a su tipo real
        index_id_map = faiss.downcast_index(self.index)
        return index_id_map.id_map.copy()

    def get_index_length(self) -> int:
        """
        Retorna la longitud del índice FAISS.
        """
        return self.index.ntotal


# def print_line():
#     print("")
#     print("-" * 28)
#     print("")


# if __name__ == "__main__":
#     manager = FaissIndexManager()  # Por defecto, "all-MiniLM-L6-v2"
#     #  INITIAL QUERY
#     print("\n[MAIN] Datos iniciales:")
#     print(manager.data.head(10))
#     print_line()
#     # INITIAL SEARCH
#     query = "Juan Perez"
#     print(f"[MAIN] Búsqueda con consulta: {query}")
#     _, similarity_df = manager.search(query)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     # DELETING ITEMS AND SEARCHING AGAIN
#     print_line()
#     ids_to_delete = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     print("[MAIN] Eliminando items con ID:", ids_to_delete)
#     print("[MAIN] Longitud inicial de datos:", len(manager.data))
#     manager.delete_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     print("[MAIN] Longitud final de datos:", len(manager.data))
#     _, similarity_df = manager.search(query)
#     print("[MAIN] Resultados de búsqueda después de eliminación:")
#     print("")
#     print(similarity_df)
#     print("")
#     print_line()
#     # INSERTING NEW DATA AND SEARCHING AGAIN
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())

#     new_data = pd.DataFrame(
#         [{"id": 123456, "identificacion": "23232123", "nombre": "Guillermo Mendoza"}]
#     )
#     print("[MAIN] Insertando nuevos datos:")
#     print("")
#     print(new_data)
#     print("")

#     manager.insert_data(new_data)
#     print("[MAIN] Nuevos datos insertados. Total registros:", len(manager.data))

#     query_new = new_data.iloc[0]["nombre"]
#     print(f"[MAIN] Búsqueda con consulta: {query_new}")
#     _, similarity_df = manager.search(query_new)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())
#     print_line()

#     # UPDATING DATA AND SEARCHING AGAIN
#     target_update = {
#         "id": 11,
#         "identificacion": "123123123222",
#         "nombre": "Pedro Duarte",
#     }
#     print(
#         f"[MAIN] Actualizando datos para ID: {target_update['id']}, nombre: {target_update['nombre']}, identificacion: {target_update['identificacion']}"
#     )
#     manager.update_data(pd.DataFrame([target_update]))
#     print("[MAIN] Datos actualizados. Total registros:", len(manager.data))

#     query_updated = target_update["nombre"]
#     print(f"[MAIN] Búsqueda con consulta: {query_updated}")
#     _, similarity_df = manager.search(query_updated)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())

#     # Ejemplo de debug
#     # manager.debug_last_embedding()
