import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from util import normalize
from database.dao.identificacion import IdentificacionDAO
from models.identificacion import Identificacion, IdentificacionCreate


class FaissIndexManager:
    """
    Clase para administrar un índice FAISS que permite
    insertar, eliminar, buscar y actualizar datos.
    """

    def __init__(self, dao: IdentificacionDAO, data):
        """
        Inicializa el modelo y crea el índice FAISS. Carga datos iniciales
        desde la base de datos y construye los embeddings.
        """
        # Cargamos el modelo y obtenemos la dimensión directamente de él
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Empleamos un IndexIDMap para poder manejar IDs de forma nativa en FAISS
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        self.db = dao

        # Cargamos datos iniciales
        self.data = data

        # Generamos embeddings iniciales
        embeddings = self._create_embeddings(self.data)
        ids = self.data["id"].astype(np.int64).values

        # Insertamos en FAISS
        self.index.add_with_ids(embeddings, ids)

        # Mantendremos en memoria una copia de los embeddings
        self.embeddings = embeddings

        print(f"[INIT] Dimensión: {self.dimension}")
        print(f"[INIT] Datos cargados: {len(self.data)} registros.")
        print(f"[INIT] Total elementos en el índice FAISS: {self.get_index_length()}")

    def _create_embeddings(self, df) -> np.ndarray:
        """
        Crea embeddings para un DataFrame utilizando SentenceTransformer.
        """
        if isinstance(df, pd.Series):
            texts = [normalize(f"{df['nombre']}{df['identificacion']}")]
        else:
            texts = df.apply(
                lambda row: normalize(f"{row['nombre']}{row['identificacion']}"), axis=1
            ).tolist()

        return self.model.encode(texts).astype("float32")

    def search(self, query: str, threshold: float = 0.1, k: int = 10):
        """
        Busca en el índice FAISS los k elementos más similares a la consulta 'query'.
        """
        query_embedding = self.model.encode([normalize(query)]).astype("float32")
        distances, faiss_ids = self.index.search(query_embedding, k)

        max_distance = distances[0].max() if distances[0].max() != 0 else 1

        similarity_results = []
        result_data = []

        for dist, idx_id in zip(distances[0], faiss_ids[0]):
            if idx_id == -1:
                continue

            registro = self.data.loc[self.data["id"] == idx_id]
            if registro.empty:
                continue

            registro = registro.iloc[0]
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

        return pd.DataFrame(result_data), pd.DataFrame(similarity_results)

    def insert_data(self, new_row: IdentificacionCreate):
        """
        Inserta un nuevo registro en el índice FAISS y en la base de datos.
        """
        # Primero insertamos en la base de datos
        self.db.insert(new_row)

        # Actualizamos los datos en memoria
        self._revalidate_data()

        # Obtenemos el último registro insertado
        latest_record = self.data.iloc[-1]

        # Creamos embedding para el nuevo registro
        new_embedding = self._create_embeddings(latest_record)

        # Añadimos al índice FAISS
        self.index.add_with_ids(
            new_embedding, np.array([latest_record["id"]], dtype=np.int64)
        )

        # Actualizamos embeddings en memoria
        self.embeddings = np.vstack([self.embeddings, new_embedding])

        print(f"[INSERT] Nuevo registro insertado con ID: {latest_record['id']}")
        print(f"[INSERT] Total elementos en el índice FAISS: {self.get_index_length()}")

    def update_data(self, row: Identificacion):
        """
        Actualiza un registro en la base de datos e índice FAISS.
        """
        # Verificamos si el ID existe
        if not self.data["id"].isin([row.id]).any():
            print(f"[UPDATE] ID {row.id} no existe en la base de datos.")
            return

        # Actualizamos en la base de datos
        self.db.update(row)

        # Actualizamos datos en memoria
        self._revalidate_data()

        # Actualizamos el índice FAISS
        self._rebuild_index()

        print(f"[UPDATE] Registro actualizado con ID: {row.id}")
        print(f"[UPDATE] Total elementos en el índice FAISS: {self.get_index_length()}")

    def delete_data(self, id: int):
        """
        Elimina un registro del índice FAISS y de la base de datos.
        """
        # Verificamos si el ID existe
        if not self.data["id"].isin([id]).any():
            print(f"[DELETE] ID {id} no existe en la base de datos.")
            return

        # Eliminamos de la base de datos
        self.db.delete(id)

        # Actualizamos datos en memoria
        self._revalidate_data()

        # Reconstruimos el índice excluyendo el ID eliminado
        self._rebuild_index_excluding(np.array([id], dtype=np.int64))

        print(f"[DELETE] Registro eliminado con ID: {id}")
        print(f"[DELETE] Total elementos en el índice FAISS: {self.get_index_length()}")

    def _revalidate_data(self):
        """
        Actualiza los datos en memoria desde la base de datos.
        """
        self.data = self.db.get_all().copy()

    def _rebuild_index_excluding(self, exclude_ids: np.ndarray):
        """
        Reconstruye el índice excluyendo IDs específicos.
        """
        exclude_ids = set(exclude_ids)
        df_remaining = self.data[~self.data["id"].isin(exclude_ids)]

        embeddings = self._create_embeddings(df_remaining)
        ids = df_remaining["id"].astype(np.int64).values

        self.index.reset()
        self.index.add_with_ids(embeddings, ids)
        self.embeddings = embeddings

    def _rebuild_index(self):
        """
        Reconstruye el índice FAISS completamente.
        """
        embeddings = self._create_embeddings(self.data)
        ids = self.data["id"].astype(np.int64).values

        self.index.reset()
        self.index.add_with_ids(embeddings, ids)
        self.embeddings = embeddings

    def get_index_length(self) -> int:
        """
        Retorna la cantidad de elementos en el índice FAISS.
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
#     query = manager.data["nombre"].tolist()[0]
#     print(f"[MAIN] Búsqueda con consulta: {query}")
#     _, similarity_df = manager.search(query)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     # DELETING ITEMS AND SEARCHING AGAIN
#     print_line()
#     delete_target = manager.data["id"].tolist()[0]
#     print("[MAIN] Eliminando items con ID:", delete_target)
#     print("[MAIN] Longitud inicial de datos:", len(manager.data))
#     manager.delete_data(delete_target)
#     print("[MAIN] Longitud final de datos:", len(manager.data))
#     _, similarity_df = manager.search(query)
#     print("[MAIN] Resultados de búsqueda después de eliminación:")
#     print("")
#     print(similarity_df)
#     print("")
#     print_line()
#     # INSERTING NEW DATA AND SEARCHING AGAIN
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())

#     from util import generate_unique_id

#     new_row = {
#         "identificacion": generate_unique_id(),
#         "nombre": "Guillermo Mendoza",
#     }
#     print("[MAIN] Insertando nuevos datos:")
#     print("")
#     print(new_row)
#     print("")

#     manager.insert_data(new_row["identificacion"], new_row["nombre"])
#     print("[MAIN] Nuevos datos insertados. Total registros:", len(manager.data))

#     query_new = new_row["nombre"]
#     print(f"[MAIN] Búsqueda con consulta: {query_new}")
#     _, similarity_df = manager.search(query_new)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())
#     print_line()

#     # UPDATING DATA AND SEARCHING AGAIN
#     target_update = Identificacion(
#         id=11, identificacion="123123123222", nombre="Pedro Duarte"
#     )

#     print(f"[MAIN] Actualizando datos para \n{target_update}")
#     manager.update_data(target_update)
#     print("[MAIN] Datos actualizados. Total registros:", len(manager.data))

#     query_updated = target_update.nombre
#     print(f"[MAIN] Búsqueda con consulta: {query_updated}")
#     _, similarity_df = manager.search(query_updated)
#     print("[MAIN] Resultados de búsqueda:")
#     print("")
#     print(similarity_df)
#     print("")
#     print("[MAIN] Total elementos en el índice FAISS:", manager.get_index_length())

#     # Ejemplo de debug
#     # manager.debug_last_embedding()
