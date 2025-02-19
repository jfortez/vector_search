import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer

from util import normalize
from database.connection import Database
from database.dao.identificacion import IdentificacionDAO
from models.identificacion import Identificacion, IdentificacionCreate
from util import generate_unique_id


class RapidfuzzIndexManager:
    """
    Clase para administrar un índice basado en RapidFuzz que permite
    insertar, eliminar, buscar y actualizar datos.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo y carga datos iniciales desde la base de datos.
        """
        self.model = SentenceTransformer(model_name)
        db = Database()
        self.db = IdentificacionDAO(db)
        self.data = self.db.get_all().copy()
        self.embeddings = self._create_embeddings(self.data)
        print(
            f"[INIT] Modelo: {model_name} | Datos cargados: {len(self.data)} registros."
        )

    def _create_embeddings(self, df: pd.DataFrame) -> dict:
        """
        Crea un diccionario de embeddings para búsqueda rápida.
        """
        texts = df.apply(
            lambda row: normalize(f"{row['nombre']} {row['identificacion']}"), axis=1
        ).tolist()
        embeddings = {idx: text for idx, text in zip(df["id"], texts)}
        return embeddings

    def search(self, query: str, threshold: float = 75, k: int = 10):
        """
        Busca los k elementos más similares a la consulta 'query'.
        """
        query_norm = normalize(query)
        results = process.extract(
            query_norm, self.embeddings, scorer=fuzz.ratio, limit=k
        )

        result_data = [
            {
                "id": idx,
                "nombre": (
                    self.data.loc[self.data["id"] == idx, "nombre"].values[0]
                    if not self.data.loc[self.data["id"] == idx, "nombre"].empty
                    else None
                ),
                "identificacion": (
                    self.data.loc[self.data["id"] == idx, "identificacion"].values[0]
                    if not self.data.loc[self.data["id"] == idx, "identificacion"].empty
                    else None
                ),
                "similaridad": f"{score}%",
            }
            for idx, text, score in results
            if score >= threshold
        ]

        return pd.DataFrame(result_data)

    def insert_data(self, new_row: IdentificacionCreate):
        """
        Inserta nuevos registros en el índice.
        """
        self.db.insert(new_row)

        # self.embeddings.update(self._create_embeddings(new_data))
        # print(f"[INSERT] Insertados {len(new_data)} registros. Total: {len(self.data)}")

    def delete_data(self, id: int):
        """
        Elimina registros del índice.
        """
        self.db.delete(id)
        self.embeddings.pop(id, None)
        print(f"[DELETE] Eliminados {id} registros. Total: {len(self.data)}")

    def update_data(self, row: Identificacion):
        """
        Actualiza registros en la base de datos.
        """
        self.db.update(row)
        # self.embeddings[row.id] = normalize(f"{row.nombre} {row.identificacion}")
        # print(f"[UPDATE] Registros actualizados: {len(updated_data)}")


# Ejemplo de uso
if __name__ == "__main__":
    manager = RapidfuzzIndexManager()

    # Buscar un término
    query = "Juan frnandez"
    print("Resultados de búsqueda:")
    print(manager.search(query))

    # Insertar nuevos datos

    manager.insert_data(
        IdentificacionCreate(identificacion=generate_unique_id(), nombre="Ana Gómez")
    )

    # Actualizar datos

    manager.update_data(
        Identificacion(id=101, identificacion="123456789", nombre="Ana González")
    )

    # Eliminar datos
    manager.delete_data(manager.data["id"].tolist()[0])
