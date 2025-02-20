import pandas as pd
from rapidfuzz import process

from util import normalize
from database.connection import Database
from database.dao.identificacion import IdentificacionDAO


class FuzzyManager:
    """
    Clase para administrar un índice basado en RapidFuzz que permite
    insertar, eliminar, buscar y actualizar datos.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo y carga datos iniciales desde la base de datos.
        """

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
        return {id: text for id, text in zip(df["id"], texts)}

    def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 60,
    ):
        """
        Busca los k elementos más similares a la consulta 'query'.
        """
        query_norm = normalize(query)
        results = process.extract(
            query_norm, self.embeddings, score_cutoff=threshold, limit=k
        )

        result_data = [
            {
                "id": id,
                "identificacion": self.data.loc[
                    self.data["id"] == id, "identificacion"
                ].values[0],
                "nombre": self.data.loc[self.data["id"] == id, "nombre"].values[0],
                "score": f"{score:.2f}%",
            }
            for _, score, id in results
        ]

        return pd.DataFrame(result_data)
