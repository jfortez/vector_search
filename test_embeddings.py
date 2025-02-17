
from database import get_identificacion
from embedding import (create_index, create_embedding, searchFromIndex)
import numpy as np

data = get_identificacion()
ids = data['id'].tolist()
# Crear índice
index = create_index()
# crear embeddings
embeddings = create_embedding(data)


def test_has_data():
    assert len(data) > 0


def test_embedding():

    # añadir datos
    index.add(np.array(embeddings, dtype='float32'))

    result = searchFromIndex("Juan Perez")
    assert len(result[0]) > 0
