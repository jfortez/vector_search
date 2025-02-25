import re
import unicodedata


# Modificamos la expresión regular para excluir explícitamente el guion bajo
_alnum_regex = re.compile(r"(?ui)[^\w\s]|_")


def normalize(s):
    # 1. Normalizar caracteres acentuados a su versión base (ej. á → a, ñ → n)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

    # 2. Eliminar caracteres no alfanuméricos (y reemplazar guiones bajos por espacios)
    string_out = _alnum_regex.sub(" ", s)

    # 3. Retornar en minúsculas y sin espacios extras
    return " ".join(string_out.split()).lower()


import uuid


def generate_unique_id(length=8):
    """Genera un número de identificación único para las pruebas."""
    return str(uuid.uuid4().int)[:length]
