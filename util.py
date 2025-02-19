import re
import unicodedata
import uuid


def generate_unique_id(length=8):
    """Genera un número de identificación único para las pruebas."""
    return str(uuid.uuid4().int)[:length]


_alnum_regex = re.compile(r"(?ui)\W")


def normalize(s):
    # 1. Normalizar caracteres acentuados a su versión base (ej. á → a, ñ → n)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

    # 2. Eliminar caracteres no alfanuméricos (incluye los apóstrofes)
    string_out = _alnum_regex.sub(" ", s)

    # 3. Retornar en minúsculas y sin espacios extras
    return string_out.strip().lower()
