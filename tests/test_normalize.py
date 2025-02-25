from util import normalize


def test_basic_strings():
    """Prueba con cadenas básicas sin complicaciones."""
    assert normalize("hello") == "hello"
    assert normalize("WORLD") == "world"
    assert normalize("Hello World") == "hello world"


def test_accents_and_special_chars():
    """Prueba con acentos y caracteres especiales."""
    assert normalize("áéíóú") == "aeiou"
    assert normalize("ñÑ") == "nn"
    assert normalize("café") == "cafe"
    assert normalize("über") == "uber"
    assert normalize("naïve") == "naive"


def test_non_alphanumeric():
    """Prueba con caracteres no alfanuméricos."""
    assert normalize("hello!!!") == "hello"
    assert normalize("what's up?") == "what s up"
    assert normalize("test@123#") == "test 123"
    assert normalize("no-way_man") == "no way man"


def test_extra_spaces():
    """Prueba con espacios múltiples y en los bordes."""
    assert normalize("  hello   world  ") == "hello world"
    assert normalize("space    between") == "space between"
    assert normalize("   ") == ""


def test_mixed_difficult_cases():
    """Prueba con entradas complejas combinando todo."""
    assert normalize("Héllö Wôrld!!!") == "hello world"
    assert normalize("¡Árbol-es_123!") == "arbol es 123"
    assert normalize("  Ção ñõ  ") == "cao no"
    assert normalize("MÁS  café...") == "mas cafe"


def test_empty_and_edge_cases():
    """Prueba con casos vacíos o extremos."""
    assert normalize("") == ""
    assert normalize("!!!???") == ""
    assert normalize("  \t\n  ") == ""


def test_unicode_tricky():
    """Prueba con caracteres Unicode más raros."""
    assert normalize("Ångström") == "angstrom"
    assert normalize("mañana★") == "manana"
    assert normalize("こんにちは") == ""


def test_uppercase_with_specials():
    """Prueba con mayúsculas y caracteres especiales."""
    assert normalize("ÄBC123!!!") == "abc123"
    assert normalize("ÑOÑO EL NIÑO") == "nono el nino"
    assert normalize("CÓDIGO-EXTRAÑO") == "codigo extrano"
