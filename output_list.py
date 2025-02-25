from search.lists import EmbeddingManager, EmbeddingModelType
from tqdm import tqdm
import time
from pathlib import Path


def main():
    t = time.time()
    inputs = [
        # "Burning Giraffe",
        # "Bi-colored Agate Box, Agate and Sapphire",
        # "Bead Necklace",
        # "On Autumn Wind",
        # "Stacy Gitlin",
        # "MR. NGO DUNG TOAN",
        # "BAYON WATER PUMP CO., LTD.",
        # "CONSTRUTORA OAS GUINEE S.A.",
        # "HEBEI CONSTRUCTION GROUP CORPORATION LIMITED GUANGXI BRANCH",
        # "SUPACHAI PRECHATERASAT",
        # "ADONIS CORREA",
        # "THIBAULT SABOURIN",
        # "FERDINAND NDIKURIYO",
        # "ABDEL RAHMAN JUMA",
        # "HAJI BASIR AND ZARJMIL COMPANY HAWALA",
        # "daniel noboa",
        # "aquiles alvarez",
        # "pablo escobar",
        # "nicolas maduro",
        # "cesar burbano",
        "burbano paul",
        "paul burbano",
        # "burbano cesar",
        # "alvarez",
        # "antonio alvarez",
        "burbano paul cristian",
        # "nicolas maduro moros",
        # "milei javier",
    ]

    model_type = EmbeddingModelType.PARAPHRASE
    # Crear una instancia de EmbeddingManager
    embedding_manager = EmbeddingManager(model_type=model_type)
    path = Path("embeddings") / embedding_manager.model_name / "output.txt"

    # Definir parámetros de búsqueda por defecto
    threshold = 0.1
    k = 20

    # Lista para almacenar los bloques de texto
    blocks = []

    # Generar el contenido de cada bloque con barra de progreso
    print("🔍 Procesando búsquedas...")
    for query in tqdm(inputs, desc="Buscando y generando bloques", unit="query"):
        # Realizar la búsqueda
        results = embedding_manager.search(query, threshold=threshold, k=k)

        # Construir el bloque de texto
        block_lines = []
        block_lines.append(f"- INPUT: [{query}]")
        block_lines.append(f"- ARGS: (threshold={threshold}, k={k})")

        if results.empty:
            block_lines.append("[DATAFRAME]")
            block_lines.append("No se encontraron resultados relevantes")
        else:
            df_string = results.to_string(index=False).split("\n")
            block_lines.append("[DATAFRAME]")
            block_lines.extend(df_string)

        blocks.append(block_lines)

    # Calcular el ancho máximo de todas las líneas en todos los bloques
    max_width = max(len(line) for block in blocks for line in block)

    # Crear la línea separadora con el ancho máximo
    separator = "-" * max_width

    # Escribir todo en el archivo
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"MODEL NAME: {model_type.value} | Shape: {embedding_manager.embeddings.shape}\n\n"
        )
        for i, block in enumerate(blocks):
            # Escribir el bloque
            for line in block:
                f.write(f"{line}\n")
            # Agregar la línea separadora (excepto después del último bloque)
            if i < len(blocks) - 1:
                f.write(f"{separator}\n\n")

    print("✅ Resultados escritos en output_list.txt")
    print(f"⏱ Tiempo total: {time.time() - t:.2f} segundos")


if __name__ == "__main__":
    main()
