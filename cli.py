import argparse
import logging
import os
from pathlib import Path
from enum import Enum
from tqdm import tqdm
import time
import json
import pandas as pd
from search.lists import EmbeddingManager, EmbeddingModelType, FaissIndexType


# Diccionario de valores por defecto
DEFAULTS = {
    "models": ["MINI_LM"],
    "indexes": ["FLAT_L2"],
    "inputs": [],
    "output": "embeddings",
    "threshold": 0.1,
    "k": 20,
    "format": "txt",
    "log": None,
    "verbose": False,
}


class CLIConfig:
    """Clase para gestionar la configuración de la CLI con argumentos secuenciales usando argparse."""

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self):
        """Crea el parser de argumentos con estructura secuencial usando DEFAULTS."""
        parser = argparse.ArgumentParser(
            description="Script CLI para búsquedas con embeddings de forma secuencial."
        )

        # Grupo para --models (obligatorio y primero, a menos que exista config.json)
        group_models = parser.add_argument_group(
            "Modelos (obligatorio y primero, a menos que exista config.json)"
        )
        group_models.add_argument(
            "--models",
            nargs="+",
            choices=[e.name for e in EmbeddingModelType],
            required=False,  # No es obligatorio si existe config.json
            help="Modelos de embeddings a usar (e.g., MINI_LM PARAPHRASE). Debe ser el primer argumento si se usa desde la CLI.",
        )

        # Grupo para los argumentos que dependen de --models
        group_dependent = parser.add_argument_group(
            "Argumentos dependientes (requieren --models o config.json)"
        )
        group_dependent.add_argument(
            "--indexes",
            nargs="*",
            choices=[e.name for e in FaissIndexType],
            default=None,
            help="Tipos de índices FAISS (e.g., FLAT_L2 IVFPQ). Opcional, default: FLAT_L2 si se usa config.json.",
        )
        group_dependent.add_argument(
            "--inputs",
            nargs="*",
            default=None,
            help="Entradas para búsqueda (e.g., 'paul burbano') o archivo .txt (e.g., inputs.txt). Opcional.",
        )
        group_dependent.add_argument(
            "--output",
            default=None,
            help="Directorio donde se guardarán los resultados (default: 'embeddings' si se usa config.json). Opcional.",
        )
        group_dependent.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Umbral de similitud para la búsqueda (default: 0.1 si se usa config.json). Opcional.",
        )
        group_dependent.add_argument(
            "--k",
            type=int,
            default=None,
            help="Número máximo de resultados por búsqueda (default: 20 si se usa config.json). Opcional.",
        )
        group_dependent.add_argument(
            "--format",
            choices=["txt", "json", "csv"],
            default=None,
            help="Formato de salida (default: txt si se usa config.json). Opcional.",
        )

        # Argumentos independientes (opcionales en cualquier momento)
        parser.add_argument(
            "--log",
            default=None,
            help="Archivo de log (opcional).",
        )
        parser.add_argument(
            "--config",
            default=None,
            help="Archivo de configuración JSON (opcional, e.g., config.json).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Mostrar todos los mensajes de logging en la consola (opcional).",
        )

        return parser

    def parse_args(self):
        """Analiza los argumentos y valida la secuencia, cargando config.json si aplica."""
        args = self.parser.parse_args()

        # Validar que --models sea el primer argumento si se usa desde la CLI y no hay config.json
        if args.models is not None:
            # Verificar que --models sea el primer argumento si se usa desde la CLI
            cli_args = " ".join(os.sys.argv[1:])
            if "--models" not in cli_args.split()[0] and not (
                args.config or os.path.exists("config.json")
            ):
                self.parser.error(
                    "El argumento --models debe ser el primero si se usa desde la CLI y no existe config.json."
                )

        # Si no se especifica --models, intentar cargar config.json
        if args.models is None:
            if args.config and os.path.exists(args.config):
                args = self._load_config_from_file(args, args.config)
            elif os.path.exists("config.json"):
                args = self._load_config_from_file(args, "config.json")
            else:
                self.parser.error(
                    "Se requiere --models o un archivo config.json existente."
                )

        # Priorizar argumentos de CLI sobre config.json
        if args.models is not None:
            args = self._override_with_cli_args(args)

        # Aplicar valores por defecto desde DEFAULTS
        for key, value in DEFAULTS.items():
            setattr(args, key, getattr(args, key) or value)

        # Validar que la longitud de inputs no sea 0
        inputs = load_inputs(args.inputs)
        if not inputs:
            self.parser.error(
                "No se proporcionaron entradas válidas. Se requiere al menos una entrada."
            )

        # Configurar logging con base en --verbose y --log
        self._setup_logging(args)

        self.args = args
        return args

    def _load_config_from_file(self, args, config_file):
        """Carga configuración desde un archivo JSON especificado, usando DEFAULTS como respaldo."""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                for key in DEFAULTS:
                    value = getattr(args, key) or config.get(key, DEFAULTS[key])
                    setattr(args, key, value)
            logging.info(f"Cargando configuración desde {config_file}.")
        except Exception as e:
            logging.error(f"Error al cargar {config_file}: {e}")
        return args

    def _override_with_cli_args(self, args):
        """Prioriza los argumentos de la CLI sobre config.json y DEFAULTS."""
        if args.models is not None:
            args.models = [
                m for m in args.models if m in [e.name for e in EmbeddingModelType]
            ] or DEFAULTS["models"]
        if args.indexes is not None:
            args.indexes = [
                i for i in args.indexes if i in [e.name for e in FaissIndexType]
            ] or DEFAULTS["indexes"]
        if args.inputs is not None:
            args.inputs = [i for i in args.inputs if i] or DEFAULTS["inputs"]
        return args

    def _setup_logging(self, args):
        """Configura el logging de manera robusta con soporte para --verbose."""
        # Configurar un logger específico para este script
        logger = logging.getLogger("search_script")
        logger.setLevel(logging.INFO)

        # Limpiar handlers existentes para evitar duplicados
        if logger.hasHandlers():
            logger.handlers.clear()

        # Configurar handler para archivo si se especifica --log
        if args.log:
            try:
                # Crear directorio para el log si no existe
                log_dir = Path(args.log).parent
                log_dir.mkdir(parents=True, exist_ok=True)

                # Verificar si el archivo es accesible y writable
                with open(args.log, "a") as f:
                    pass  # Solo verificar acceso

                # Configurar handler para archivo
                file_handler = logging.FileHandler(args.log, mode="w", encoding="utf-8")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                logger.addHandler(file_handler)

                # Log de prueba para verificar archivo
                logger.info(
                    "Configuración de logging validada para archivo: %s", args.log
                )
            except PermissionError:
                logger.error(
                    "No se puede escribir en el archivo de log: %s. Usando solo consola.",
                    args.log,
                )
            except Exception as e:
                logger.error("Error al configurar logging para archivo: %s", str(e))

        # Configurar handler para consola según --verbose
        console_level = logging.DEBUG if args.verbose else logging.WARNING
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

        # Reasignar el logger global para que todas las llamadas a logging usen este logger
        logging.getLogger("").handlers = logger.handlers
        logging.getLogger("").level = logger.level


class OutputFormatter:
    """Clase para formatear y escribir resultados en diferentes formatos."""

    @staticmethod
    def generate_block(query, results, threshold, k, format):
        """Genera un bloque de datos para el formato especificado, reduciendo repetición."""
        if format == "txt":
            block_lines = [
                f"- INPUT: [{query}]",
                f"- ARGS: (threshold={threshold}, k={k})",
            ]
            if results.empty:
                block_lines.extend(
                    ["[DATAFRAME]", "No se encontraron resultados relevantes"]
                )
            else:
                df_string = (
                    pd.DataFrame(results.to_dict("records"))
                    .to_string(index=False)
                    .split("\n")
                )
                block_lines.append("[DATAFRAME]")
                block_lines.extend(df_string)
            return block_lines
        elif format == "json":
            if results.empty:
                return {
                    "input": query,
                    "k": k,
                    "threshold": threshold,
                    "results": "No se encontraron resultados relevantes",
                }
            return {
                "input": query,
                "k": k,
                "threshold": threshold,
                "results": results.to_dict("records"),
            }
        elif format == "csv":
            if results.empty:
                return pd.DataFrame(
                    {
                        "input": [query],
                        "k": [k],
                        "threshold": [threshold],
                        "results": ["No se encontraron resultados relevantes"],
                    }
                ), ["input", "k", "threshold", "results"]
            result_df = pd.DataFrame(results.to_dict("records"))
            result_df["input"] = query
            result_df["k"] = k
            result_df["threshold"] = threshold
            columns = ["input", "k", "threshold"] + [
                col
                for col in result_df.columns
                if col not in ["input", "k", "threshold"]
            ]
            return result_df[columns], columns

    @staticmethod
    def write_output(
        path, blocks, output_format, model_type, index_type, embeddings_shape
    ):
        """Escribe los resultados en el formato especificado con sufijo adecuado."""
        if output_format == "txt":
            path = path.with_suffix(".txt")
            max_width = max(len(line) for block in blocks for line in block)
            separator = "-" * max_width
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"MODEL NAME: {model_type.value} | FAISS Index Model: {index_type.value} | Shape: {embeddings_shape}\n\n"
                )
                for i, block in enumerate(blocks):
                    for line in block:
                        f.write(f"{line}\n")
                    if i < len(blocks) - 1:
                        f.write(f"{separator}\n\n")
        elif output_format == "json":
            path = path.with_suffix(".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": model_type.value,
                        "index_type": index_type.value,
                        "shape": embeddings_shape,
                        "results": blocks,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        elif output_format == "csv":
            path = path.with_suffix(".csv")
            all_data = pd.DataFrame()
            column_order = None
            for block in blocks:
                df, cols = block if isinstance(block, tuple) else (block, None)
                if column_order is None and cols:
                    column_order = cols
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)

            if not all_data.empty and column_order:
                all_data = all_data[column_order]
                all_data.to_csv(path, index=False)
                logging.info(f"DataFrame escrito en {path}")
            else:
                logging.warning("No hay DataFrame válido para escribir en CSV.")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("No se encontraron resultados relevantes")


class SearchProcessor:
    """Clase para procesar las búsquedas y gestionar la ejecución."""

    def __init__(self, args):
        self.model_types = [EmbeddingModelType[m] for m in args.models]
        self.index_types = [
            FaissIndexType[i] for i in (args.indexes or DEFAULTS["indexes"])
        ]
        self.inputs = load_inputs(args.inputs or DEFAULTS["inputs"])
        self.output_dir = Path(args.output or DEFAULTS["output"])
        self.threshold = args.threshold or DEFAULTS["threshold"]
        self.k = args.k or DEFAULTS["k"]
        self.format = args.format or DEFAULTS["format"]
        self.formatter = OutputFormatter()

    def process_search(self):
        """Procesa las búsquedas para cada combinación de modelo e índice."""
        for model_type in self.model_types:
            for index_type in self.index_types:
                self._process_combination(model_type, index_type)

    def _process_combination(self, model_type, index_type):
        """Procesa una combinación específica de modelo e índice."""
        start_time = time.time()
        logging.info(f"Procesando modelo: {model_type.name}, índice: {index_type.name}")

        embedding_manager = EmbeddingManager(
            model_type=model_type, index_type=index_type
        )
        path = (
            self.output_dir
            / embedding_manager.model_name
            / f"output_{index_type.value}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        blocks = []
        for query in tqdm(
            self.inputs, desc=f"Búsqueda {model_type.name}-{index_type.name}"
        ):
            try:
                results = embedding_manager.search(
                    query, threshold=self.threshold, k=self.k
                )
                block = self.formatter.generate_block(
                    query, results, self.threshold, self.k, self.format
                )
                if self.format == "csv" and isinstance(block, tuple):
                    blocks.append(block)
                else:
                    blocks.append(block)
            except Exception as e:
                logging.error(f"Error procesando '{query}': {e}")
                if self.format == "txt":
                    blocks.append([f"- INPUT: [{query}]", f"- ERROR: {e}"])
                elif self.format == "json":
                    blocks.append(
                        {
                            "input": query,
                            "k": self.k,
                            "threshold": self.threshold,
                            "results": f"Error: {e}",
                        }
                    )
                elif self.format == "csv":
                    blocks.append(
                        (
                            pd.DataFrame(
                                {
                                    "input": [query],
                                    "k": [self.k],
                                    "threshold": [self.threshold],
                                    "results": [f"Error: {e}"],
                                }
                            ),
                            ["input", "k", "threshold", "results"],
                        )
                    )

        self.formatter.write_output(
            path,
            blocks,
            self.format,
            model_type,
            index_type,
            str(embedding_manager.embeddings.shape),
        )

        logging.info(f"Resultados escritos en {path}.{self.format}")
        logging.info(f"Tiempo de ejecución: {time.time() - start_time:.2f} segundos")


def load_inputs(inputs_arg):
    """Carga entradas desde argumentos o archivos, validando y eliminando duplicados."""
    if not inputs_arg:
        return []

    inputs = []
    for item in inputs_arg:
        if item.endswith(".txt"):
            try:
                with open(item, "r", encoding="utf-8") as f:
                    file_inputs = [line.strip() for line in f if line.strip()]
                    if not file_inputs:
                        logging.warning(f"El archivo {item} está vacío.")
                    inputs.extend(file_inputs)
            except FileNotFoundError:
                logging.error(f"No se encontró el archivo {item}.")
            except Exception as e:
                logging.error(f"Error al leer {item}: {e}")
        elif item.strip():
            inputs.append(item)
        else:
            logging.info("Entrada vacía ignorada.")

    unique_inputs = list(dict.fromkeys(inputs))
    if len(unique_inputs) < len(inputs):
        logging.info(
            f"Se eliminaron {len(inputs) - len(unique_inputs)} entradas duplicadas."
        )

    return unique_inputs


def main():
    """Punto de entrada principal para ejecutar el script."""
    config = CLIConfig()
    args = config.parse_args()

    processor = SearchProcessor(args)
    processor.process_search()


if __name__ == "__main__":
    main()
