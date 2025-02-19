import urllib.parse
from typing import Optional


class DatabaseConfig:
    """Configuración de la base de datos."""

    def __init__(
        self,
        driver: str = "SQL Server",
        server: str = "DESKTOP-IRH7NFN",
        database: str = "Identificaciones",
        trusted_connection: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.driver = driver
        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection
        self.username = username
        self.password = password

    def get_connection_string(self) -> str:
        """Genera la cadena de conexión."""
        params = {
            "DRIVER": f"{{{self.driver}}}",
            "SERVER": self.server,
            "DATABASE": self.database,
        }

        if self.trusted_connection:
            params["Trusted_Connection"] = "yes"
        elif self.username and self.password:
            params["UID"] = self.username
            params["PWD"] = self.password

        return urllib.parse.quote_plus(
            ";".join(f"{key}={value}" for key, value in params.items())
        )
