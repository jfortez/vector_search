import pandas as pd
from sqlalchemy import create_engine
import urllib

# Conectar con SQL Server
def get_engine():
    params = urllib.parse.quote_plus(
        "DRIVER={SQL Server};"
        "SERVER=DESKTOP-IRH7NFN;"
        "DATABASE=Identificaciones;"
        "Trusted_Connection=yes;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    return engine




# Obtener datos de la base de datos
def get_data():
    engine = get_engine()
    query = "SELECT id, identificacion, nombre FROM Personas;"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df
