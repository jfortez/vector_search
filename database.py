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
def get_identificacion():
    engine = get_engine()
    query = "SELECT id, identificacion, nombre FROM Personas;"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


# delete identificacion
def delete_identificacion(id):
    engine = get_engine()
    query = f"DELETE FROM Personas WHERE id = {id};"
    engine.execute(query)
    engine.dispose()


# update identificacion
def update_identificacion(id, identificacion, nombre):
    engine = get_engine()
    query = f"UPDATE Personas SET identificacion = '{identificacion}', nombre = '{nombre}' WHERE id = {id};"
    engine.execute(query)
    engine.dispose()


# insert identificacion
def insert_identificacion(identificacion, nombre):
    engine = get_engine()
    query = f"INSERT INTO Personas (identificacion, nombre) VALUES ('{identificacion}', '{nombre}');"
    engine.execute(query)
