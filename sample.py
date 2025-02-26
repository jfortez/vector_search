from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from util import normalize
from search.lists import EmbeddingModelType
import numpy as np

data = [
    {
        "CredencialId": 11429,
        "Identificacion": "0100036300",
        "NombreCompleto": "CARLOS ENRIQUE TORRES MARTINEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11430,
        "Identificacion": "0100236793",
        "NombreCompleto": "MARIA CRISTINA ARIAS PAREDES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11431,
        "Identificacion": "0100275502",
        "NombreCompleto": "JOSE FERNANDO GARCIA VELEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11432,
        "Identificacion": "0100290261",
        "NombreCompleto": "MANUEL JESUS MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11433,
        "Identificacion": "0100550607",
        "NombreCompleto": "ROSA ELVIRA GUAMAN DURAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11434,
        "Identificacion": "0100767177",
        "NombreCompleto": "CARLOS ALFONSO LOPEZ IDROVO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11435,
        "Identificacion": "0100961457",
        "NombreCompleto": "MANUEL VICTOR MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11436,
        "Identificacion": "0101056091",
        "NombreCompleto": "LUIS ALBERTO MARTINEZ LASSO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11437,
        "Identificacion": "0101083244",
        "NombreCompleto": "LUIS ALBERTO ZAMBRANO SARMIENTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11438,
        "Identificacion": "0101139921",
        "NombreCompleto": "MIGUEL ANGEL VELEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11439,
        "Identificacion": "0101173078",
        "NombreCompleto": "ROSA NORMA LOPEZ LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11440,
        "Identificacion": "0101179273",
        "NombreCompleto": "MANUEL CABRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11441,
        "Identificacion": "0101196038",
        "NombreCompleto": "LUIS JAIME GUAMAN NAULA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11442,
        "Identificacion": "0101208122",
        "NombreCompleto": "CARLOS HUMBERTO LOPEZ SARMIENTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11443,
        "Identificacion": "0101218410",
        "NombreCompleto": "JUAN CARLOS GONZALEZ VINTIMILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11444,
        "Identificacion": "0101222503",
        "NombreCompleto": "JORGE EDISON GARCIA PARRA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11445,
        "Identificacion": "0101285674",
        "NombreCompleto": "LUIS ALBERTO LEMA ORTIZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11446,
        "Identificacion": "0101316008",
        "NombreCompleto": "LUIS ALBERTO PALACIOS RIVERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11447,
        "Identificacion": "0101386043",
        "NombreCompleto": "LUIS ARIOLFO SANCHEZ FLORES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11448,
        "Identificacion": "0101409332",
        "NombreCompleto": "MARIA ROSARIO VIVAR CAMPOVERDE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11449,
        "Identificacion": "0101685394",
        "NombreCompleto": "JOSE HERNANDO VELEZ AUCAPIÑA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11450,
        "Identificacion": "0101715597",
        "NombreCompleto": "JORGE EUGENIO LOPEZ JARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11451,
        "Identificacion": "0101891109",
        "NombreCompleto": "LILIA DEL CARMEN CABRERA ROJAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11452,
        "Identificacion": "0101941631",
        "NombreCompleto": "LUIS PATRICIO AREVALO BARRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11453,
        "Identificacion": "0101971018",
        "NombreCompleto": "JUAN CARLOS LOPEZ LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11454,
        "Identificacion": "0102030749",
        "NombreCompleto": "ROSA MARIA MOSQUERA MOSQUERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11455,
        "Identificacion": "0102041076",
        "NombreCompleto": "VICTOR IVAN CABRERA CABRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11456,
        "Identificacion": "0102155025",
        "NombreCompleto": "JOSE RICARDO SERRANO SALGADO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11457,
        "Identificacion": "0102176526",
        "NombreCompleto": "LUIS ALBERTO MENDEZ VINTIMILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11458,
        "Identificacion": "0102206539",
        "NombreCompleto": "DIEGO IVAN GARCIA CARDENAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11459,
        "Identificacion": "0102252517",
        "NombreCompleto": "LUIS ARTURO GUAMAN GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11460,
        "Identificacion": "0102353034",
        "NombreCompleto": "CARLOS RODRIGO MENDEZ PEREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11461,
        "Identificacion": "0102378395",
        "NombreCompleto": "FREDDY HERNAN ESPINOZA CALLE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11462,
        "Identificacion": "0102383981",
        "NombreCompleto": "MANUEL ALEJANDRO MOROCHO MOROCHO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11463,
        "Identificacion": "0102503216",
        "NombreCompleto": "DIEGO FERNANDO ANDRADE TORRES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11464,
        "Identificacion": "0102581212",
        "NombreCompleto": "JUAN CARLOS LOPEZ PATIÑO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11465,
        "Identificacion": "0102651064",
        "NombreCompleto": "FATIMA IRENE SANCHEZ TAPIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11466,
        "Identificacion": "0102676434",
        "NombreCompleto": "JUAN CARLOS LOPEZ QUIZHPI",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11467,
        "Identificacion": "0102708492",
        "NombreCompleto": "MONICA ESPERANZA GONZALEZ GONZALEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11468,
        "Identificacion": "0102708492-",
        "NombreCompleto": "MONICA ESPERANZA GONZALEZ GONZALEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11469,
        "Identificacion": "0102710019",
        "NombreCompleto": "CARLOS ALBERTO ROJAS PACURUCU",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11470,
        "Identificacion": "0102814548",
        "NombreCompleto": "ESTEBAN LEONARDO CALLE CALLE",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11471,
        "Identificacion": "0102946993",
        "NombreCompleto": "LUIS ENRIQUE GUAMAN SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11472,
        "Identificacion": "0103020855",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11473,
        "Identificacion": "0103488896",
        "NombreCompleto": "CARLOS ARMANDO PEREZ BRITO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11474,
        "Identificacion": "0103747093",
        "NombreCompleto": "JORGE EDUARDO CABRERA PEREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11475,
        "Identificacion": "0103830527",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ CARPIO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11476,
        "Identificacion": "0103923546",
        "NombreCompleto": "MARCO ANTONIO ANDRADE FLORES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11477,
        "Identificacion": "0103975645",
        "NombreCompleto": "JOSE RIGOBERTO CABRERA LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11478,
        "Identificacion": "0104001110",
        "NombreCompleto": "JOSE ANTONIO CALDERON BARRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11479,
        "Identificacion": "0104028527",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ UZHCA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11480,
        "Identificacion": "0104031265",
        "NombreCompleto": "CARLOS FERNANDO GOMEZ CRESPO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11481,
        "Identificacion": "0104164363",
        "NombreCompleto": "JORGE LUIS GARCIA TAPIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11482,
        "Identificacion": "0104200803",
        "NombreCompleto": "JORGE ENRIQUE VARGAS SICHA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11483,
        "Identificacion": "0104279518",
        "NombreCompleto": "ANA LUCIA GUAMAN GUERRERO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11484,
        "Identificacion": "0104545686",
        "NombreCompleto": "CARLOS ALBERTO LOPEZ VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11485,
        "Identificacion": "0104545686",
        "NombreCompleto": "LOPEZ VERA LOPEZ VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11486,
        "Identificacion": "0104627930",
        "NombreCompleto": "MARIA HORTENCIA JIMENEZ JIMENEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11487,
        "Identificacion": "0104628656",
        "NombreCompleto": "JUAN JOSE DELGADO ORAMAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11488,
        "Identificacion": "0104648282",
        "NombreCompleto": "JOSE VICENTE ZAMBRANO TENESACA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11489,
        "Identificacion": "0104968367",
        "NombreCompleto": "JOSE LUIS CABRERA LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11490,
        "Identificacion": "0105090807",
        "NombreCompleto": "JOSE LORENZO SALINAS SALINAS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11491,
        "Identificacion": "0105094163",
        "NombreCompleto": "MARIA ELISABETH RAMOS BRITO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11492,
        "Identificacion": "0105362222",
        "NombreCompleto": "JAVIER EDUARDO SANCHEZ SAMANIEGO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11493,
        "Identificacion": "0105376313",
        "NombreCompleto": "MIGUEL ANGEL RODRIGUEZ RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11494,
        "Identificacion": "0150609428",
        "NombreCompleto": "JOSE LUIS ROBLES JARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11495,
        "Identificacion": "0200049401",
        "NombreCompleto": "LUIS ALFONSO RAMIREZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11496,
        "Identificacion": "0200267862",
        "NombreCompleto": "LUIS ALBERTO GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11497,
        "Identificacion": "0200386837",
        "NombreCompleto": "FLOR MARIA GARCIA CARRILLO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11498,
        "Identificacion": "0200408938",
        "NombreCompleto": "JORGE ALBERTO GARCIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11499,
        "Identificacion": "0200426831",
        "NombreCompleto": "LUIS ALBERTO LARA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11500,
        "Identificacion": "0200437721",
        "NombreCompleto": "LUIS ALFREDO RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11501,
        "Identificacion": "0200510428",
        "NombreCompleto": "SAUL WASHINGTON ESCOBAR ORELLANA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11502,
        "Identificacion": "0200575470",
        "NombreCompleto": "LUIS ALBERTO RODRIGUEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11503,
        "Identificacion": "0200642759",
        "NombreCompleto": "LUIS VICENTE SOLANO ANGULO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11504,
        "Identificacion": "0200671154",
        "NombreCompleto": "PEDRO ALBERTO VERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11505,
        "Identificacion": "0200680916",
        "NombreCompleto": "MIGUEL ANGEL SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11506,
        "Identificacion": "0200764603",
        "NombreCompleto": "CARLOS ALBERTO ZAPATA SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11507,
        "Identificacion": "0200785806",
        "NombreCompleto": "LAURA BEATRIZ GUAMAN",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11508,
        "Identificacion": "0200788768",
        "NombreCompleto": "LUIS BOLIVAR VERGARA ORTIZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11509,
        "Identificacion": "0200944296",
        "NombreCompleto": "ANGEL VICENTE HURTADO REA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11510,
        "Identificacion": "0200947141",
        "NombreCompleto": "JORGE LUIS RODRIGUEZ VASCONEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11511,
        "Identificacion": "0200959179",
        "NombreCompleto": "JOSE SALVADOR JIMENEZ MODUMBA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11512,
        "Identificacion": "0200972636",
        "NombreCompleto": "ANGEL OSWALDO BONILLA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11513,
        "Identificacion": "0200980290",
        "NombreCompleto": "JUAN CARLOS GONZALEZ LOMBEIDA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11514,
        "Identificacion": "0200980308",
        "NombreCompleto": "GUILLERMO EDUARDO RODRIGUEZ PAREDES",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11515,
        "Identificacion": "0201067956",
        "NombreCompleto": "MARIA LUCILA RAMOS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11516,
        "Identificacion": "0201155264",
        "NombreCompleto": "CARLOS ALBERTO ZAPATA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11517,
        "Identificacion": "0201161072",
        "NombreCompleto": "LUIS ENRIQUE SALAZAR LOPEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11518,
        "Identificacion": "0201164712",
        "NombreCompleto": "VICTOR HUGO MENDOZA VERDESOTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11519,
        "Identificacion": "0201216694",
        "NombreCompleto": "FRANKLIN EDUARDO GARCIA GALLEGOS",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11520,
        "Identificacion": "0201229440",
        "NombreCompleto": "MARIA TRANSITO HERRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11521,
        "Identificacion": "0201317443",
        "NombreCompleto": "LUIS ALBERTO SANCHEZ VELASCO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11522,
        "Identificacion": "0201563384",
        "NombreCompleto": "MARCO ANTONIO GARCIA GARCIA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11523,
        "Identificacion": "0201576949",
        "NombreCompleto": "JUAN CARLOS MUÑOZ SANCHEZ",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11524,
        "Identificacion": "0201605607",
        "NombreCompleto": "JOSE LUIS SANCHEZ MOYA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11525,
        "Identificacion": "0201607256",
        "NombreCompleto": "JOSE LUIS RUIZ SOTO",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11526,
        "Identificacion": "0201610292",
        "NombreCompleto": "WALTER HERNAN GRANIZO HERRERA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11527,
        "Identificacion": "0201644994",
        "NombreCompleto": "JOSE LUIS SALAZAR CAJILEMA",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
    {
        "CredencialId": 11528,
        "Identificacion": "0201688686",
        "NombreCompleto": "JUAN CARLOS BRAVO GAIBOR",
        "FuenteId": 100,
        "CargaId": 101,
        "FechaCarga": "2022-08-31T15:11:15.263Z",
    },
]


if __name__ == "__main__":

    # Función para normalizar embeddings
    def normalize_embedding(emb):
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # Configuración del modelo
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    d = model.get_sentence_embedding_dimension()  # 384 dimensiones

    # Preparar datos para embeddings
    name_sentences = [normalize(item["NombreCompleto"]) for item in data]
    id_sentences = [
        item["Identificacion"] if item["Identificacion"] else "" for item in data
    ]

    # Generar embeddings
    name_embeddings = model.encode(name_sentences, show_progress_bar=True).astype(
        "float32"
    )
    id_embeddings = model.encode(id_sentences, show_progress_bar=True).astype("float32")

    # Normalizar embeddings
    name_embeddings_normalized = np.array(
        [normalize_embedding(emb) for emb in name_embeddings]
    )
    id_embeddings_normalized = np.array(
        [normalize_embedding(emb) for emb in id_embeddings]
    )

    # Combinar embeddings en un solo vector
    combined_embeddings = np.hstack(
        [name_embeddings_normalized, id_embeddings_normalized]
    )

    # Crear índice FAISS con producto interno
    index = faiss.IndexFlatIP(2 * d)
    index.add(combined_embeddings)

    # Función para preparar la consulta
    def prepare_query_embedding(query):
        if query.isdigit():  # Búsqueda por ID
            query_id_embedding = model.encode([query], show_progress_bar=False).astype(
                "float32"
            )[0]
            query_id_embedding_normalized = normalize_embedding(query_id_embedding)
            return np.hstack(
                [np.zeros(d, dtype="float32"), query_id_embedding_normalized]
            )
        else:  # Búsqueda por nombre o combinación
            query_name = normalize(query)
            query_name_embedding = model.encode(
                [query_name], show_progress_bar=False
            ).astype("float32")[0]
            query_name_embedding_normalized = normalize_embedding(query_name_embedding)
            return np.hstack(
                [query_name_embedding_normalized, np.zeros(d, dtype="float32")]
            )

    # Función para buscar y mostrar resultados
    def search(query, top_k=5):
        query_embedding = prepare_query_embedding(query)
        distances, indices = index.search(np.array([query_embedding]), top_k)
        print(distances[0])
        print(indices[0])
        results = []
        max_d = max(distances[0].max(), 1e-10)
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = f"{name_sentences[idx]} {id_sentences[idx]}"
            similarity = 1 - (score / max_d) if max_d > 0 else 0
            results.append(
                {
                    "result": result,
                    "idx": idx,
                    "score": score,
                    "similarity": similarity,
                    "distance": f"{distances[0][0]:.2f}",
                }
            )
        return pd.DataFrame(results)

    # Pruebas
    test_queries = ["0201688686", "juan carlos", "torres 36300", "juan crlos"]
    for query in test_queries:
        df = search(query)
        print(f"\nquery: {query}")
        print(df)
