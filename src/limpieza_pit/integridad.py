"""Verificación de integridad de los datos de entrada."""

import hashlib
import json

import pandas as pd
import pyarrow.parquet as pq


def sha256_archivo(ruta, bloque=1 << 20):
    """SHA256 de un archivo, leído por bloques para no cargarlo en memoria."""
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for trozo in iter(lambda: f.read(bloque), b''):
            h.update(trozo)
    return h.hexdigest()


def verificar_sha256sums(directorio, lista='SHA256SUMS.txt'):
    """Compara cada archivo con su huella en ``lista`` y detecta archivos no listados."""
    esperado = pd.read_csv(directorio / lista, sep=r'\s+', header=None,
                           names=['sha256_esperado', 'archivo'], engine='python')
    esperado['sha256_calculado'] = [sha256_archivo(directorio / a) for a in esperado['archivo']]
    esperado['ok'] = esperado['sha256_esperado'] == esperado['sha256_calculado']
    presentes = {p.relative_to(directorio).as_posix() for p in directorio.rglob('*')
                 if p.is_file() and p.name != '.DS_Store'} - {lista}
    no_listados = sorted(presentes - set(esperado['archivo']))
    return esperado, no_listados


def verificar_hash(ruta, esperado):
    """Comprueba que un archivo sin SHA256SUMS conserva el hash registrado al empezar el TFM."""
    calculado = sha256_archivo(ruta)
    assert calculado == esperado, f'{ruta.name} ha cambiado: {calculado} != {esperado}'
    return calculado


def verificar_esquemas(directorio, diccionario='diccionario_esquemas.json'):
    """Compara nombres, tipos y nº de filas de cada parquet con el diccionario de esquemas."""
    filas = []
    for tabla in json.loads((directorio / diccionario).read_text()):
        ruta = directorio / tabla['file']
        real = [(c.name, str(c.type)) for c in pq.read_schema(ruta)]
        esperado = [(c['name'], c['type']) for c in tabla['columns']]
        filas.append({'archivo': tabla['file'], 'filas_esperadas': tabla['rows'],
                      'filas_reales': pq.ParquetFile(ruta).metadata.num_rows,
                      'columnas': len(real), 'esquema_ok': real == esperado})
    res = pd.DataFrame(filas)
    res['filas_ok'] = res['filas_esperadas'] == res['filas_reales']
    return res
