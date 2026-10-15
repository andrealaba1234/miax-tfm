"""Escritura del panel limpio, del registro de decisiones y del informe."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .integridad import sha256_archivo


def sustituir_filas(tabla, filas, valores, columnas):
    """Sustituye en ``tabla`` (pyarrow) los valores de ``columnas`` solo en las posiciones ``filas``.

    ``valores`` debe estar ordenado como ``filas``. El resto de valores (incluidos nulos) no cambia.
    """
    mascara = np.zeros(tabla.num_rows, dtype=bool)
    mascara[np.asarray(filas)] = True
    for col in columnas:
        i = tabla.schema.get_field_index(col)
        original = tabla[col].combine_chunks()
        nueva = pc.replace_with_mask(original, pa.array(mascara),
                                     pa.array(np.asarray(valores[col]), type=original.type))
        tabla = tabla.set_column(i, col, nueva)
    return tabla


def anadir_columnas(tabla, df, columnas):
    """Añade columnas de ``df`` (alineado fila a fila con ``tabla``) al final de la tabla."""
    for col in columnas:
        serie = df[col]
        if pd.api.types.is_datetime64_any_dtype(serie):
            serie = serie.dt.date
        tabla = tabla.append_column(col, pa.Array.from_pandas(serie))
    return tabla


def escribir_parquet(tabla, ruta, columnas_nuevas, params):
    """Escribe el panel con la descripción de las columnas nuevas y los parámetros en los metadatos."""
    meta = {**(tabla.schema.metadata or {}),
            b'limpieza_columnas_nuevas': json.dumps(columnas_nuevas, ensure_ascii=False).encode(),
            b'limpieza_parametros': json.dumps(params, default=str).encode()}
    pq.write_table(tabla.replace_schema_metadata(meta), ruta, compression='zstd')


def construir_log(panel, decisiones, simbolos, columnas=('membership_period', 'tipo_hueco', 'motivo_ret', 'ret_tr_limpio')):
    """Una fila por (fila del panel, decisión aplicada). ``decisiones``: {nombre: máscara booleana}."""
    partes = [panel.loc[m, ['date', 'assetid', *columnas]].assign(decision=d) for d, m in decisiones.items() if m.any()]
    log = pd.concat(partes, ignore_index=True)
    log.insert(2, 'simbolo', log['assetid'].map(simbolos))
    return log


def resumen_decisiones(panel, decisiones):
    """Filas, % del panel y valores afectados por cada decisión."""
    return pd.DataFrame([{'decision': d, 'filas': int(m.sum()), 'pct_panel': 100 * m.mean(),
                          'valores': panel.loc[m, 'assetid'].nunique()}
                         for d, m in decisiones.items()]).set_index('decision')


def guardar_informe(informe, directorio, nombre='informe_limpieza.json'):
    """Añade los hashes de los parquet de salida y guarda el informe en JSON."""
    informe = {**informe, 'salida': {p.name: sha256_archivo(p) for p in sorted(directorio.glob('*.parquet'))}}
    (directorio / nombre).write_text(json.dumps(informe, indent=2, ensure_ascii=False, default=str))
    return informe
