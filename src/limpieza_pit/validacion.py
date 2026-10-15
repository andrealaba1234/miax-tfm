"""Validaciones posteriores a la limpieza. Marcan o comprueban; nunca modifican precios."""

import numpy as np
import pandas as pd

from .limpieza import aplicar_limpieza

# Columnas producidas por aplicar_limpieza que deben ser idénticas con y sin el futuro
COLUMNAS_CAUSALES = ['fecha_ultimo_precio_real', 'sesiones_desde_ultimo_precio', 'precio_imputado',
                     'close_totalreturn', 'close_none', 'tipo_hueco', 'precio_utilizable', 'excluible',
                     'ret_tr_limpio', 'ret_none_limpio', 'motivo_ret', 'ret_anomalo',
                     'accion_corporativa_ajustada', 'posible_split_no_ajustado', 'sesiones_mismo_cierre',
                     'precio_congelado']


def comprobaciones_basicas(panel, col_ret_original=None):
    """Duplicados, precios no positivos o nulos en filas utilizables y, opcionalmente,
    diferencia máxima entre ``ret_tr_limpio`` y el retorno original en las filas 'normal'."""
    util = panel['precio_utilizable']
    res = {
        'duplicados (date, assetid)': int(panel.duplicated(['date', 'assetid']).sum()),
        'close_totalreturn <= 0 (filas utilizables)': int((panel.loc[util, 'close_totalreturn'] <= 0).sum()),
        'close_none <= 0 (filas utilizables)': int((panel.loc[util, 'close_none'] <= 0).sum()),
        'precio nulo en filas utilizables': int(panel.loc[util, ['close_totalreturn', 'close_none']]
                                                .isna().any(axis=1).sum()),
    }
    if col_ret_original is not None:
        normal = panel['motivo_ret'] == 'normal'
        res['retornos "normal" sin retorno original'] = int(panel.loc[normal, col_ret_original].isna().sum())
        res['máx. |ret_tr_limpio − retorno original| (normal)'] = float(
            (panel.loc[normal, 'ret_tr_limpio'] - panel.loc[normal, col_ret_original]).abs().max())
    return pd.Series(res, name='resultado', dtype=object)


def ohlc_incoherente(ohlc, con_precio, variantes, tol):
    """Filas con precio real que incumplen ``low <= min(open, close)`` o ``high >= max(open, close)``.

    ``variantes``: lista de tuplas (open, high, low, close) con nombres de columna de ``ohlc``.
    Devuelve la máscara combinada y un resumen por variante.
    """
    total = pd.Series(False, index=ohlc.index)
    resumen = {}
    for o, h, l, c in variantes:
        mal = ((ohlc[l] > np.fmin(ohlc[o], ohlc[c]) * (1 + tol)) | (ohlc[h] < np.fmax(ohlc[o], ohlc[c]) * (1 - tol))
               | (ohlc[l] > ohlc[h] * (1 + tol))) & con_precio
        resumen[c] = {'filas_incoherentes': int(mal.sum()), 'open_nulo_con_precio': int((ohlc[o].isna() & con_precio).sum())}
        total |= mal
    return total, pd.DataFrame(resumen).T


def detectar_reversiones(panel, umbral=0.25, neto=0.05):
    """Picos que se revierten al día siguiente (diagnóstico: mira el día siguiente)."""
    r_hoy = panel['ret_tr_limpio']
    r_manana = panel.groupby('assetid')['ret_tr_limpio'].shift(-1)
    mascara = ((r_hoy.abs() > umbral) & (r_manana.abs() > umbral) & (np.sign(r_hoy) != np.sign(r_manana))
               & (((1 + r_hoy) * (1 + r_manana) - 1).abs() < neto))
    return mascara, r_manana


def prueba_lookahead(panel_base, historia, cal, params, cortes, referencia=None, evaluacion=None,
                     columnas=COLUMNAS_CAUSALES, limpieza=aplicar_limpieza):
    """Prueba de truncamiento: limpiar con datos hasta T debe dar las mismas filas <= T
    que limpiar con todo el dataset. Lanza AssertionError si alguna fila cambia.

    ``limpieza``: función (panel_base, historia, cal, params, evaluacion) -> panel limpio.
    Se puede sustituir por una limpieza trampa para el control negativo.
    """
    if referencia is None:
        referencia = limpieza(panel_base, historia, cal, params, evaluacion)
    filas = []
    for corte in cortes:
        T = pd.Timestamp(corte)
        truncado = limpieza(panel_base[panel_base['date'] <= T], historia[historia['date'] <= T], cal, params, evaluacion)
        ref = referencia.loc[referencia['date'] <= T, columnas].reset_index(drop=True)
        pd.testing.assert_frame_equal(truncado[columnas].reset_index(drop=True), ref)
        dia_T = truncado['date'] == T
        filas.append({'corte': corte, 'filas_comparadas': len(ref),
                      'imputadas_hasta_T': int(truncado['precio_imputado'].sum()),
                      'imputadas_el_dia_T': int(truncado.loc[dia_T, 'precio_imputado'].sum()),
                      'excluibles_el_dia_T': int(truncado.loc[dia_T, 'excluible'].sum()),
                      'resultado': 'idénticas ✔'})
    return pd.DataFrame(filas)


def limpieza_trampa_backfill(panel_base, historia, cal, params, evaluacion=None):
    """Limpieza con look-ahead deliberado (backfill) para el control negativo de la prueba."""
    p = aplicar_limpieza(panel_base, historia, cal, params, evaluacion)
    p['close_totalreturn'] = p.groupby('assetid')['close_totalreturn'].bfill()
    return p


def limpieza_trampa_interpolacion(panel_base, historia, cal, params, evaluacion=None):
    """Limpieza con look-ahead deliberado: interpola linealmente los precios imputados entre
    el precio anterior y el SIGUIENTE (error habitual). Sirve aunque no queden huecos sin rellenar."""
    p = aplicar_limpieza(panel_base, historia, cal, params, evaluacion)
    sin_imputar = p['close_totalreturn'].mask(p['precio_imputado'])
    p['close_totalreturn'] = sin_imputar.groupby(p['assetid']).transform(lambda s: s.interpolate(limit_area='inside'))
    return p


def control_negativo_lookahead(panel_base, historia, cal, params, corte, evaluacion=None,
                               limpieza=limpieza_trampa_backfill, columna='close_totalreturn'):
    """Aplica la prueba a una limpieza trampa: debe detectar diferencias. Devuelve cuántas filas cambian."""
    T = pd.Timestamp(corte)
    completo = limpieza(panel_base, historia, cal, params, evaluacion)
    truncado = limpieza(panel_base[panel_base['date'] <= T], historia[historia['date'] <= T], cal, params, evaluacion)
    a = truncado[columna].reset_index(drop=True)
    b = completo.loc[completo['date'] <= T, columna].reset_index(drop=True)
    distintas = int((~((a == b) | (a.isna() & b.isna()))).sum())
    assert distintas > 0, 'El control negativo debería detectar look-ahead'
    return distintas


def cartera_equiponderada(panel, col_ret, excluir=None):
    """Retorno diario medio de los miembros (ignora NaN) y nº de miembros con retorno."""
    p = panel if excluir is None else panel[~panel[excluir]]
    g = p.groupby('date')[col_ret]
    return pd.DataFrame({'ret': g.mean(), 'n': g.count()})
