"""Reglas de limpieza. Una función por decisión; todas causales (solo miran al pasado)."""

import numpy as np
import pandas as pd

PARAMETROS_POR_DEFECTO = {
    'FFILL_MAX_SESIONES': 5,      # máximo de sesiones sin precio que se rellenan con el último precio
    'UMBRAL_HUECOS_VALOR': None,  # opción A (descartar valores enteros): desactivada
    'UMBRAL_RET_ANOMALO': 0.50,   # |retorno diario| a partir del cual se marca para revisión
    'UMBRAL_DIF_AJUSTE': 0.20,    # |ret_none - ret_tr| que indica operación corporativa ajustada
    'TOL_SPLIT': 0.02,            # tolerancia relativa para considerar un factor de split entero
    'TOL_OHLC': 1e-6,             # tolerancia relativa en low <= open, close <= high
    'RACHA_CONGELADO': 5,         # sesiones seguidas con el mismo cierre para marcar precio congelado
}

MOTIVOS_RET_VALIDOS = ['normal', 'imputado_cero', 'acumulado_tras_hueco_corto']


def anadir_ultimo_precio_real(panel, historia, cal):
    """Añade, para cada fila, el último precio REAL estrictamente anterior y las sesiones transcurridas.

    Usa la historia completa del valor (también días en que no era miembro), porque el
    último precio conocido puede ser de antes de su entrada en el índice.
    Columnas nuevas: ``fecha_ultimo_precio_real``, ``close_tr_previo``, ``close_none_previo``,
    ``sesiones_desde_ultimo_precio``.
    """
    derecha = historia[['assetid', 'date', 'close_totalreturn', 'close_none']].rename(columns={
        'date': 'fecha_ultimo_precio_real', 'close_totalreturn': 'close_tr_previo',
        'close_none': 'close_none_previo'})
    res = pd.merge_asof(panel, derecha, left_on='date', right_on='fecha_ultimo_precio_real', by='assetid',
                        direction='backward', allow_exact_matches=False)
    res['sesiones_desde_ultimo_precio'] = cal.posicion(res['date']) - cal.posicion(res['fecha_ultimo_precio_real'])
    return res


def imputar_ffill_causal(panel, max_sesiones):
    """Rellena con el último precio real las filas sin precio si han pasado <= ``max_sesiones`` sesiones.

    El criterio es «sesiones desde el último precio», no «duración total del hueco»
    (que solo se conoce al terminar el hueco y sería look-ahead). Marca ``precio_imputado``;
    ``has_price`` conserva su valor original.
    """
    imputar = ~panel['has_price'] & panel['sesiones_desde_ultimo_precio'].le(max_sesiones)
    panel['precio_imputado'] = imputar
    panel['close_totalreturn'] = panel['close_totalreturn'].mask(imputar, panel['close_tr_previo'])
    panel['close_none'] = panel['close_none'].mask(imputar, panel['close_none_previo'])
    return panel


def valores_imputados(panel, precios, cols_cierre, cols_desde_cierre):
    """Valores a escribir en el panel completo para las filas con ``precio_imputado``.

    ``precios``: precios reales (``assetid``, ``date`` + ``cols_cierre``) de los valores con filas imputadas.
    ``cols_cierre``: columnas que toman el último valor real (cierres de cada variante).
    ``cols_desde_cierre``: {columna: cierre} para las que se igualan a ese cierre (open, high, low).
    Devuelve un DataFrame ordenado por ``fila`` (índice de la fila en ``panel``).
    """
    filas = panel.loc[panel['precio_imputado'], ['date', 'assetid']].reset_index(names='fila')
    previos = precios[['assetid', 'date'] + cols_cierre].rename(columns={'date': 'fecha_previa'})
    res = pd.merge_asof(filas.sort_values('date'), previos.sort_values('fecha_previa'), left_on='date',
                        right_on='fecha_previa', by='assetid', direction='backward', allow_exact_matches=False)
    for col, cierre in cols_desde_cierre.items():
        res[col] = res[cierre]
    return res.drop(columns='fecha_previa').sort_values('fila', ignore_index=True)


def clasificar_filas_sin_precio(panel, evaluacion=None):
    """Etiqueta las filas sin precio que no se rellenan y las marca como ``excluible``.

    ``tipo_hueco``: ninguno | imputado_ffill | prolongado_sin_imputar | sin_precio_previo.
    ``evaluacion`` (opcional): Series assetid → evaluación del propio dataset, que se copia
    en ``evaluacion_fuente`` para las filas sin precio real.
    """
    panel['precio_utilizable'] = panel['has_price'] | panel['precio_imputado']
    panel['excluible'] = ~panel['precio_utilizable']
    panel['tipo_hueco'] = np.select(
        [panel['has_price'], panel['precio_imputado'], panel['fecha_ultimo_precio_real'].isna()],
        ['ninguno', 'imputado_ffill', 'sin_precio_previo'], default='prolongado_sin_imputar')
    if evaluacion is not None:
        panel['evaluacion_fuente'] = panel['assetid'].map(evaluacion).where(~panel['has_price'])
    return panel


def marcar_valores_descartados(panel, umbral):
    """Opción A: marca todas las filas de los valores con más de ``umbral`` de filas excluibles.

    Con ``umbral=None`` (opción B, por defecto) no se descarta nada. Ojo: la opción A usa
    la historia completa del valor, así que es look-ahead por construcción.
    """
    pct_excluible = panel.groupby('assetid')['excluible'].mean()
    descartados = pct_excluible.index[pct_excluible > umbral] if umbral is not None else []
    panel['valor_descartado'] = panel['assetid'].isin(descartados)
    return panel, pct_excluible


def recalcular_retornos(panel, max_sesiones):
    """Retorno diario limpio sobre ``close_totalreturn`` (y ``close_none``) con su motivo.

    ``motivo_ret``: normal | imputado_cero | acumulado_tras_hueco_corto | tras_hueco_largo
    | sin_precio | primer_precio. Tras un hueco de más de ``max_sesiones`` sesiones el
    retorno es NaN: sería la variación de meses o años atribuida a un solo día.
    """
    k = panel['sesiones_desde_ultimo_precio']
    panel['motivo_ret'] = np.select(
        [~panel['precio_utilizable'], k.isna(), panel['precio_imputado'], k.eq(1), k.le(max_sesiones + 1)],
        ['sin_precio', 'primer_precio', 'imputado_cero', 'normal', 'acumulado_tras_hueco_corto'],
        default='tras_hueco_largo')
    valido = panel['motivo_ret'].isin(MOTIVOS_RET_VALIDOS)
    panel['ret_tr_limpio'] = (panel['close_totalreturn'] / panel['close_tr_previo'] - 1).where(valido)
    panel['ret_none_limpio'] = (panel['close_none'] / panel['close_none_previo'] - 1).where(valido)
    return panel


def marcar_retornos_anomalos(panel, umbral):
    """``ret_anomalo``: |ret_tr_limpio| > umbral. Se marca para revisión, no se borra."""
    panel['ret_anomalo'] = panel['ret_tr_limpio'].abs() > umbral
    return panel


def detectar_saltos_split(panel, umbral_dif, tol_split):
    """Distingue splits bien ajustados de posibles splits no ajustados.

    ``accion_corporativa_ajustada``: el precio sin ajustar salta y el ajustado no.
    ``posible_split_no_ajustado``: retorno anómalo en el que ambas series saltan igual y
    el factor es casi un entero >= 2.
    ``capital_event_sesion_previa`` (si existe ``capital_event``): Norgate marca el evento
    en la sesión anterior a la fecha efectiva.
    """
    r_tr, r_none = panel['ret_tr_limpio'], panel['ret_none_limpio']
    dif = (r_none - r_tr).abs()
    panel['accion_corporativa_ajustada'] = dif > umbral_dif
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(r_tr > 0, 1 + r_tr, 1 / (1 + r_tr))
    entero = np.round(factor)
    cerca_entero = (entero >= 2) & (np.abs(factor - entero) / factor < tol_split)
    panel['posible_split_no_ajustado'] = panel['ret_anomalo'] & cerca_entero & (dif < 0.05)
    if 'capital_event' in panel:
        panel['capital_event_sesion_previa'] = panel.groupby('assetid')['capital_event'].shift(1)
    return panel


def marcar_precios_congelados(panel, min_sesiones):
    """Marca rachas de ``min_sesiones`` o más sesiones seguidas con el mismo ``close_none``.

    La racha se cuenta solo hacia atrás (``sesiones_mismo_cierre`` es su longitud hasta
    ese día), así que los primeros días de una racha no se marcan: marcarlos sería look-ahead.
    """
    igual = (panel['has_price'] & panel['sesiones_desde_ultimo_precio'].eq(1)
             & panel['close_none'].eq(panel['close_none_previo']))
    bloque = (~igual).groupby(panel['assetid']).cumsum()          # se reinicia cuando el precio cambia
    repeticiones = igual.astype(int).groupby([panel['assetid'], bloque]).cumsum()
    panel['sesiones_mismo_cierre'] = (repeticiones + 1).where(panel['has_price']).astype('Int32')
    panel['precio_congelado'] = panel['sesiones_mismo_cierre'].ge(min_sesiones).fillna(False).astype(bool)
    return panel


def marcar_salida_por_fin_cotizacion(panel, periodos_fin):
    """Marca la última sesión de los periodos que terminan porque termina la serie de precios.

    ``periodos_fin``: DataFrame con ``assetid``, ``membership_period``, ``last_included_session``.
    """
    claves = periodos_fin.rename(columns={'last_included_session': 'date'})[['date', 'assetid', 'membership_period']]
    marca = panel[['date', 'assetid', 'membership_period']].merge(claves.assign(_m=True), how='left',
                                                                  on=['date', 'assetid', 'membership_period'])
    panel['salida_por_fin_cotizacion'] = marca['_m'].fillna(False).astype(bool).to_numpy()
    return panel


def aplicar_limpieza(panel_base, historia, cal, params, evaluacion=None):
    """Aplica todas las reglas causales, en orden, sobre una copia de ``panel_base``.

    No incluye las marcas que dependen de toda la historia (opción A, salidas por fin de
    cotización): por eso esta función es la que se somete a la prueba de look-ahead.
    """
    F = params['FFILL_MAX_SESIONES']
    p = anadir_ultimo_precio_real(panel_base.copy(), historia, cal)
    p = imputar_ffill_causal(p, F)
    p = clasificar_filas_sin_precio(p, evaluacion)
    p = recalcular_retornos(p, F)
    p = marcar_retornos_anomalos(p, params['UMBRAL_RET_ANOMALO'])
    p = detectar_saltos_split(p, params['UMBRAL_DIF_AJUSTE'], params['TOL_SPLIT'])
    p = marcar_precios_congelados(p, params['RACHA_CONGELADO'])
    return p
