"""Calendario de sesiones y reconstrucción de la pertenencia al índice."""

import numpy as np
import pandas as pd


class Calendario:
    """Sesiones de negociación ordenadas.

    Permite contar «sesiones sin precio» en lugar de días naturales, de modo que
    fines de semana y festivos nunca cuentan como huecos.
    """

    def __init__(self, sesiones):
        self.sesiones = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(sesiones))))

    def __len__(self):
        return len(self.sesiones)

    def __repr__(self):
        return f'Calendario({self.sesiones[0].date()} → {self.sesiones[-1].date()}, {len(self):,} sesiones)'

    def posicion(self, fechas):
        """Número de orden de cada fecha en el calendario (NaN si es nula o no es sesión)."""
        pos = self.sesiones.get_indexer(pd.DatetimeIndex(fechas)).astype(float)
        pos[pos < 0] = np.nan
        return pos

    def fechas_fuera(self, fechas):
        """Fechas que no son sesión del calendario."""
        fechas = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(fechas)))
        return fechas.difference(self.sesiones)


def expandir_periodos(periodos, sesiones):
    """Una fila (date, assetid, membership_period) por cada sesión de cada periodo.

    Los extremos ``first_included_session`` y ``last_included_session`` son inclusivos.
    """
    fechas = pd.DatetimeIndex(sesiones).to_numpy()
    ini = np.searchsorted(fechas, periodos['first_included_session'].to_numpy(), side='left')
    fin = np.searchsorted(fechas, periodos['last_included_session'].to_numpy(), side='right')
    n = fin - ini
    idx = np.concatenate([np.arange(a, b) for a, b in zip(ini, fin)])
    return pd.DataFrame({'date': fechas[idx],
                         'assetid': np.repeat(periodos['assetid'].to_numpy(), n),
                         'membership_period': np.repeat(periodos['membership_period'].to_numpy(), n)})


def periodos_desde_indicador(filas, cal, inicio, fin, col_miembro='in_sp500'):
    """Reconstruye periodos de pertenencia a partir de un indicador diario 0/1.

    Se usa cuando el dataset no trae los periodos explícitos (p. ej. el S&P 500, donde
    cada fila de precio lleva ``in_sp500``). Un periodo es una secuencia de filas con
    indicador 1 **sin ninguna fila con indicador 0 entre medias**. Las sesiones sin fila
    dentro de un periodo se consideran de pertenencia (son huecos de precio); las que
    caen entre una fila con 1 y otra con 0 no se asignan, porque su pertenencia es
    desconocida.

    ``inicio`` y ``fin`` son el primer y último día de cobertura del dataset: un periodo
    que empieza (termina) en ellos está censurado a la izquierda (derecha).
    """
    f = filas[['assetid', 'date', col_miembro]].sort_values(['assetid', 'date'])
    nuevo_bloque = (f['assetid'] != f['assetid'].shift()) | (f[col_miembro] != f[col_miembro].shift())
    f = f.assign(bloque=nuevo_bloque.cumsum())
    periodos = (f[f[col_miembro] == 1].groupby('bloque')
                .agg(assetid=('assetid', 'first'), first_included_session=('date', 'min'),
                     last_included_session=('date', 'max'), filas_observadas=('date', 'size'))
                .reset_index(drop=True))
    periodos.insert(1, 'membership_period', periodos.groupby('assetid').cumcount().add(1).astype('int32'))
    periodos['sesiones'] = (cal.posicion(periodos['last_included_session'])
                            - cal.posicion(periodos['first_included_session']) + 1).astype(int)
    periodos['sesiones_sin_fila'] = periodos['sesiones'] - periodos['filas_observadas']
    periodos['left_censored'] = periodos['first_included_session'] == pd.Timestamp(inicio)
    periodos['right_censored'] = periodos['last_included_session'] == pd.Timestamp(fin)
    return periodos


def identificar_tramos(filas, cal):
    """Agrupa en tramos las sesiones consecutivas sin precio de un mismo valor.

    Mismo criterio que ``auditar_huecos.py`` del dataset del Nasdaq 100: número de
    sesión menos número de fila dentro del valor. Devuelve las filas con su ``tramo``
    y un resumen por tramo (``assetid``, ``inicio``, ``fin``, ``sesiones``).
    """
    f = filas.sort_values(['assetid', 'date']).copy()
    f['pos'] = cal.posicion(f['date'])
    f['tramo'] = (f.groupby('assetid')['pos'].diff() != 1).cumsum()
    tramos = (f.groupby('tramo').agg(assetid=('assetid', 'first'), inicio=('date', 'min'),
                                     fin=('date', 'max'), sesiones=('date', 'size'))
              .reset_index())
    return f, tramos


def precio_vecino(filas, fechas_precio, col_fecha, direccion, nombre):
    """Fecha del precio real más cercano estrictamente antes ('backward') o después ('forward').

    Solo para diagnóstico: con ``direccion='forward'`` mira al futuro.
    """
    derecha = fechas_precio[['assetid', 'date']].rename(columns={'date': nombre}).sort_values(nombre)
    return pd.merge_asof(filas.sort_values(col_fecha), derecha, left_on=col_fecha, right_on=nombre,
                         by='assetid', direction=direccion, allow_exact_matches=False)


def clasificar_tramos(tramos, fechas_precio, cal, max_sesiones):
    """Clasifica cada tramo sin precio según la serie de precios completa (diagnóstico).

    A · sin cotización puntual: hay precio antes y después y la serie se interrumpe
        como mucho ``max_sesiones`` sesiones.
    B · prolongado: hay precio anterior pero la interrupción es más larga.
    C · sin historia previa: no existe ningún precio anterior del valor.
    """
    t = precio_vecino(tramos, fechas_precio, 'inicio', 'backward', 'precio_anterior')
    t = precio_vecino(t, fechas_precio, 'fin', 'forward', 'precio_posterior')
    t['hueco_serie_sesiones'] = cal.posicion(t['precio_posterior']) - cal.posicion(t['precio_anterior']) - 1
    t['clase'] = np.select([t['precio_anterior'].isna(), t['hueco_serie_sesiones'].le(max_sesiones)],
                           ['C · sin historia previa', 'A · sin cotización puntual'], default='B · prolongado')
    return t
