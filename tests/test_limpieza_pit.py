"""Tests de src/limpieza_pit con datos sintéticos pequeños, donde el resultado correcto se conoce.

Ejecutar desde la raíz del proyecto:  python3 -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from limpieza_pit.calendario import Calendario, expandir_periodos, identificar_tramos, periodos_desde_indicador
from limpieza_pit.limpieza import PARAMETROS_POR_DEFECTO, aplicar_limpieza
from limpieza_pit.validacion import (control_negativo_lookahead, limpieza_trampa_interpolacion,
                                     prueba_lookahead)

SESIONES = pd.bdate_range('2020-01-01', periods=30)
CAL = Calendario(SESIONES)
PARAMS = {**PARAMETROS_POR_DEFECTO, 'FFILL_MAX_SESIONES': 3, 'RACHA_CONGELADO': 4}


def construir(precios_por_sesion):
    """Un único valor (assetid=1), miembro las 30 sesiones. ``None`` = sin precio ese día."""
    panel = pd.DataFrame({'date': SESIONES, 'assetid': 1, 'membership_period': 1})
    panel['close_totalreturn'] = pd.array(precios_por_sesion, dtype='Float64').astype(float)
    panel['close_none'] = panel['close_totalreturn']
    panel['has_price'] = panel['close_totalreturn'].notna()
    historia = panel.loc[panel['has_price'], ['assetid', 'date', 'close_totalreturn', 'close_none']]
    return panel, historia.reset_index(drop=True)


# Precio 10, 11, 12…; hueco corto (sesiones 5-6) y hueco largo (sesiones 12-19)
PRECIOS = [10.0 + i for i in range(30)]
for i in [5, 6] + list(range(12, 20)):
    PRECIOS[i] = None
PANEL, HISTORIA = construir(PRECIOS)
LIMPIO = aplicar_limpieza(PANEL, HISTORIA, CAL, PARAMS)


def test_calendario_cuenta_sesiones_no_dias():
    viernes, lunes = SESIONES[2], SESIONES[3]
    assert (lunes - viernes).days == 3 and np.diff(CAL.posicion([viernes, lunes]))[0] == 1
    assert np.isnan(CAL.posicion([pd.Timestamp('2020-01-04')])[0])   # sábado: no es sesión


def test_ffill_rellena_solo_hasta_el_maximo():
    imp = LIMPIO['precio_imputado'].to_numpy()
    assert imp[[5, 6]].all()                            # hueco corto: completo
    assert imp[12:15].all() and not imp[15:20].any()    # hueco largo: solo las 3 primeras sesiones
    assert LIMPIO.loc[5, 'close_totalreturn'] == 14.0 and LIMPIO.loc[6, 'close_totalreturn'] == 14.0


def test_tipo_hueco_y_excluible():
    assert (LIMPIO.loc[15:19, 'tipo_hueco'] == 'prolongado_sin_imputar').all()
    assert LIMPIO.loc[15:19, 'excluible'].all() and not LIMPIO.loc[:14, 'excluible'].any()


def test_retornos_y_motivos():
    m = LIMPIO['motivo_ret']
    assert m[0] == 'primer_precio' and m[1] == 'normal'
    assert m[5] == 'imputado_cero' and LIMPIO.loc[5, 'ret_tr_limpio'] == 0
    assert m[7] == 'acumulado_tras_hueco_corto'
    assert LIMPIO.loc[7, 'ret_tr_limpio'] == pytest.approx(17 / 14 - 1)     # todo el movimiento del hueco
    assert m[20] == 'tras_hueco_largo' and np.isnan(LIMPIO.loc[20, 'ret_tr_limpio'])


def test_precio_congelado_cuenta_solo_hacia_atras():
    precios = [10.0] * 6 + [11.0] * 24              # 6 sesiones iguales al principio
    p, h = construir(precios)
    r = aplicar_limpieza(p, h, CAL, PARAMS)
    assert r['sesiones_mismo_cierre'].tolist()[:7] == [1, 2, 3, 4, 5, 6, 1]
    assert r['precio_congelado'].tolist()[:7] == [False, False, False, True, True, True, False]


def test_prueba_lookahead_pasa_y_control_negativo_falla():
    cortes = [SESIONES[5], SESIONES[13]]          # en mitad del hueco corto y del largo
    res = prueba_lookahead(PANEL, HISTORIA, CAL, PARAMS, cortes)
    assert (res['resultado'] == 'idénticas ✔').all()
    assert control_negativo_lookahead(PANEL, HISTORIA, CAL, PARAMS, SESIONES[5],
                                      limpieza=limpieza_trampa_interpolacion) > 0


def test_periodos_desde_indicador():
    # in = 1 1 [sin fila] 1 0 0 1 1 → dos periodos; la sesión sin fila es pertenencia
    fechas = SESIONES[:8].delete(2)
    filas = pd.DataFrame({'assetid': 7, 'date': fechas, 'in_sp500': [1, 1, 1, 0, 0, 1, 1]})
    per = periodos_desde_indicador(filas, CAL, SESIONES[0], SESIONES[7])
    assert per['membership_period'].tolist() == [1, 2]
    assert per['sesiones'].tolist() == [4, 2] and per['sesiones_sin_fila'].tolist() == [1, 0]
    assert per['left_censored'].tolist() == [True, False] and per['right_censored'].tolist() == [False, True]
    assert len(expandir_periodos(per, SESIONES)) == 6


def test_identificar_tramos():
    filas = PANEL.loc[~PANEL['has_price'], ['date', 'assetid']]
    _, tramos = identificar_tramos(filas, CAL)
    assert tramos['sesiones'].tolist() == [2, 8]
