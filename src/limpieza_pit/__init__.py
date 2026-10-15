"""Limpieza de paneles point-in-time (PIT) de índices bursátiles.

Funciones comunes a los notebooks de limpieza del Nasdaq 100 y del S&P 500.

Esquema de trabajo
------------------
Todas las funciones operan sobre dos DataFrames con nombres de columna canónicos
(los del dataset del Nasdaq 100). Cada notebook renombra sus columnas a este esquema
al cargar los datos:

``panel``    Una fila por (``date``, ``assetid``) en la que el valor es miembro del índice.
             Columnas: ``date``, ``assetid``, ``membership_period``, ``has_price``,
             ``close_totalreturn`` (cierre ajustado por splits y dividendos) y
             ``close_none`` (cierre sin ajustar). Opcionales: ``capital_event``, ``volume_none``.
             Debe estar ordenado por (``date``, ``assetid``).
``historia`` Todos los precios reales disponibles de cada valor, también fuera de la
             pertenencia: ``assetid``, ``date``, ``close_totalreturn``, ``close_none``.
             Debe estar ordenado por ``date``.

Principios
----------
- No se borra ninguna fila: los problemas se marcan con columnas indicadoras.
- Ninguna regla de limpieza usa información posterior a la fecha de cada fila
  (se comprueba con :func:`validacion.prueba_lookahead`).
"""

from .calendario import Calendario, expandir_periodos, identificar_tramos, periodos_desde_indicador
from .limpieza import (
    PARAMETROS_POR_DEFECTO,
    anadir_ultimo_precio_real,
    aplicar_limpieza,
    clasificar_filas_sin_precio,
    detectar_saltos_split,
    imputar_ffill_causal,
    marcar_precios_congelados,
    marcar_retornos_anomalos,
    marcar_salida_por_fin_cotizacion,
    marcar_valores_descartados,
    recalcular_retornos,
    valores_imputados,
)

__all__ = [
    'Calendario', 'expandir_periodos', 'identificar_tramos', 'periodos_desde_indicador',
    'PARAMETROS_POR_DEFECTO', 'anadir_ultimo_precio_real', 'aplicar_limpieza',
    'clasificar_filas_sin_precio', 'detectar_saltos_split', 'imputar_ffill_causal',
    'marcar_precios_congelados', 'marcar_retornos_anomalos', 'marcar_salida_por_fin_cotizacion',
    'marcar_valores_descartados', 'recalcular_retornos', 'valores_imputados',
]
