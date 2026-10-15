"""Gráficos de los notebooks de limpieza: paleta fija, trazos finos y etiquetas directas."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLOR = {'azul': '#2a78d6', 'naranja': '#eb6834', 'aqua': '#1baf7a', 'amarillo': '#eda100'}
FONDO, TINTA, TINTA_2, GRIS_REAL = '#fcfcfb', '#0b0b0b', '#52514e', '#d9d8d2'


def aplicar_estilo():
    plt.rcParams.update({
        'figure.dpi': 110, 'figure.facecolor': FONDO, 'axes.facecolor': FONDO,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7',
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'lines.linewidth': 2, 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'text.color': TINTA, 'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
        'legend.frameon': False,
    })


def grafico_miembros_supervivientes(miembros, indice):
    """Miembros por sesión: universo real frente a solo empresas que cotizan hoy."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(miembros.index, miembros['todos'], color=COLOR['azul'])
    ax.plot(miembros.index, miembros['solo_supervivientes'], color=COLOR['naranja'])
    ax.annotate('Miembros reales (PIT)', (miembros.index[-1], miembros['todos'].iloc[-1]),
                xytext=(-5, 8), textcoords='offset points', ha='right', color=TINTA)
    ax.annotate('Solo empresas que cotizan hoy\n(sesgo de supervivencia)',
                (miembros.index[300], miembros['solo_supervivientes'].iloc[300]),
                xytext=(0, -38), textcoords='offset points', color=TINTA)
    ax.set_ylim(0, miembros['todos'].max() * 1.12)
    ax.set_ylabel('Valores en el índice')
    ax.set_title(f'Miembros del {indice} por sesión: universo real frente a solo supervivientes')
    plt.tight_layout()
    plt.show()


def grafico_huecos_por_anio(pct_anio, por_anio):
    """Barras apiladas: % de filas de miembro sin precio por año y clase de tramo."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    base = np.zeros(len(pct_anio))
    for clase, color in zip(pct_anio.columns, [COLOR['azul'], COLOR['naranja'], COLOR['aqua']]):
        ax.bar(pct_anio.index, pct_anio[clase], bottom=base, color=color, edgecolor=FONDO, linewidth=1,
               width=0.8, label=f'{clase} ({int(por_anio[clase].sum()):,} filas)')
        base += pct_anio[clase].to_numpy()
    ax.set_ylabel('% de filas de miembro sin precio')
    ax.set_title('Huecos de precio por año y tipo de tramo')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def grafico_crecimiento(crec, series, titulo, ylabel, anotaciones=(), margen_superior=None):
    """Crecimiento acumulado (escala log) de varias series.

    ``series``: lista de (columna, etiqueta, color). ``anotaciones``: lista de
    (fecha, columna, texto, desplazamiento en puntos).
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col, etiqueta, color in series:
        ax.plot(crec.index, crec[col], color=color, label=f'{etiqueta}: ×{crec[col].iloc[-1]:,.1f}')
    for fecha, col, texto, desplaz in anotaciones:
        ax.annotate(texto, (fecha, crec.loc[fecha, col]), xytext=desplaz, textcoords='offset points',
                    fontsize=9, color=TINTA, arrowprops=dict(arrowstyle='-', color=TINTA_2, lw=0.8))
    ax.set_yscale('log')
    if margen_superior:
        ax.set_ylim(top=crec.max().max() * margen_superior)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def _segmentos(df, col, cal):
    """Agrupa sesiones consecutivas de pertenencia con el mismo estado en segmentos (inicio, ancho, estado)."""
    df = df.sort_values('date')
    pos = cal.posicion(df['date'])
    corte = (df[col] != df[col].shift()) | (np.diff(pos, prepend=np.nan) != 1)
    seg = df.groupby(corte.cumsum()).agg(inicio=('date', 'min'), fin=('date', 'max'), estado=(col, 'first'))
    x0 = mdates.date2num(seg['inicio'])
    return x0, mdates.date2num(seg['fin']) + 1 - x0, seg['estado']


def grafico_mapa_huecos(datos, afectados, etiquetas, cal, ffill_max):
    """Mapa antes/después de los valores con huecos: un panel por estado del dataset.

    ``datos``: filas del panel de los valores ``afectados`` con ``date``, ``assetid``,
    ``has_price`` y ``tipo_hueco``. Solo se dibujan los días de pertenencia.
    """
    colores_despues = {'ninguno': GRIS_REAL, 'imputado_ffill': GRIS_REAL,
                       'prolongado_sin_imputar': COLOR['naranja'], 'sin_precio_previo': COLOR['azul']}
    colores_antes = {True: GRIS_REAL, False: TINTA_2}
    alto = max(3.5, 0.3 * len(afectados) + 1.5)
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(13, alto), sharey=True)
    for y, aid in enumerate(afectados):
        d = datos[datos['assetid'] == aid]
        for ax, col, colores in [(ax_a, 'has_price', colores_antes), (ax_d, 'tipo_hueco', colores_despues)]:
            x0, ancho, estado = _segmentos(d, col, cal)
            ax.broken_barh(list(zip(x0, ancho)), (y - 0.35, 0.7), facecolors=[colores[e] for e in estado])
        imp = d.loc[d['tipo_hueco'] == 'imputado_ffill', 'date']
        ax_d.scatter(imp, np.full(len(imp), y), marker='^', s=36, color=COLOR['aqua'],
                     edgecolor=FONDO, linewidth=0.8, zorder=3)
    ax_a.set_yticks(range(len(afectados)), etiquetas)
    for ax in (ax_a, ax_d):
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(axis='x', color='#e6e5e0', linewidth=0.6)
        ax.grid(axis='y', visible=False)
        ax.set_ylim(-0.8, len(afectados) - 0.2)
    n_sin = int((~datos['has_price']).sum())
    cuenta = datos['tipo_hueco'].value_counts().reindex(list(colores_despues), fill_value=0)
    ax_a.set_title(f'Antes: {n_sin:,} filas «sin precio», sin distinguir')
    ax_d.set_title('Después: cada tramo clasificado y tratado')
    ax_a.legend(handles=[Patch(color=GRIS_REAL, label='Miembro con precio'),
                         Patch(color=TINTA_2, label=f'Miembro sin precio ({n_sin:,})')],
                loc='upper left', bbox_to_anchor=(0, -0.05), ncol=2, fontsize=9)
    ax_d.legend(handles=[Patch(color=GRIS_REAL, label='Miembro con precio real'),
                         Line2D([], [], marker='^', ls='', color=COLOR['aqua'], markersize=7,
                                label=f"Imputado, ffill ≤ {ffill_max} sesiones ({cuenta['imputado_ffill']:,})"),
                         Patch(color=COLOR['naranja'], label=f"Excluible: prolongado ({cuenta['prolongado_sin_imputar']:,})"),
                         Patch(color=COLOR['azul'], label=f"Excluible: sin precio previo ({cuenta['sin_precio_previo']:,})")],
                loc='upper left', bbox_to_anchor=(0, -0.05), ncol=2, fontsize=9)
    ax_a.text(0, -0.12 * 8.5 / alto, '† deslistado hoy', transform=ax_a.transAxes, fontsize=9, color=TINTA_2)
    plt.tight_layout()
    plt.show()
