"""Utilidades para los notebooks."""

import inspect
from pathlib import Path

from IPython.display import Markdown, display


def raiz_proyecto(marcador='src'):
    """Directorio raíz del proyecto (el primero, subiendo desde el actual, que contiene ``marcador``)."""
    return next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / marcador).is_dir())


def ver_codigo(*funciones):
    """Muestra en el notebook el código fuente de funciones del módulo, para revisarlas sin abrir src/."""
    for f in funciones:
        ruta = Path(inspect.getsourcefile(f)).name
        display(Markdown(f'`src/limpieza_pit/{ruta}` · **`{f.__name__}`**\n```python\n{inspect.getsource(f)}```'))
