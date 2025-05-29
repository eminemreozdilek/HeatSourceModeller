import numpy as np
from PySide6.QtWidgets import QTableWidget


def table_to_numpy(table_widget: QTableWidget) -> np.ndarray:
    """Converts a QTableWidget to a NumPy array."""
    rows = table_widget.rowCount()
    cols = table_widget.columnCount()
    numpy_array = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            item = table_widget.item(i, j)
            if item is not None:
                try:
                    numpy_array[i, j] = float(item.text())
                except ValueError:
                    numpy_array[i, j] = item.text()
            else:
                numpy_array[i, j] = 0.0

    return numpy_array.astype(float)
