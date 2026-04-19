import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt

class PandasTableModel(QAbstractTableModel):
    def __init__(self, data_frame: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._dataframe = data_frame

    def rowCount(self, parent=None):
        return len(self._dataframe.index)

    def columnCount(self, parent=None):
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._dataframe.iat[index.row(), index.column()]
            if pd.isna(val):
                return "-"
            if isinstance(val, float):
                # Apply similar format: 1,000.00 limits, .4f, .6f
                abs_val = abs(val)
                if abs_val >= 1000:
                    return f"{val:,.2f}"
                if abs_val >= 1:
                    return f"{val:.4f}"
                return f"{val:.6f}"
            return str(val)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if column < 0 or column >= len(self._dataframe.columns):
            return
        col_name = self._dataframe.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder

        self.layoutAboutToBeChanged.emit()
        try:
            self._dataframe = self._dataframe.sort_values(by=col_name, ascending=ascending, kind="mergesort")
        except Exception:
            # Best-effort; keep existing order if sorting fails.
            pass
        self.layoutChanged.emit()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._dataframe.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(self._dataframe.index[section])
        return None
