import sys
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMenuBar, QMenu,
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QSpinBox, QMessageBox, QLineEdit, QDialog)

from PySide6.QtGui import QAction, QIcon, QPalette, QColor
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor
import pyvista as pv

import command_writer as cw
import table_editor as te
import spline as sp


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fergani-Kaynak Modelleyici")
        icon = QIcon("icon.ico")
        self.setWindowIcon(icon)

        self.mesh = None
        self.selected_ids = []
        self.spline_actor = None
        self.point_actors = {}
        self.select_mode = False

        self.welding_length = 0.0

        # Central Widget Layout
        self.widget_central = QWidget()
        self.setCentralWidget(self.widget_central)
        self.layout_central = QHBoxLayout()
        self.widget_central.setLayout(self.layout_central)

        # Editor Layout
        self.layout_editor = QVBoxLayout()

        # PyVista render window
        self.plotter = QtInteractor(self)
        self.plotter.enable_trackball_style()
        self.layout_central.addWidget(self.plotter)

        # Editor Panel
        self.widget_editor = QWidget()
        self.widget_editor.setMinimumWidth(320)
        self.widget_editor.setMaximumWidth(320)
        self.widget_editor.setLayout(self.layout_editor)
        self.layout_central.addWidget(self.widget_editor)

        # Selection toggle button
        self.button_select = QPushButton("Activate Select Mode")
        self.button_select.setCheckable(True)
        self.button_select.clicked.connect(self.toggle_selection_mode)
        self.button_select.setMinimumWidth(300)
        self.button_select.setMaximumWidth(300)
        self.layout_editor.addWidget(self.button_select)

        # Reset toggle button
        self.button_reset = QPushButton("Reset")
        self.button_reset.clicked.connect(self.reset_tables)
        self.button_reset.setMinimumWidth(300)
        self.button_reset.setMaximumWidth(300)
        self.layout_editor.addWidget(self.button_reset)

        # Table to show coordinates
        self.table_points = QTableWidget(0, 3)
        self.table_points.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.table_points.cellChanged.connect(self.update_point_from_table)
        self.table_points.setMinimumWidth(300)
        self.table_points.setMaximumWidth(300)
        self.table_points.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout_editor.addWidget(self.table_points)

        # Add Welding Parameters
        self.label_welding_length_title = QLabel(f"Welding Length: {self.welding_length}")
        self.layout_editor.addWidget(self.label_welding_length_title)

        self.label_welding_duration_title = QLabel(f"Welding Duration:")
        self.layout_editor.addWidget(self.label_welding_duration_title)

        self.spinbox_welding_duration = QSpinBox()
        self.spinbox_welding_duration.setMaximum(100000)
        self.spinbox_welding_duration.setSingleStep(1)
        self.spinbox_welding_duration.setValue(2)
        self.spinbox_welding_duration.setMinimum(2)
        self.layout_editor.addWidget(self.spinbox_welding_duration)

        self.label_cooling_duration_title = QLabel(f"Cooling Duration:")
        self.layout_editor.addWidget(self.label_cooling_duration_title)

        self.spinbox_cooling_duration = QSpinBox()
        self.spinbox_cooling_duration.setMaximum(100000)
        self.spinbox_cooling_duration.setSingleStep(1)
        self.spinbox_cooling_duration.setValue(1)
        self.spinbox_cooling_duration.setMinimum(0)
        self.layout_editor.addWidget(self.spinbox_cooling_duration)

        self.table_parameters = QTableWidget(9, 1)
        self.table_parameters.setVerticalHeaderLabels([
            "Heat Central",
            "A",
            "B",
            "C Front",
            "C Rear",
            "Factor Front",
            "Factor Rear",
            "Convection",
            "Test Tempreture"])
        self.table_parameters.setHorizontalHeaderLabels(["Value"])
        self.table_parameters.setMinimumWidth(300)
        self.table_parameters.setMaximumWidth(300)
        self.table_parameters.setMaximumHeight(300)
        self.table_parameters.setMinimumHeight(300)
        self.table_parameters.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_parameters.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_parameters.setItem(0, 0, QTableWidgetItem("600000.0"))
        self.table_parameters.setItem(1, 0, QTableWidgetItem("5.0"))
        self.table_parameters.setItem(2, 0, QTableWidgetItem("5.0"))
        self.table_parameters.setItem(3, 0, QTableWidgetItem("5.0"))
        self.table_parameters.setItem(4, 0, QTableWidgetItem("10.0"))
        self.table_parameters.setItem(5, 0, QTableWidgetItem("0.67"))
        self.table_parameters.setItem(6, 0, QTableWidgetItem("1.33"))
        self.table_parameters.setItem(7, 0, QTableWidgetItem("2.0e8"))
        self.table_parameters.setItem(8, 0, QTableWidgetItem("22.0"))
        self.layout_editor.addWidget(self.table_parameters)

        # Line edit + button for adding by ID
        self.input_ids = QLineEdit()
        self.input_ids.setPlaceholderText("Enter point IDs, e.g. 0,5,12")
        self.layout_editor.addWidget(self.input_ids)

        self.button_add_ids = QPushButton("Add Points by ID")
        self.button_add_ids.setMinimumWidth(300)
        self.button_add_ids.clicked.connect(self.add_points_by_id)
        self.layout_editor.addWidget(self.button_add_ids)

        # Button to list all mesh points
        self.button_list_points = QPushButton("Show All Mesh Points")
        self.button_list_points.setMinimumWidth(300)
        self.button_list_points.clicked.connect(self.show_all_points)
        self.layout_editor.addWidget(self.button_list_points)

        # Toggle mesh point labels
        self.button_toggle_labels = QPushButton("Show Mesh Labels")
        self.button_toggle_labels.setCheckable(True)
        self.button_toggle_labels.setMinimumWidth(300)
        self.button_toggle_labels.clicked.connect(self.toggle_labels)
        self.layout_editor.addWidget(self.button_toggle_labels)

        # Will hold the labels actor so we can remove it later
        self.labels_actor = None

        # Setup the menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.menu = QMenu("Files", self)
        self.menu_bar.addMenu(self.menu)

        self.import_action = QAction("Import STL", self)
        self.import_action.triggered.connect(self.import_stl)
        self.menu.addAction(self.import_action)

        self.import_csv_action = QAction("Import CSV", self)
        self.import_csv_action.triggered.connect(self.import_csv)
        self.menu.addAction(self.import_csv_action)

        self.export_csv_action = QAction("Export CSV", self)
        self.export_csv_action.triggered.connect(self.export_csv)
        self.menu.addAction(self.export_csv_action)

        self.export_action = QAction("Export APDL", self)
        self.export_action.triggered.connect(self.export_apdl)
        self.menu.addAction(self.export_action)

    def toggle_labels(self):
        """Show or hide labels for every mesh point (ID: X,Y,Z)."""
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Please import an STL first.")
            # un-check the toggle
            self.button_toggle_labels.setChecked(False)
            return

        if self.button_toggle_labels.isChecked():
            # Build a list of "ID: x,y,z" strings
            coords = self.mesh.points
            labels = [
                # f"{i}: {pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}"
                f"{i}"
                for i, pt in enumerate(coords)
            ]
            # Add the labels actor
            self.labels_actor = self.plotter.add_point_labels(
                self.mesh,  # the mesh to pull points from
                labels,  # labels list, one per point
                point_size=0,  # hide the little glyphs
                font_size=10,
                shape_opacity=0  # fully transparent background
            )
            self.button_toggle_labels.setText("Hide Mesh Labels")
        else:
            # Remove the labels actor
            if self.labels_actor is not None:
                self.plotter.remove_actor(self.labels_actor)
                self.labels_actor = None
            self.button_toggle_labels.setText("Show Mesh Labels")

        self.plotter.render()

    def reset_tables(self):
        # remove all point actors from the plotter
        for actor in self.point_actors.values():
            self.plotter.remove_actor(actor)
        self.point_actors.clear()

        # clear the list of selected point IDs
        self.selected_ids.clear()

        # remove spline if it exists
        if self.spline_actor:
            self.plotter.remove_actor(self.spline_actor)
            self.spline_actor = None

        # delete all rows in the points table (headers stay)
        self.table_points.setRowCount(0)

        # reset welding length and update label
        self.welding_length = 0.0
        self.label_welding_length_title.setText(f"Welding Length: {self.welding_length:.3f}")

        # re‑render the scene to reflect removals
        self.plotter.render()

    def toggle_selection_mode(self):
        self.select_mode = self.button_select.isChecked()
        if self.select_mode:
            self.button_select.setText("Deactivate Select Mode")
            self.plotter.enable_point_picking(
                callback=self.on_pick,
                use_picker=True,
                show_message=False,
                show_point=False,
                left_clicking=False
            )
        else:
            self.button_select.setText("Activate Select Mode")
            self.plotter.disable_picking()

    def import_stl(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open STL File", "", "STL files (*.stl)"
        )
        if filename:
            self.load_stl(filename)

    def export_apdl(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save APDL", "", "APDL files (*.txt)")
        if filename:
            points = te.table_to_numpy(self.table_points)
            values = te.table_to_numpy(self.table_parameters).flatten()

            cw.write_apdl_commands(filename,
                                   points,
                                   values,
                                   self.spinbox_welding_duration.value(),
                                   self.spinbox_cooling_duration.value())

    def load_stl(self, filename):
        self.plotter.clear()
        self.mesh = pv.read(filename)

        self.mesh.points *= 1000
        self.plotter.add_mesh(
            self.mesh,
            color="lightgray",
            opacity=0.5,
            show_edges=True
        )
        self.plotter.reset_camera()
        self.plotter.show_axes()

        self.plotter.show_grid(
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True, )

        self.selected_ids.clear()
        self.point_actors.clear()
        self.table_points.setRowCount(0)
        if self.spline_actor:
            self.plotter.remove_actor(self.spline_actor)
            self.spline_actor = None

    def on_pick(self, picker, event=None):
        if not self.mesh:
            return

        picked_point = picker
        point_id = self.mesh.find_closest_point(picked_point, n=1)

        if point_id not in self.selected_ids:
            self.selected_ids.append(point_id)
            coords = self.mesh.points[point_id]
            actor = self.plotter.add_points(
                coords,
                render_points_as_spheres=True,
                point_size=10,
                color="blue"
            )

            self.point_actors[point_id] = actor

            row = self.table_points.rowCount()
            self.table_points.insertRow(row)
            for i, val in enumerate(coords):
                item = QTableWidgetItem(f"{val:.3f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.table_points.setItem(row, i, item)

        self.update_spline()

    def update_spline(self):
        if self.spline_actor:
            self.plotter.remove_actor(self.spline_actor)

        pts = te.table_to_numpy(self.table_points)

        if pts.shape[0] >= 2:
            self.welding_length = sp.calculate_position_and_directions(pts, 2)[0]
            self.label_welding_length_title.setText(f"Welding Length: {self.welding_length:.3f}")
            # Draw spline
            spline = pv.Spline(pts, n_points=100)
            self.spline_actor = self.plotter.add_mesh(
                spline,
                color="red",
                line_width=5
            )

        self.plotter.render()

    def update_point_from_table(self, row, col):
        try:
            x = float(self.table_points.item(row, 0).text())
            y = float(self.table_points.item(row, 1).text())
            z = float(self.table_points.item(row, 2).text())
            point_id = self.selected_ids[row]

            self.plotter.remove_actor(self.point_actors[point_id])
            self.point_actors[point_id] = self.plotter.add_points(
                np.array([x, y, z]),
                render_points_as_spheres=True,
                point_size=10,
                color="blue"
            )
            self.update_spline()

        except Exception as e:
            print("Error updating point:", e)

    def add_points_by_id(self):
        """Parse IDs from the line edit and add them as picked points."""
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Please import an STL first.")
            return

        text = self.input_ids.text()
        try:
            ids = [int(s.strip()) for s in text.split(",") if s.strip() != '']
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Please enter only integers separated by commas.")
            return

        max_id = self.mesh.n_points - 1
        added = 0
        for point_id in ids:
            if point_id < 0 or point_id > max_id:
                QMessageBox.warning(self,
                                    "ID out of range",
                                    f"Point ID {point_id} is out of range (0 to {max_id})."
                                    )
                continue

            if point_id in self.selected_ids:
                # already added
                continue

            coords = self.mesh.points[point_id]
            actor = self.plotter.add_points(
                coords,
                render_points_as_spheres=True,
                point_size=10,
                color="blue"
            )
            self.point_actors[point_id] = actor
            self.selected_ids.append(point_id)

            # add to table
            row = self.table_points.rowCount()
            self.table_points.insertRow(row)
            for i, val in enumerate(coords):
                item = QTableWidgetItem(f"{val:.3f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.table_points.setItem(row, i, item)

            added += 1

        if added:
            self.update_spline()
            self.plotter.render()
        else:
            QMessageBox.information(self, "No new points", "No valid new IDs were added.")

    def show_all_points(self):
        """Open a dialog listing every mesh point ID and its coordinates."""
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh loaded", "Please import an STL first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("All Mesh Points (ID, X, Y, Z)")
        table = QTableWidget()
        n = self.mesh.n_points
        table.setRowCount(n)
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["ID", "X", "Y", "Z"])
        table.setMinimumSize(500, 400)

        for i in range(n):
            coords = self.mesh.points[i]
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            for j, v in enumerate(coords, start=1):
                item = QTableWidgetItem(f"{v:.3f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, item)

        layout = QVBoxLayout()
        layout.addWidget(table)
        dialog.setLayout(layout)
        dialog.resize(600, 500)
        dialog.exec()

    def export_csv(self):
        """Save table_points to a CSV file."""
        if self.table_points.rowCount() == 0:
            QMessageBox.information(self, "No data", "There are no points to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        # get numpy array [n_points × 3]
        arr = te.table_to_numpy(self.table_points)
        # make a DataFrame with columns X, Y, Z
        df = pd.DataFrame(arr, columns=["X", "Y", "Z"])
        try:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Saved {len(df)} points to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save CSV:\n{e}")

    def import_csv(self):
        """Load X,Y,Z triplets from a CSV into table_points."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read CSV:\n{e}")
            return

        # Expect exactly three columns
        if df.shape[1] != 3:
            QMessageBox.critical(self, "Invalid format", "CSV must have exactly 3 columns (X,Y,Z).")
            return

        # Clear any existing points & actors
        self.reset_tables()

        # Populate table_points
        n = len(df)
        self.table_points.setRowCount(n)
        for i in range(n):
            for j, col in enumerate(df.columns[:3]):
                val = df.iloc[i, j]
                item = QTableWidgetItem(f"{val:.3f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.table_points.setItem(i, j, item)

        # Recompute spline from imported coords
        self.update_spline()
        QMessageBox.information(self, "Imported", f"Loaded {n} points from:\n{path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1920, 1080)
    window.show()
    sys.exit(app.exec())
