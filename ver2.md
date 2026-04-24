import sys
import os
import io
import traceback
import datetime
import numpy as np
import pandas as pd
import ezdxf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit,
                               QHBoxLayout, QScrollArea, QFrame, QSplitter, QComboBox,
                               QInputDialog, QMessageBox, QProgressDialog, QDialog)
from PySide6.QtGui import QTextCursor, QIcon, QCloseEvent
from PySide6.QtCore import Qt

from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union, polygonize, split, nearest_points, snap
import shapely.affinity as affinity
from shapely.strtree import STRtree
from collections import defaultdict, deque

matplotlib.use('QtAgg')
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False


# =========================================================================
# [디버깅 팝업] D단계 시각화: 가상 슬릿 위치 + 적분 경로 + q₀ 분포
# =========================================================================
class ShearFlowSingleDialog(QDialog):
    def __init__(self, graph_edges, graph_nodes, root_node, mode="q_final", parent=None):
        super().__init__(parent)

        self.graph_edges = graph_edges
        self.graph_nodes = graph_nodes
        self.root_node = root_node
        self.mode = mode

        title_map = {
            "q0": "q0 (Open Section)",
            "qc": "qc (Circulation)",
            "q_final": "q_final (Total Shear Flow)"
        }

        self.setWindowTitle(title_map.get(mode, mode))
        self.resize(900, 800)

        layout = QVBoxLayout(self)

        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.draw_plot()

    def showEvent(self, event):
        self.draw_plot()

    def get_value(self, edge):
        if self.mode == "q0":
            return float(edge.get("q0_mean", 0.0))

        elif self.mode == "qc":
            return float(edge.get("qc_correction", 0.0))

        elif self.mode == "q_final":
            return float(edge.get("q_final_mean", 0.0))

        return 0.0

    def draw_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal")

        values = [self.get_value(e) for e in self.graph_edges]
        max_abs = max([abs(v) for v in values], default=1.0)
        if max_abs < 1e-12:
            max_abs = 1.0

        cmap = plt.get_cmap("coolwarm")

        for e in self.graph_edges:
            coords = list(e["line"].coords)
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]

            val = self.get_value(e)
            color = cmap(0.5 + 0.5 * val / max_abs)
            lw = 1.5 + 4.0 * abs(val) / max_abs

            ax.plot(xs, ys, color=color, linewidth=lw)

        # root 표시
        if self.root_node in self.graph_nodes:
            x, y = self.graph_nodes[self.root_node]["coord"]
            ax.scatter([x], [y], s=60)
            ax.text(x, y, "ROOT", fontsize=9)

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=-max_abs, vmax=max_abs)
        )
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax, label="Shear Flow [N/mm]")

        ax.set_title(self.windowTitle())
        ax.grid(True)

        self.canvas.draw()

    def change_mode(self, text):
        self.mode = text
        self.draw_plot()

    def get_edge_value(self, edge):
        if self.mode == "q0":
            if "q0_mean" in edge:
                return edge["q0_mean"]
            if "q0" in edge and edge["q0"] is not None:
                return float(np.mean(edge["q0"]))
            return 0.0

        elif self.mode == "qc_correction":
            return float(edge.get("qc_correction", 0.0))

        elif self.mode == "q_final":
            if "q_final_mean" in edge:
                return edge["q_final_mean"]
            if "q_final" in edge and edge["q_final"] is not None:
                return float(np.mean(edge["q_final"]))
            return 0.0

        return 0.0

    def draw_plot(self):
        print("\n--- DRAW QC CHECK ---")
        for e in self.graph_edges[:5]:
            print(e["id"], e.get("qc_correction"))
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        values = [self.get_edge_value(e) for e in self.graph_edges]
        max_abs = max([abs(v) for v in values], default=1.0)
        if max_abs < 1e-12:
            max_abs = 1.0

        cmap = plt.get_cmap("rainbow")

        for e in self.graph_edges:
            line = e["line"]
            coords = list(line.coords)
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]

            val = self.get_edge_value(e)
            color = cmap(0.5 + 0.5 * val / max_abs)

            lw = 1.5 + 4.0 * abs(val) / max_abs

            ax.plot(xs, ys, color=color, linewidth=lw)

            mid = line.interpolate(0.5, normalized=True)
            ax.text(
                mid.x,
                mid.y,
                f"E{e['id']}\n{val:.5f}",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )

        # root node 표시
        if self.root_node is not None and self.root_node in self.graph_nodes:
            x, y = self.graph_nodes[self.root_node]["coord"]
            ax.scatter([x], [y], s=80, marker="o")
            ax.text(x, y, "ROOT", fontsize=9, fontweight="bold")

        title_map = {
            "q0": "Open-section Basic Shear Flow q0",
            "qc_correction": "Circulation Correction qc",
            "q_final": "Final Shear Flow q = q0 + qc"
        }

        ax.set_title(title_map.get(self.mode, self.mode))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=-max_abs, vmax=max_abs)
        )
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax, label="Shear Flow [N/mm]")

        self.canvas.draw()



# =========================================================================
# 메인 클래스
# =========================================================================
class UltimateShipAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HHI-FAIVE - Ship Floating Strength Analyzer")
        self.setWindowIcon(QIcon('icon.ico'))
        self.resize(2000, 1200)
        self.saved_frames_data = []
        self.current_dxf_path = ""
        self.is_processing = False
        self.debug_dialogs = []
        self.reset_analysis_data()
        self.init_ui()

    def reset_analysis_data(self):
        self.raw_1999_lines = []
        self.left_1999_segments = []
        self.lines_1102 = []
        self.lines_1102_raw = []
        self.lines_157 = []
        self.lines_6001 = []
        self.lines_7001 = []
        self.lines_8001 = []
        self.lines_9001 = []
        self.lines_minus1204 = []
        self.hull_centroid = Point(0, 0)
        self.shell_thickness_inputs = []
        self.is_calculated = False
        self.mesh_cells = []
        self.centerlines = []
        self.shear_flow_centerlines = []
        self.cell_points = []
        self.max_shell_q_idx = -1
        self.max_shell_thk = 0.0
        self.q_per_v = 0.0
        self.max_Q = 0.0

        self.graph_edges = []
        self.graph_nodes = {}
        self.root_node = None
        self.slit_edge_ids = set()

        self.calc_total_area = 0.0
        self.calc_ixx = 0.0
        self.calc_depth = 0.0
        self.calc_na_bl = 0.0
        self.calc_z_top = 0.0
        self.calc_z_btm = 0.0

        self.base_report = ""
        self.act_fb = 0.0
        self.act_fs = 0.0
        self.allow_fb = 0.0
        self.allow_fs = 0.0
        self.raw_swbm = 0.0
        self.raw_shear = 0.0
        self.max_layer_name = ""
        self.calc_max_q_val = 0.0
        for d in self.debug_dialogs:
            d.close()
        self.debug_dialogs.clear()

    def init_ui(self):
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)

        self.input_style = "background-color: white; color: black; border: 1px solid #ABB2B9; padding: 2px;"
        self.field_width = 100

        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel_layout.setAlignment(Qt.AlignTop)

        gen_box = QFrame()
        gen_box.setStyleSheet("background: #F2F4F4; border-radius: 5px; padding: 5px;")
        gen_vbox = QVBoxLayout(gen_box)
        gen_vbox.addWidget(QLabel("<b>[General Settings]</b>"))
        for lbl, attr, dval in [("Scale:", "txt_scale", "100"), ("H-Ext (mm):", "txt_ext", "10"),
                                ("V-Ext (mm):", "txt_perp", "10")]:
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            h.addStretch()
            le = QLineEdit(dval)
            le.setFixedWidth(self.field_width)
            le.setStyleSheet(self.input_style)
            setattr(self, attr, le)
            h.addWidget(le)
            gen_vbox.addLayout(h)
        control_panel_layout.addWidget(gen_box)

        self.btn_load = QPushButton("1. DXF Load 📂")
        self.btn_load.setFixedHeight(40)
        self.btn_load.setStyleSheet(
            "background-color: #2E86C1; color: white; font-weight: bold; margin-top: 5px; margin-bottom: 5px;")
        self.btn_load.clicked.connect(self.load_and_process_dxf)
        control_panel_layout.addWidget(self.btn_load)

        struct_box = QFrame()
        struct_box.setStyleSheet("background: #FEF9E7; border-radius: 5px; padding: 5px;")
        struct_vbox = QVBoxLayout(struct_box)
        struct_vbox.addWidget(QLabel("<b>[Structure Settings]</b>"))

        self.combo_section = QComboBox()
        self.combo_section.addItems(["Continuous", "Discontinuous"])
        self.combo_section.setFixedWidth(self.field_width)
        self.combo_hull = QComboBox()
        self.combo_hull.addItems(["S/H", "D/H"])
        self.combo_hull.setFixedWidth(self.field_width)
        self.combo_hull.setEnabled(False)

        h_sec = QHBoxLayout()
        h_sec.addWidget(QLabel("Section:"))
        h_sec.addStretch()
        h_sec.addWidget(self.combo_section)
        struct_vbox.addLayout(h_sec)

        h_hull = QHBoxLayout()
        h_hull.addWidget(QLabel("Hull:"))
        h_hull.addStretch()
        h_hull.addWidget(self.combo_hull)
        struct_vbox.addLayout(h_hull)

        self.combo_section.currentTextChanged.connect(self.on_section_changed)
        control_panel_layout.addWidget(struct_box)

        shear_box = QFrame()
        shear_box.setStyleSheet("background: #EBF5FB; border-radius: 5px; padding: 5px; margin-top: 10px;")
        shear_vbox = QVBoxLayout(shear_box)
        shear_vbox.addWidget(QLabel("<b>[Loading Setting]</b>"))
        for lbl, attr, dval in [("S.W.B.M (tm):", "txt_swbm", "0"), ("Shear (t):", "txt_shear_v", "10"),
                                ("Grade (K):", "txt_grade_k", "1.00")]:
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            h.addStretch()
            le = QLineEdit(dval)
            le.setFixedWidth(self.field_width)
            le.setStyleSheet(self.input_style)
            setattr(self, attr, le)
            h.addWidget(le)
            shear_vbox.addLayout(h)
        control_panel_layout.addWidget(shear_box)

        control_panel_layout.addWidget(QLabel("<b>[S/SHELL Thickness (mm)]</b>"))
        self.thickness_scroll = QScrollArea()
        self.thickness_scroll.setWidgetResizable(True)
        self.thickness_scroll.setMinimumHeight(200)
        self.scroll_content = QWidget()
        self.thickness_layout = QVBoxLayout(self.scroll_content)
        self.thickness_layout.setAlignment(Qt.AlignTop)
        self.thickness_scroll.setWidget(self.scroll_content)
        control_panel_layout.addWidget(self.thickness_scroll)

        self.btn_calc = QPushButton("2. Cross Section Analysis 🧮")
        self.btn_calc.setFixedHeight(50)
        self.btn_calc.setStyleSheet(
            "background-color: #28B463; color: white; font-weight: bold; font-size: 14px; margin-top: 5px;")
        self.btn_calc.clicked.connect(self.calculate_total_inertia)
        control_panel_layout.addWidget(self.btn_calc)

        eval_box = QFrame()
        eval_box.setStyleSheet("background: #FADBD8; border-radius: 5px; padding: 5px; margin-top: 10px;")
        eval_vbox = QVBoxLayout(eval_box)
        eval_vbox.addWidget(QLabel("<b>[Post-Calc Evaluation]</b>"))
        h_mat = QHBoxLayout()
        h_mat.addWidget(QLabel("Material:"))
        h_mat.addStretch()
        self.combo_material_type = QComboBox()
        self.combo_material_type.addItems(["Mild", "H.T", "H.T with BKT"])
        self.combo_material_type.setFixedWidth(self.field_width)
        h_mat.addWidget(self.combo_material_type)
        eval_vbox.addLayout(h_mat)

        self.btn_eval = QPushButton("3. STRENGTH Analysis ⛓️")
        self.btn_eval.setFixedHeight(40)
        self.btn_eval.setStyleSheet("background-color: #E67E22; color: white; font-weight: bold;")
        self.btn_eval.setEnabled(False)
        self.btn_eval.clicked.connect(self.evaluate_strength)
        eval_vbox.addWidget(self.btn_eval)
        control_panel_layout.addWidget(eval_box)

        main_layout.addWidget(control_panel)

        work_area = QWidget()
        work_layout = QVBoxLayout(work_area)
        viz_splitter = QSplitter(Qt.Horizontal)
        for i, title in enumerate(["[Cross Section View (1D)]", "[Closed Cells & Points]"]):
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.addWidget(QLabel(f"<b>{title}</b>"))
            fig = Figure()
            can = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(can, self))
            lay.addWidget(can, stretch=1)
            setattr(self, f"fig{i + 1}", fig)
            setattr(self, f"can{i + 1}", can)
            viz_splitter.addWidget(container)
        work_layout.addWidget(viz_splitter, stretch=7)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(300)
        self.result_box.setStyleSheet("font-family: 'Consolas'; font-size: 13px;")
        work_layout.addWidget(self.result_box)
        main_layout.addWidget(work_area, stretch=7)

        history_panel = QWidget()
        history_panel.setFixedWidth(300)
        history_layout = QVBoxLayout(history_panel)
        history_layout.setAlignment(Qt.AlignTop)
        history_layout.addWidget(QLabel("<b>[Saved Frames for Excel]</b>"))
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_content = QWidget()
        self.history_list_layout = QVBoxLayout(self.history_content)
        self.history_list_layout.setAlignment(Qt.AlignTop)
        self.history_scroll.setWidget(self.history_content)
        history_layout.addWidget(self.history_scroll, stretch=1)

        self.btn_save_frame = QPushButton("4. Add Frame to List 💾")
        self.btn_save_frame.setFixedHeight(40)
        self.btn_save_frame.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold;")
        self.btn_save_frame.clicked.connect(self.save_current_frame)
        self.btn_save_frame.setEnabled(False)
        history_layout.addWidget(self.btn_save_frame)

        self.btn_export_excel = QPushButton("5. Export All to Excel 📊")
        self.btn_export_excel.setFixedHeight(40)
        self.btn_export_excel.setStyleSheet(
            "background-color: #1E8449; color: white; font-weight: bold; margin-top: 5px;")
        self.btn_export_excel.clicked.connect(self.export_to_excel)
        history_layout.addWidget(self.btn_export_excel)

        main_layout.addWidget(history_panel)
        main_scroll.setWidget(main_container)
        self.setCentralWidget(main_scroll)

    def on_section_changed(self, text):
        self.combo_hull.setEnabled(text == "Discontinuous")

    def _extract_pts(self, e, scale):
        try:
            if e.dxftype() == 'LINE':
                return [(e.dxf.start.x * scale, e.dxf.start.y * scale), (e.dxf.end.x * scale, e.dxf.end.y * scale)]
            elif e.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                return [(p[0] * scale, p[1] * scale) for p in e.get_points()]
        except:
            return None
        return None

    def heal_1102_collinear(self, lines, threshold_gap=150.0):
        if not lines: return []
        bridges = []
        groups = {}
        for l in lines:
            c = list(l.coords)
            p1, p2 = np.array(c[0]), np.array(c[-1])
            v = p2 - p1
            L = np.linalg.norm(v)
            if L < 1e-6: continue
            a = np.degrees(np.arctan2(v[1], v[0])) % 180.0
            ak = round(a, 0)
            th = np.radians(ak)
            rho = round((-p1[0] * np.sin(th) + p1[1] * np.cos(th)) / 10.0) * 10.0
            key = (ak, rho)
            if key not in groups: groups[key] = []
            groups[key].append((l, p1, p2))
        for (ak, _), grp in groups.items():
            if len(grp) < 2: continue
            dv = np.array([np.cos(np.radians(ak)), np.sin(np.radians(ak))])
            segs = sorted([(np.dot(p1, dv), np.dot(p2, dv), p1, p2) for _, p1, p2 in grp],
                          key=lambda x: min(x[0], x[1]))
            for i in range(len(segs) - 1):
                pe = segs[i][2] if segs[i][0] > segs[i][1] else segs[i][3]
                pn = segs[i + 1][3] if segs[i + 1][0] > segs[i + 1][1] else segs[i + 1][2]
                g = np.linalg.norm(pn - pe)
                if 0.1 < g <= threshold_gap:
                    bridges.append(LineString([tuple(pe), tuple(pn)]))
        return lines + bridges

    def heal_1999_collinear(self, line_infos, threshold_gap=500.0):
        if not line_infos: return []
        bridges = []
        groups = {}
        for info in line_infos:
            l = info['line']
            c = list(l.coords)
            p1, p2 = np.array(c[0]), np.array(c[-1])
            v = p2 - p1
            L = np.linalg.norm(v)
            if L < 1e-6: continue
            a = np.degrees(np.arctan2(v[1], v[0])) % 180.0
            ak = round(a, 0)
            th = np.radians(ak)
            rho = round((-p1[0] * np.sin(th) + p1[1] * np.cos(th)) / 10.0) * 10.0
            key = (ak, rho)
            if key not in groups: groups[key] = []
            groups[key].append((info, p1, p2))
        for (ak, _), grp in groups.items():
            if len(grp) < 2: continue
            dv = np.array([np.cos(np.radians(ak)), np.sin(np.radians(ak))])
            segs = sorted([(np.dot(p1, dv), np.dot(p2, dv), p1, p2, info) for info, p1, p2 in grp],
                          key=lambda x: min(x[0], x[1]))
            for i in range(len(segs) - 1):
                pe = segs[i][2] if segs[i][0] > segs[i][1] else segs[i][3]
                pn = segs[i + 1][3] if segs[i + 1][0] > segs[i + 1][1] else segs[i + 1][2]
                g = np.linalg.norm(pn - pe)
                if 0.1 < g <= threshold_gap:
                    thk = (segs[i][4]['thickness'] + segs[i + 1][4]['thickness']) / 2.0
                    bridges.append({
                        'line': LineString([tuple(pe), tuple(pn)]),
                        'thickness': thk, 'type': '1999', 'is_bridge': True
                    })
        return line_infos + bridges

    def load_and_process_dxf(self):
        if self.is_processing: return
        fname, _ = QFileDialog.getOpenFileName(self, 'Select DXF File', '', 'DXF files (*.dxf)')
        if not fname: return
        fname = os.path.abspath(os.path.normpath(fname))
        self.reset_analysis_data()
        self.result_box.clear()
        self.current_dxf_path = fname
        try:
            scale = float(self.txt_scale.text())
            try:
                doc = ezdxf.readfile(fname, encoding='cp949')
            except:
                try:
                    doc = ezdxf.readfile(fname, encoding='utf-8')
                except:
                    doc = ezdxf.readfile(fname)

            msp = doc.modelspace()
            active_layers = {l.dxf.name for l in doc.layers if l.is_on() and not l.is_frozen()}
            t_1999, t_1204 = [], []
            t_layers = {"-1102": [], "157": [], "6001": [], "7001": [], "8001": [], "9001": []}

            for e in msp:
                layer = e.dxf.layer.strip()
                if layer not in active_layers: continue
                pts = self._extract_pts(e, scale)
                if not pts or len(pts) < 2: continue
                ls = LineString(pts)
                if layer == "1999":
                    t_1999.append(ls)
                elif layer == "-1204":
                    t_1204.append(ls)
                elif layer in t_layers:
                    t_layers[layer].append(ls)

            if t_1999:
                u_1999 = unary_union(t_1999)
                self.cx = u_1999.centroid.x
                self.cy_base = u_1999.bounds[1]
            else:
                self.cx = self.cy_base = 0.0

            shift = lambda ls: LineString([(p[0] - self.cx, p[1] - self.cy_base) for p in ls.coords])

            self.raw_1999_lines = [shift(ls) for ls in t_1999]
            self.lines_minus1204 = [shift(ls) for ls in t_1204]
            m_1999 = unary_union(self.raw_1999_lines)
            self.hull_centroid = m_1999.centroid

            cutters = [LineString([tuple(np.array(c.coords[0]) - 20), tuple(np.array(c.coords[-1]) + 20)])
                       for c in [shift(ls) for ls in t_1204]]
            split_res = split(m_1999, unary_union(cutters)) if cutters else m_1999
            pieces = list(split_res.geoms) if hasattr(split_res, 'geoms') else [split_res]
            self.left_1999_segments = sorted([g for g in pieces if g.centroid.x <= 0.1 and g.length > 0.1],
                                             key=lambda s: (-round(s.centroid.y, 2), s.centroid.x))

            self.lines_1102 = [shift(ls) for ls in t_layers["-1102"]]
            self.lines_1102_raw = list(self.lines_1102)
            self.lines_157 = [shift(ls) for ls in t_layers["157"]]
            self.lines_6001 = [shift(ls) for ls in t_layers["6001"]]
            self.lines_7001 = [shift(ls) for ls in t_layers["7001"]]
            self.lines_8001 = [shift(ls) for ls in t_layers["8001"]]
            self.lines_9001 = [shift(ls) for ls in t_layers["9001"]]
            self.refresh_ui()
            self.result_box.append(f"✅ Successfully loaded: {os.path.basename(fname)}")
        except Exception as e:
            self.result_box.setText(f"❌ Load Error Detailed:\n{traceback.format_exc()}")

    # =====================================================================
    # 메인 연산
    # =====================================================================

   def calculate_total_inertia(self):
        if self.is_processing: return
        self.is_processing = True
        self.btn_calc.setEnabled(False)
        self.btn_load.setEnabled(False)

        progress = QProgressDialog("Processing...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()

        def filter_short(lines, ml=100.0):
            return [l for l in lines if l.length >= ml]

        def remove_overlapping(lines, dt=10.0, at=5.0):
            lines = sorted(lines, key=lambda x: x.length, reverse=True)
            kept = []
            kept_meta = []
            for l in lines:
                c = list(l.coords)
                ps, pe = np.array(c[0]), np.array(c[-1])
                v = pe - ps
                ln = np.linalg.norm(v)
                if ln < 1e-6: continue
                ang = np.degrees(np.arctan2(v[1], v[0])) % 180
                dup = False
                for km in kept_meta:
                    ak, pk1, vk, lk = km['ang'], km['ps'], km['v'], km['ln']
                    if min(abs(ang - ak), 180 - abs(ang - ak)) > at: continue
                    vu = vk / lk
                    mid = (ps + pe) / 2.0
                    if np.linalg.norm(mid - (pk1 + np.dot(mid - pk1, vu) * vu)) > dt: continue
                    t1, t2 = np.dot(ps - pk1, vu), np.dot(pe - pk1, vu)
                    if min(lk, max(t1, t2)) - max(0, min(t1, t2)) > ln * 0.8:
                        dup = True;
                        break
                if not dup:
                    kept.append(l)
                    kept_meta.append({'ang': ang, 'ps': ps, 'v': v, 'ln': ln})
            return kept

        def split_by_slope(line, at=5.0):
            coords = list(line.coords)
            if len(coords) < 3: return [line]
            segs = []
            cur = [coords[0]]
            for i in range(1, len(coords) - 1):
                cur.append(coords[i])
                v1 = np.array(coords[i]) - np.array(coords[i - 1])
                v2 = np.array(coords[i + 1]) - np.array(coords[i])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6: continue
                a = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
                if a > at:
                    segs.append(LineString(cur))
                    cur = [coords[i]]
            cur.append(coords[-1])
            if len(cur) >= 2: segs.append(LineString(cur))
            return segs

        def match_pairs(lines, max_dist=100.0, angle_tol=20.0, overlap_tolerance=5.0):
            if not lines: return []
            ls_sorted = sorted(lines, key=lambda x: x.length)
            meta = []
            for l in ls_sorted:
                c = list(l.coords)
                ps, pe = np.array(c[0]), np.array(c[-1])
                v = pe - ps
                ln = np.linalg.norm(v)
                if ln < 1e-6:
                    meta.append(None);
                    continue
                minx, miny = min(ps[0], pe[0]), min(ps[1], pe[1])
                maxx, maxy = max(ps[0], pe[0]), max(ps[1], pe[1])
                meta.append({'ps': ps, 'pe': pe, 'v': v, 'ln': ln,
                             'unit': v / ln, 'ang': np.degrees(np.arctan2(v[1], v[0])) % 180,
                             'mid': (ps + pe) / 2.0,
                             'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy})
            used = {i: [] for i in range(len(ls_sorted))}
            pairs = []
            for i in range(len(ls_sorted)):
                if i % 50 == 0: QApplication.processEvents()
                if meta[i] is None: continue
                mi = meta[i]
                best_j, best_d, best_ov = -1, float('inf'), None
                mi_expand_minx = mi['minx'] - max_dist
                mi_expand_maxx = mi['maxx'] + max_dist
                mi_expand_miny = mi['miny'] - max_dist
                mi_expand_maxy = mi['maxy'] + max_dist

                for j in range(i + 1, len(ls_sorted)):
                    if meta[j] is None: continue
                    mj = meta[j]

                    if (mj['maxx'] < mi_expand_minx or mj['minx'] > mi_expand_maxx or
                            mj['maxy'] < mi_expand_miny or mj['miny'] > mi_expand_maxy):
                        continue

                    ad = min(abs(mi['ang'] - mj['ang']), 180 - abs(mi['ang'] - mj['ang']))
                    if ad > angle_tol: continue
                    proj_infinite = mj['ps'] + np.dot(mi['mid'] - mj['ps'], mj['unit']) * mj['unit']
                    d = np.linalg.norm(mi['mid'] - proj_infinite)
                    if d > max_dist: continue
                    t1 = np.dot(mi['ps'] - mj['ps'], mj['unit'])
                    t2 = np.dot(mi['pe'] - mj['ps'], mj['unit'])
                    ov_s, ov_e = max(0, min(t1, t2)), min(mj['ln'], max(t1, t2))
                    if (ov_e - ov_s) < mi['ln'] * 0.1: continue
                    is_blocked = False
                    for (us, ue) in used[j]:
                        if min(ov_e, ue) - max(ov_s, us) > overlap_tolerance:
                            is_blocked = True;
                            break
                    if is_blocked: continue
                    if d < best_d:
                        best_d, best_j, best_ov = d, j, (ov_s, ov_e)
                if best_j >= 0 and best_ov:
                    used[best_j].append(best_ov)
                    pairs.append((i, best_j, best_ov, best_d))
            return [(ls_sorted[i], ls_sorted[j], ov, dist) for i, j, ov, dist in pairs]

        def create_centerlines(pairs):
            result = []
            for short_line, long_line, (ov_s, ov_e), dist in pairs:
                cs = list(short_line.coords)
                cl_c = list(long_line.coords)
                ps1, ps2 = np.array(cs[0]), np.array(cs[-1])
                pl1 = np.array(cl_c[0])
                vl = np.array(cl_c[-1]) - pl1
                ll = np.linalg.norm(vl)
                if ll < 1e-6: continue
                vl_u = vl / ll
                mids = []
                for frac in np.linspace(0, 1, 5):
                    pt_s = ps1 + (ps2 - ps1) * frac
                    t = np.dot(pt_s - pl1, vl_u)
                    pt_l = pl1 + t * vl_u
                    mids.append(tuple((pt_s + pt_l) / 2.0))
                result.append({'line': LineString(mids), 'thickness': round(dist * 2) / 2.0})
            return result

        def create_continuous_stiffener_centerlines(pairs):
            long_line_map = {}
            for short_line, long_line, (ov_s, ov_e), dist in pairs:
                idx = id(long_line)
                if idx not in long_line_map:
                    long_line_map[idx] = {'long_line': long_line, 'shorts': [], 'dist': dist}
                long_line_map[idx]['shorts'].append(short_line)

            result = []
            for data in long_line_map.values():
                ll = data['long_line']
                dist = data['dist']
                shorts = data['shorts']

                cl_c = list(ll.coords)
                p1, p2 = np.array(cl_c[0]), np.array(cl_c[-1])
                v = p2 - p1
                length = np.linalg.norm(v)
                if length < 1e-6: continue
                vu = v / length

                sl = shorts[0]
                sc = list(sl.coords)
                ps_mid = (np.array(sc[0]) + np.array(sc[-1])) / 2.0
                pl_mid = (p1 + p2) / 2.0

                vec = ps_mid - pl_mid
                proj = np.dot(vec, vu) * vu
                perp = vec - proj
                perp_len = np.linalg.norm(perp)

                if perp_len < 1e-6:
                    n = np.array([-vu[1], vu[0]])
                else:
                    n = perp / perp_len

                offset = n * (dist / 2.0)
                new_coords = [tuple(np.array(pt) + offset) for pt in cl_c]
                result.append({'line': LineString(new_coords), 'thickness': round(dist * 2) / 2.0})

            return result

        def raycast_extend(centerlines, max_dist=100.0):
            extended_pts = []
            result = []
            bounds = [cl['line'].bounds for cl in centerlines]

            for i, cl in enumerate(centerlines):
                if i % 50 == 0: QApplication.processEvents()
                coords = list(cl['line'].coords)
                if len(coords) < 2:
                    result.append(cl);
                    continue
                for ei in [0, -1]:
                    p = np.array(coords[ei])
                    nb = 1 if ei == 0 else -2
                    v = p - np.array(coords[nb])
                    vn = np.linalg.norm(v)
                    if vn < 1e-6: continue
                    d = v / vn

                    conn = False
                    for j, o in enumerate(centerlines):
                        if i == j: continue
                        ob = bounds[j]
                        if ob[2] < p[0] - 5.0 or ob[0] > p[0] + 5.0 or ob[3] < p[1] - 5.0 or ob[1] > p[1] + 5.0:
                            continue
                        for op in [list(o['line'].coords)[0], list(o['line'].coords)[-1]]:
                            if np.linalg.norm(p - np.array(op)) < 5.0:
                                conn = True;
                                break
                        if conn: break
                    if conn: continue

                    ray = LineString([tuple(p), tuple(p + d * max_dist)])
                    rb = ray.bounds
                    bp, bd = None, max_dist
                    for j, o in enumerate(centerlines):
                        if i == j: continue
                        ob = bounds[j]
                        if rb[2] < ob[0] or rb[0] > ob[2] or rb[3] < ob[1] or rb[1] > ob[3]:
                            continue

                        inter = ray.intersection(o['line'])
                        if inter.is_empty: continue
                        pts = [inter] if inter.geom_type == 'Point' else \
                            list(inter.geoms) if inter.geom_type == 'MultiPoint' else []
                        for pt in pts:
                            dd = np.linalg.norm(np.array([pt.x, pt.y]) - p)
                            if 1e-3 < dd < bd:
                                bd = dd;
                                bp = (pt.x, pt.y)
                    if bp:
                        overshoot = 0.1
                        bp_o = (bp[0] + d[0] * overshoot, bp[1] + d[1] * overshoot)
                        if ei == 0:
                            coords[0] = bp_o
                        else:
                            coords[-1] = bp_o
                        extended_pts.append(np.array(bp_o))
                result.append({'line': LineString(coords), 'thickness': cl['thickness'],
                               'type': cl.get('type', 'internal')})
            return result, extended_pts

        def bridge_open_nodes(centerlines, extended_pts, max_gap=100.0):
            ep_map = defaultdict(list)
            for idx, cl in enumerate(centerlines):
                coords = list(cl['line'].coords)
                ks = (round(coords[0][0], 1), round(coords[0][1], 1))
                ke = (round(coords[-1][0], 1), round(coords[-1][1], 1))
                ep_map[ks].append({'idx': idx, 'side': 'start', 'pt': np.array(coords[0]),
                                   'thk': cl['thickness'], 'type': cl.get('type', '')})
                ep_map[ke].append({'idx': idx, 'side': 'end', 'pt': np.array(coords[-1]),
                                   'thk': cl['thickness'], 'type': cl.get('type', '')})
            open_eps = []
            for key, conns in ep_map.items():
                if len(conns) != 1: continue
                info = conns[0]
                is_ext = any(np.linalg.norm(info['pt'] - ep) < 1.0 for ep in extended_pts)
                if is_ext: continue
                open_eps.append(info)
            matched = set()
            bridges = []
            for i, e1 in enumerate(open_eps):
                if i in matched: continue
                bj, bd = -1, float('inf')
                p1 = e1['pt']
                for j, e2 in enumerate(open_eps):
                    if j <= i or j in matched or e1['idx'] == e2['idx']: continue
                    p2 = e2['pt']
                    if abs(p1[0] - p2[0]) > max_gap or abs(p1[1] - p2[1]) > max_gap:
                        continue
                    d = np.linalg.norm(p1 - p2)
                    if 0.1 < d <= max_gap and d < bd:
                        bd = d;
                        bj = j
                if bj >= 0:
                    e2 = open_eps[bj]
                    mid = tuple((e1['pt'] + e2['pt']) / 2.0)
                    bridges.append({'line': LineString([tuple(e1['pt']), mid]),
                                    'thickness': e1['thk'], 'type': 'bridge'})
                    bridges.append({'line': LineString([mid, tuple(e2['pt'])]),
                                    'thickness': e2['thk'], 'type': 'bridge'})
                    matched.add(i)
                    matched.add(bj)
            return centerlines + bridges, len(bridges) // 2

        def filter_redundant_nodes(centerlines):
            ep_map = defaultdict(list)
            for idx, cl in enumerate(centerlines):
                coords = list(cl['line'].coords)
                ks = (round(coords[0][0], 1), round(coords[0][1], 1))
                ke = (round(coords[-1][0], 1), round(coords[-1][1], 1))
                ep_map[ks].append((idx, 'start'))
                ep_map[ke].append((idx, 'end'))
            merged = set()
            mp = []
            for key, conns in ep_map.items():
                if len(conns) != 2: continue
                ia, sa = conns[0]
                ib, sb = conns[1]
                if ia == ib or ia in merged or ib in merged: continue
                ca, cb = centerlines[ia], centerlines[ib]
                if abs(ca['thickness'] - cb['thickness']) > 0.5: continue
                if ca.get('type') != cb.get('type'): continue
                mp.append((ia, sa, ib, sb))
                merged.add(ia)
                merged.add(ib)
            nl = []
            for ia, sa, ib, sb in mp:
                ca = list(centerlines[ia]['line'].coords)
                cb = list(centerlines[ib]['line'].coords)
                if sa == 'start': ca = ca[::-1]
                if sb == 'end': cb = cb[::-1]
                nl.append({'line': LineString(ca + cb[1:]), 'thickness': centerlines[ia]['thickness'],
                           'type': centerlines[ia].get('type', 'internal')})
            for idx, cl in enumerate(centerlines):
                if idx not in merged: nl.append(cl)
            return nl

        def split_all_lines_at_intersections(centerlines):
            all_line_geoms = [cl['line'] for cl in centerlines]
            bounds = [g.bounds for g in all_line_geoms]
            intersection_points = []

            for i in range(len(all_line_geoms)):
                if i % 50 == 0: QApplication.processEvents()
                b1 = bounds[i]
                for j in range(i + 1, len(all_line_geoms)):
                    b2 = bounds[j]
                    if b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]:
                        continue
                    try:
                        inter = all_line_geoms[i].intersection(all_line_geoms[j])
                        if inter.is_empty: continue
                        if inter.geom_type == 'Point':
                            intersection_points.append(inter)
                        elif inter.geom_type == 'MultiPoint':
                            intersection_points.extend(inter.geoms)
                        elif inter.geom_type == 'GeometryCollection':
                            for geom in inter.geoms:
                                if geom.geom_type == 'Point':
                                    intersection_points.append(geom)
                    except:
                        pass
            for g in all_line_geoms:
                intersection_points.append(Point(g.coords[0]))
                intersection_points.append(Point(g.coords[-1]))
            if not intersection_points:
                return centerlines
            unique_points = []
            for pt in intersection_points:
                if not unique_points or min(pt.distance(upt) for upt in unique_points) > 1e-3:
                    unique_points.append(pt)
            splitter = unary_union(unique_points)
            new_centerlines = []
            for cl in centerlines:
                line = cl['line']
                try:
                    snapped_line = snap(line, splitter, 0.05)
                    res = split(snapped_line, splitter)
                    geoms = list(res.geoms) if hasattr(res, 'geoms') else [res]
                    for geom in geoms:
                        new_centerlines.append({
                            'line': geom, 'thickness': cl['thickness'],
                            'type': cl.get('type', 'internal')
                        })
                except:
                    new_centerlines.append(cl)
            return new_centerlines

        try:
            self.raw_swbm = float(self.txt_swbm.text())
            self.raw_shear = float(self.txt_shear_v.text())

            y_mins = []
            input_thks = []
            for i, l_seg in enumerate(self.left_1999_segments):
                if progress.wasCanceled(): raise UserWarning("User canceled.")
                try:
                    t = float(self.shell_thickness_inputs[i].text())
                except:
                    t = 10.0
                input_thks.append(t)
                y_mins.append(l_seg.bounds[1] - t / 2.0)

            thickness_y_min = min(y_mins) if y_mins else 0.0

            c1102 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_1102_raw]
            c157 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_157]
            c6001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_6001]
            c7001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_7001]
            c8001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_8001]
            c9001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_9001]
            l1999s = [affinity.translate(l, yoff=-thickness_y_min) for l in self.left_1999_segments]

            progress.setLabelText("Step 1: Loading layers...")
            QApplication.processEvents()

            l1999f = []
            for i, ls in enumerate(l1999s):
                ta = input_thks[i] if i < len(input_thks) else 10.0
                l1999f.append({'line': ls, 'thickness': ta, 'type': '1999'})
                l1999f.append({'line': affinity.scale(ls, xfact=-1.0, origin=(0, 0)), 'thickness': ta, 'type': '1999'})

            progress.setLabelText("Step 2: Pre-filtering...")
            if progress.wasCanceled(): raise UserWarning("Canceled.")
            f1102 = remove_overlapping(filter_short(c1102, 100.0), dt=1.0)
            f157 = remove_overlapping(filter_short(c157, 100.0), dt=1.0)

            progress.setLabelText("Step 3: Healing -1102...")
            if progress.wasCanceled(): raise UserWarning("Canceled.")
            h1102 = self.heal_1102_collinear(f1102, threshold_gap=150)

            progress.setLabelText("Step 3.5: Healing 1999...")
            if progress.wasCanceled(): raise UserWarning("Canceled.")
            l1999f = self.heal_1999_collinear(l1999f, threshold_gap=500.0)

            progress.setLabelText("Step 4: Splitting 157 at slope changes...")
            s157 = []
            for l in f157:
                s157.extend(split_by_slope(l, at=5.0))
            s157 = [s for s in s157 if s.length >= 30.0]

            progress.setLabelText("Step 5: Pair matching...")
            QApplication.processEvents()
            p1102 = match_pairs(h1102, max_dist=100.0, angle_tol=20.0, overlap_tolerance=5.0)
            p157 = match_pairs(s157, max_dist=100.0, angle_tol=20.0, overlap_tolerance=5.0)

            progress.setLabelText("Step 6: Creating centerlines...")
            cl1102 = create_centerlines(p1102)
            for cl in cl1102: cl['type'] = '1102'
            cl157 = create_centerlines(p157)
            for cl in cl157: cl['type'] = '157'
            cl1999 = [l for l in l1999f if l['line'].length > 100.0 or l.get('is_bridge', False)]

            all_cl = cl1999 + cl1102 + cl157

            progress.setLabelText("Step 7: Ray-casting (≤100mm)...")
            if progress.wasCanceled(): raise UserWarning("Canceled.")
            all_cl, ext_pts = raycast_extend(all_cl, max_dist=100.0)

            progress.setLabelText("Step 8: Bridging open nodes (≤100mm)...")
            if progress.wasCanceled(): raise UserWarning("Canceled.")
            all_cl, bc = bridge_open_nodes(all_cl, ext_pts, max_gap=100.0)

            progress.setLabelText("Step 9: Filtering redundant nodes...")
            prev = -1
            for _ in range(10):
                nl = filter_redundant_nodes(all_cl)
                if len(nl) == prev: break
                prev = len(nl)
                all_cl = nl

            all_cl = split_all_lines_at_intersections(all_cl)

            progress.setLabelText("Step 10: Detecting closed loops (Cells)...")
            QApplication.processEvents()

            planarized_network = unary_union([cl['line'] for cl in all_cl])
            try:
                from shapely import set_precision
                planarized_network = set_precision(planarized_network, grid_size=0.01)
            except ImportError:
                pass

            raw_loops = list(polygonize(planarized_network))
            # [수정] 미세 오차 루프만 제외하고, 셀/슬릿 개수 1:1 매칭을 위해 1.0 이상 필터링 완화
            self.mesh_cells = [poly for poly in raw_loops if poly.area >= 1.0]

            self.shear_flow_centerlines = list(all_cl)

            # =============================================================
            # 종골재(Stiffener) 추출 및 1D Inertia 연산을 위해 centerlines 생성
            # =============================================================
            progress.setLabelText("Step 11: 종골재(Stiffener) 1D 추출 및 공간 인덱싱 병합 중...")
            QApplication.processEvents()

            raw_stiffs = c6001 + c7001 + c8001 + c9001
            stiff_f1 = filter_short(raw_stiffs, 20.0)
            stiff_f2 = remove_overlapping(stiff_f1, dt=1.0)

            stiff_s = []
            for l in stiff_f2: stiff_s.extend(split_by_slope(l, at=5.0))
            stiff_pairs = match_pairs(stiff_s, max_dist=50.0, angle_tol=20.0, overlap_tolerance=5.0)

            stiff_cl = create_continuous_stiffener_centerlines(stiff_pairs)
            for c in stiff_cl: c['type'] = 'stiffener'

            target_lines = all_cl
            healed_stiff_cl = []

            target_geoms = [o['line'] for o in target_lines]
            tree = STRtree(target_geoms)

            for cl in stiff_cl:
                coords = list(cl['line'].coords)
                for ei in [0, -1]:
                    p = np.array(coords[ei])
                    nb = 1 if ei == 0 else -2
                    v = p - np.array(coords[nb])
                    vn = np.linalg.norm(v)
                    if vn < 1e-6: continue
                    d = v / vn

                    p_point = Point(p)

                    res_open = tree.query(p_point.buffer(1e-3))
                    if len(res_open) > 0 and isinstance(res_open[0], (int, np.integer)):
                        close_1e3 = [target_geoms[i] for i in res_open]
                    else:
                        close_1e3 = list(res_open)

                    is_open = True
                    for cg in close_1e3:
                        if cg == cl['line']: continue
                        if cg.distance(p_point) < 1e-3:
                            is_open = False;
                            break

                    if is_open:
                        res_10 = tree.query(p_point.buffer(10.0))
                        if len(res_10) > 0 and isinstance(res_10[0], (int, np.integer)):
                            close_10 = [target_geoms[i] for i in res_10]
                        else:
                            close_10 = list(res_10)

                        close_lines = []
                        for cg in close_10:
                            if cg == cl['line']: continue
                            if cg.distance(p_point) <= 10.0:
                                close_lines.append(cg)

                        if close_lines:
                            ray = LineString([tuple(p), tuple(p + d * 30.0)])
                            bp, bd = None, 30.0
                            for cg in close_lines:
                                inter = ray.intersection(cg)
                                if not inter.is_empty:
                                    pts = [inter] if inter.geom_type == 'Point' else list(
                                        inter.geoms) if inter.geom_type == 'MultiPoint' else []
                                    for pt in pts:
                                        dd = np.linalg.norm(np.array([pt.x, pt.y]) - p)
                                        if dd < bd:
                                            bd = dd;
                                            bp = (pt.x, pt.y)
                            if bp and bd <= 10.0:
                                if ei == 0:
                                    coords[0] = bp
                                else:
                                    coords[-1] = bp
                healed_stiff_cl.append({'line': LineString(coords), 'thickness': cl['thickness'], 'type': cl['type']})

            all_combined_cl = all_cl + healed_stiff_cl
            all_combined_cl = split_all_lines_at_intersections(all_combined_cl)

            self.centerlines = all_combined_cl

            # =============================================================
            # 11.5: 1D Line 기반의 이너시아(Inertia) 연산
            # =============================================================
            progress.setLabelText("Step 11.5: 1D Line 기반 이너시아(Inertia) 연산...")
            QApplication.processEvents()

            a_1d_total = 0.0
            qx_1d_total = 0.0
            segments_1d = []

            for cl in self.centerlines:
                coords = list(cl['line'].coords)
                thk = cl.get('thickness', 10.0)
                if thk <= 0: thk = 10.0

                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    dx, dy = x2 - x1, y2 - y1
                    L = np.hypot(dx, dy)
                    if L < 1e-6: continue
                    a = L * thk
                    yc = (y1 + y2) / 2.0
                    a_1d_total += a
                    qx_1d_total += a * yc
                    segments_1d.append((a, yc, dy))

            if a_1d_total > 0:
                na_1d = qx_1d_total / a_1d_total
                ixx_1d = 0.0
                for a, yc, dy in segments_1d:
                    i_local = (a * (dy ** 2)) / 12.0
                    ixx_1d += i_local + a * ((yc - na_1d) ** 2)

                all_1d_y = [p[1] for cl in self.centerlines for p in cl['line'].coords]
                y_max_1d = max(all_1d_y)
                y_min_1d = min(all_1d_y)
                depth_1d = (y_max_1d - y_min_1d) * 1e-3
                na_bl_1d = na_1d * 1e-3
                dist_top_1d = y_max_1d - na_1d
                dist_btm_1d = na_1d - y_min_1d

                z_top_1d = (ixx_1d / dist_top_1d * 1e-9) if dist_top_1d != 0 else 0
                z_btm_1d = (ixx_1d / dist_btm_1d * 1e-9) if dist_btm_1d != 0 else 0
            else:
                na_1d = ixx_1d = depth_1d = na_bl_1d = z_top_1d = z_btm_1d = 0.0

            self.calc_total_area = a_1d_total
            self.calc_ixx = ixx_1d
            self.calc_depth = depth_1d
            self.calc_na_bl = na_1d
            self.calc_z_top = z_top_1d
            self.calc_z_btm = z_btm_1d

            z_act_top_mm3 = self.calc_ixx / dist_top_1d if dist_top_1d != 0 else 1e-9
            self.act_fb = (abs(self.raw_swbm) * 9.80665 * 1e6) / z_act_top_mm3

            # =============================================================
            # Step 12: Macro Edge 추출 (전체 단면 사용, 엄격한 degree=1 필터)
            # =============================================================
            progress.setLabelText("Step 12: 전체 단면도 기반 Macro Edge 추출 중...")
            QApplication.processEvents()

            # -------------------------------------------------------------
            # Half-section shear model 생성
            # - 좌현 반단면 기준: x <= CENTER_X_MAX 영역만 사용
            # - 중앙 밴드(-15 < x < 20)에 완전히 들어오는 1D 라인은 중앙 shared wall로 간주
            #   → 반단면 전용 복사본에서만 thickness 1/2 적용
            # - 원본 self.centerlines / 이너시아 계산 파이프라인은 유지
            # -------------------------------------------------------------
            CENTER_X_MIN = -15.0
            CENTER_X_MAX = 20.0
            HALF_EPS = 1e-6

            from shapely.geometry import box
            from shapely.geometry import LineString as ShapelyLineString

            def _extract_lines(geom):
                if geom.is_empty:
                    return []
                if geom.geom_type == 'LineString':
                    return [geom]
                if geom.geom_type == 'MultiLineString':
                    return list(geom.geoms)
                if geom.geom_type == 'GeometryCollection':
                    out = []
                    for g in geom.geoms:
                        if g.geom_type == 'LineString':
                            out.append(g)
                        elif g.geom_type == 'MultiLineString':
                            out.extend(list(g.geoms))
                    return out
                return []

            def _clone_cl_with_line(cl, line, thickness=None, extra=None):
                d = dict(cl)
                d['line'] = line
                if thickness is not None:
                    d['thickness'] = thickness
                if extra:
                    d.update(extra)
                return d

            all_bounds = [cl['line'].bounds for cl in self.shear_flow_centerlines if not cl['line'].is_empty]
            if not all_bounds:
                raise ValueError('No shear-flow centerlines available for half-section graph.')

            min_y_all = min(b[1] for b in all_bounds)
            min_x_all = min(b[0] for b in all_bounds)
            max_y_all = max(b[3] for b in all_bounds)
            pad = 10000.0
            half_clip_poly = box(min_x_all - pad, min_y_all - pad, CENTER_X_MAX, max_y_all + pad)

            full_shear_lines_original = list(self.shear_flow_centerlines)
            half_shear_lines = []

            for cl in full_shear_lines_original:
                line = cl['line']
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                xs = [p[0] for p in coords]
                min_x = min(xs)
                max_x = max(xs)

                # 중앙 밴드에 완전히 들어오는 선: 중앙 shared wall로 간주, 두께 1/2
                is_center_band_line = (min_x >= CENTER_X_MIN - HALF_EPS and max_x <= CENTER_X_MAX + HALF_EPS)
                if is_center_band_line:
                    half_shear_lines.append(_clone_cl_with_line(
                        cl,
                        ShapelyLineString(coords),
                        thickness=float(cl.get('thickness', 10.0)) * 0.5,
                        extra={
                            'is_half_model': True,
                            'is_center_shared_wall': True,
                            'original_thickness': float(cl.get('thickness', 10.0))
                        }
                    ))
                    continue

                # 완전히 우현 쪽이면 제외
                if min_x > CENTER_X_MAX + HALF_EPS:
                    continue

                # 좌현 쪽 또는 중심선을 가로지르는 선: x <= CENTER_X_MAX 영역만 사용
                try:
                    clipped = line.intersection(half_clip_poly)
                except Exception:
                    clipped = line

                for g in _extract_lines(clipped):
                    if g.length < 1e-6:
                        continue
                    half_shear_lines.append(_clone_cl_with_line(
                        cl,
                        g,
                        thickness=float(cl.get('thickness', 10.0)),
                        extra={
                            'is_half_model': True,
                            'is_center_shared_wall': False,
                            'original_thickness': float(cl.get('thickness', 10.0))
                        }
                    ))

            # 반단면 전단류 계산에는 half_shear_lines만 사용
            self.full_shear_flow_centerlines = full_shear_lines_original
            self.shear_flow_centerlines = half_shear_lines
            full_shear_lines = half_shear_lines

            # 반단면용 cell 재검출
            if full_shear_lines:
                half_planarized_network = unary_union([cl['line'] for cl in full_shear_lines])
                try:
                    from shapely import set_precision
                    half_planarized_network = set_precision(half_planarized_network, grid_size=0.01)
                except ImportError:
                    pass
                half_raw_loops = list(polygonize(half_planarized_network))
                self.mesh_cells = [poly for poly in half_raw_loops if poly.area >= 1.0]

            raw_edges = []
            for i, cl in enumerate(full_shear_lines):
                c = list(cl['line'].coords)
                if len(c) < 2: continue
                p1 = (round(c[0][0], 2), round(c[0][1], 2))
                p2 = (round(c[-1][0], 2), round(c[-1][1], 2))
                if p1 == p2: continue  # 점으로 수축된 선분 제외

                thk = cl.get('thickness', 10.0)
                raw_edges.append({'p1': p1, 'p2': p2, 'line': cl['line'], 'thk': thk})

            # Degree=1 맹장 엣지 반복 제거 (Pruning)
            changed = True
            safe_loop_count = 0
            while changed and safe_loop_count < 1000:
                safe_loop_count += 1
                if safe_loop_count % 10 == 0:
                    QApplication.processEvents()

                changed = False
                node_deg = defaultdict(int)
                for e in raw_edges:
                    node_deg[e['p1']] += 1
                    node_deg[e['p2']] += 1

                survivors = []
                for e in raw_edges:
                    if node_deg[e['p1']] <= 1 or node_deg[e['p2']] <= 1:
                        changed = True
                    else:
                        survivors.append(e)

                if not survivors and raw_edges:
                    survivors = raw_edges
                    break
                raw_edges = survivors

            deg = defaultdict(int)
            adj = defaultdict(list)
            for i, e in enumerate(raw_edges):
                deg[e['p1']] += 1
                deg[e['p2']] += 1
                adj[e['p1']].append(i)
                adj[e['p2']].append(i)

            strict_vertices = set()
            for pt, d in deg.items():
                if abs(pt[0]) <= 1e-2:
                    strict_vertices.add(pt)
                elif d >= 3:
                    strict_vertices.add(pt)
                elif d == 2:
                    e1_idx, e2_idx = adj[pt]
                    e1, e2 = raw_edges[e1_idx], raw_edges[e2_idx]

                    def get_vec_away(edge, point):
                        coords = list(edge['line'].coords)
                        p_start = (round(coords[0][0], 2), round(coords[0][1], 2))
                        if p_start == point:
                            v = np.array(coords[1]) - np.array(coords[0])
                        else:
                            v = np.array(coords[-2]) - np.array(coords[-1])
                        norm = np.linalg.norm(v)
                        return (v / norm) if norm > 1e-6 else np.array([0.0, 0.0])

                    v1 = get_vec_away(e1, pt)
                    v2 = get_vec_away(e2, pt)
                    cos_th = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_th))

                    if abs(angle - 180.0) > 2.0:
                        strict_vertices.add(pt)
                elif d == 1:
                    strict_vertices.add(pt)

            if not strict_vertices and raw_edges:
                strict_vertices.add(raw_edges[0]['p1'])

            self.cell_points = list(strict_vertices)
            self.graph_nodes = {nid: {'coord': pt} for nid, pt in enumerate(self.cell_points)}
            pt_to_nid = {pt: nid for nid, pt in enumerate(self.cell_points)}

            macro_edges = []
            visited_raw_edges = set()
            current_eid = 0
            node_edges = defaultdict(list)

            for start_pt in self.cell_points:
                for edge_idx in adj.get(start_pt, []):
                    if edge_idx in visited_raw_edges:
                        continue

                    current_pt = start_pt
                    current_edge_idx = edge_idx
                    path_coords = []
                    thk = raw_edges[current_edge_idx]['thk']

                    loop_safeguard = 0
                    while True:
                        loop_safeguard += 1
                        if loop_safeguard > 1000: break
                        if loop_safeguard % 100 == 0: QApplication.processEvents()

                        visited_raw_edges.add(current_edge_idx)
                        edge = raw_edges[current_edge_idx]

                        c = list(edge['line'].coords)
                        e_p1, e_p2 = edge['p1'], edge['p2']

                        if e_p2 == current_pt:
                            c = c[::-1]
                            nxt_pt = e_p1
                        else:
                            nxt_pt = e_p2

                        if not path_coords:
                            path_coords.extend(c)
                        else:
                            path_coords.extend(c[1:])

                        current_pt = nxt_pt

                        if current_pt in strict_vertices:
                            break

                        next_edges = [ei for ei in adj[current_pt] if ei not in visited_raw_edges]
                        if not next_edges:
                            break
                        current_edge_idx = next_edges[0]

                    macro_line = LineString(path_coords)
                    mid_pt = macro_line.interpolate(0.5, normalized=True)
                    cell_count = sum(1 for poly in self.mesh_cells if poly.exterior.distance(mid_pt) < 1.0)

                    # [수정] Slit 선정 개선 (공유 격벽 배제 및 외곽 센터라인 최우선 타겟화)
                    # Cost가 가장 높은 엣지가 Kruskal에서 최종 거부되어 Slit이 됨
                    if cell_count >= 2:
                        cost = -10000.0  # Shared Wall: 무조건 Tree Edge로 편입 (물리적 경로 보존)
                    elif cell_count == 1:
                        # Outer Wall: 최상단(Y 큼)이면서 정중앙(X=0)일수록 가장 높은 Cost를 받아 Slit이 됨
                        cost = mid_pt.y * 100.0 - abs(mid_pt.x) * 2.0
                    else:
                        cost = -20000.0  # 맹장 선분 등

                    sn = pt_to_nid[start_pt]
                    en = pt_to_nid.get(current_pt, sn)

                    # Self-loop(sn == en)도 그대로 macro_edges에 등록 (이후 Kruskal이 무조건 Slit으로 처리함)
                    edge_dict = {
                        'id': current_eid, 'start_node': sn, 'end_node': en,
                        'line': macro_line, 'length': macro_line.length,
                        'thickness': thk, 'cell_count': cell_count, 'cost': cost
                    }
                    macro_edges.append(edge_dict)
                    node_edges[sn].append(current_eid)
                    if sn != en:
                        node_edges[en].append(current_eid)
                    current_eid += 1

            self.graph_edges = macro_edges

            # =============================================================
            # =============================================================
            # Step 13: q0 계산 - Leaf -> Root 보존형 누적
            # =============================================================
            progress.setLabelText("Step 13: q0 leaf-to-root 보존형 누적 및 검증 중...")
            QApplication.processEvents()

            if not self.graph_nodes or not self.graph_edges:
                raise ValueError(
                    "Graph nodes/edges are empty. Macro edge extraction failed before q0 calculation."
                )

            # -------------------------------------------------------------
            # 1. reference slit 1개 선정: 외곽 + deck + centerline 우선
            # -------------------------------------------------------------
            max_y = max(node['coord'][1] for node in self.graph_nodes.values())
            min_y = min(node['coord'][1] for node in self.graph_nodes.values())
            tol_x = 100.0
            tol_y = 100.0

            candidates = []
            for e in self.graph_edges:
                if e.get('cell_count', 0) != 1:
                    continue
                mid_pt = e['line'].interpolate(0.5, normalized=True)
                if abs(mid_pt.x) > tol_x:
                    continue
                if mid_pt.y < max_y - tol_y:
                    continue
                score = abs(mid_pt.x) + 2.0 * abs(max_y - mid_pt.y)
                candidates.append((score, e))

            # fallback: centerline/deck 조건이 너무 엄격해서 후보가 없으면 outer edge 중 deck/centerline 근접 우선
            if not candidates:
                for e in self.graph_edges:
                    if e.get('cell_count', 0) != 1:
                        continue
                    mid_pt = e['line'].interpolate(0.5, normalized=True)
                    score = 10.0 * abs(mid_pt.x) + abs(max_y - mid_pt.y)
                    candidates.append((score, e))

            candidates.sort(key=lambda x: x[0])
            best_slit = candidates[0][1] if candidates else None
            self.slit_edge_ids = {best_slit['id']} if best_slit else set()

            for e in self.graph_edges:
                e['is_slit'] = (e['id'] in self.slit_edge_ids)
                e['S_accumulated'] = None
                e['bfs_direction'] = None

            # -------------------------------------------------------------
            # 2. Root Node 설정: leaf-to-root 누적의 최종 종착지
            #    선저/센터라인에 가까운 node를 우선 root로 둔다.
            # -------------------------------------------------------------
            self.root_node = min(
                self.graph_nodes.keys(),
                key=lambda nid: (
                    self.graph_nodes[nid]['coord'][1],
                    abs(self.graph_nodes[nid]['coord'][0])
                )
            )

            # -------------------------------------------------------------
            # 3. non-slit 그래프에서 spanning forest 생성
            # -------------------------------------------------------------
            adj_all = defaultdict(list)
            for e in self.graph_edges:
                if e['id'] in self.slit_edge_ids:
                    continue
                adj_all[e['start_node']].append(e)
                adj_all[e['end_node']].append(e)

            tree_edge_ids = set()
            parent_node = {}
            parent_edge = {}
            children = defaultdict(list)
            node_depth = {nid: float('inf') for nid in self.graph_nodes}
            component_roots = []
            visited_nodes = set()

            start_nodes = [self.root_node] + list(self.graph_nodes.keys())
            for start in start_nodes:
                if start in visited_nodes:
                    continue
                component_roots.append(start)
                visited_nodes.add(start)
                parent_node[start] = None
                parent_edge[start] = None
                node_depth[start] = 0
                queue = deque([start])

                while queue:
                    u = queue.popleft()
                    for e in adj_all[u]:
                        eid = e['id']
                        v = e['start_node'] if e['end_node'] == u else e['end_node']
                        if v not in visited_nodes:
                            visited_nodes.add(v)
                            parent_node[v] = u
                            parent_edge[v] = eid
                            children[u].append(v)
                            node_depth[v] = node_depth[u] + 1
                            tree_edge_ids.add(eid)
                            queue.append(v)

            # -------------------------------------------------------------
            # 4. 각 edge 자체에서 S_local = ∫ y t ds 직접 적분
            # -------------------------------------------------------------
            for edge in self.graph_edges:
                geom = edge['line']
                coords = list(geom.coords)
                if len(coords) < 2:
                    edge['sample_s'] = np.array([0.0], dtype=float)
                    edge['sample_pts'] = [(coords[0][0], coords[0][1])] if coords else [(0.0, 0.0)]
                    edge['S_local_array'] = np.array([0.0], dtype=float)
                    edge['S_local'] = 0.0
                    continue

                s_vals = [0.0]
                pts = [(coords[0][0], coords[0][1])]
                current_L = 0.0
                for i in range(1, len(coords)):
                    p1 = np.array(coords[i - 1], dtype=float)
                    p2 = np.array(coords[i], dtype=float)
                    seg_L = np.linalg.norm(p2 - p1)
                    if seg_L < 1e-9:
                        continue
                    current_L += seg_L
                    s_vals.append(current_L)
                    pts.append((coords[i][0], coords[i][1]))

                edge['sample_s'] = np.array(s_vals, dtype=float)
                edge['sample_pts'] = pts

                y_prime = np.array([pt[1] for pt in pts], dtype=float) - self.calc_na_bl
                t = float(edge.get('thickness', 0.0))

                S_loc = np.zeros(len(s_vals), dtype=float)
                curr = 0.0
                for k in range(1, len(s_vals)):
                    y_avg = 0.5 * (y_prime[k - 1] + y_prime[k])
                    ds_val = s_vals[k] - s_vals[k - 1]
                    curr += y_avg * t * ds_val
                    S_loc[k] = curr

                edge['S_local_array'] = S_loc
                edge['S_local'] = float(curr)

            # -------------------------------------------------------------
            # 5. Leaf -> Root 보존형 S 누적
            #    leaf에서 S=0으로 시작하고, junction에서는 child contribution을 합산한다.
            # -------------------------------------------------------------
            s_at_node = {nid: 0.0 for nid in self.graph_nodes}
            tree_edge_by_id = {e['id']: e for e in self.graph_edges}

            nodes_by_depth = sorted(
                self.graph_nodes.keys(),
                key=lambda nid: node_depth.get(nid, 0),
                reverse=True
            )

            for nid in nodes_by_depth:
                pnode = parent_node.get(nid)
                peid = parent_edge.get(nid)
                if pnode is None or peid is None:
                    continue

                e = tree_edge_by_id[peid]
                S_start = s_at_node[nid]

                # traversal: nid -> pnode
                if e['start_node'] == nid and e['end_node'] == pnode:
                    # geometry 방향과 traversal 방향 동일
                    e['S_accumulated'] = S_start + e['S_local_array']
                    e['bfs_direction'] = 'forward'
                    S_exit = float(e['S_accumulated'][-1])
                elif e['end_node'] == nid and e['start_node'] == pnode:
                    # geometry 방향과 traversal 방향 반대
                    # geometry order에서 start(parent) 값이 exit, end(child) 값이 S_start
                    e['S_accumulated'] = S_start + (e['S_local'] - e['S_local_array'])
                    e['bfs_direction'] = 'reverse'
                    S_exit = float(e['S_accumulated'][0])
                else:
                    continue

                # junction 보존: parent node로 들어오는 S contribution을 합산
                s_at_node[pnode] = s_at_node.get(pnode, 0.0) + S_exit

            # -------------------------------------------------------------
            # 6. tree 밖 edge 복원 및 loop residual 저장
            # -------------------------------------------------------------
            loop_residuals = []
            for e in self.graph_edges:
                if e.get('S_accumulated') is not None:
                    continue

                sn = e['start_node']
                en = e['end_node']
                s_sn = s_at_node.get(sn)
                s_en = s_at_node.get(en)

                # 더 깊은 node에서 root 방향으로 적분하는 것을 우선
                if s_sn is not None and s_en is not None:
                    if node_depth.get(sn, 0) >= node_depth.get(en, 0):
                        e['S_accumulated'] = s_sn + e['S_local_array']
                        e['bfs_direction'] = 'forward'
                        residual = (s_sn + e['S_local']) - s_en
                    else:
                        e['S_accumulated'] = s_en + (e['S_local'] - e['S_local_array'])
                        e['bfs_direction'] = 'reverse'
                        residual = (s_en + e['S_local']) - s_sn
                    loop_residuals.append((e['id'], float(residual)))
                elif s_sn is not None:
                    e['S_accumulated'] = s_sn + e['S_local_array']
                    e['bfs_direction'] = 'forward'
                elif s_en is not None:
                    e['S_accumulated'] = s_en + (e['S_local'] - e['S_local_array'])
                    e['bfs_direction'] = 'reverse'
                else:
                    e['S_accumulated'] = None
                    e['bfs_direction'] = None

            # -------------------------------------------------------------
            # 7. q0 산출
            # -------------------------------------------------------------
            V_shear = abs(self.raw_shear) * 1000.0 * 9.80665
            Ixx = self.calc_ixx
            missing_s_edges = []
            max_q0_val = 0.0

            for edge in self.graph_edges:
                if edge.get('S_accumulated') is not None:
                    if Ixx > 1e-6:
                        q0 = (V_shear * edge['S_accumulated']) / Ixx
                    else:
                        q0 = np.zeros(len(edge['sample_s']), dtype=float)
                    edge['q0'] = q0
                    edge['q0_geom'] = q0
                    edge['q0_mean'] = float(np.mean(q0)) if len(q0) else 0.0
                    max_q0_val = max(max_q0_val, float(np.max(np.abs(q0))))
                else:
                    missing_s_edges.append(edge['id'])
                    edge['q0'] = np.zeros(len(edge['sample_s']), dtype=float)
                    edge['q0_geom'] = edge['q0']
                    edge['q0_mean'] = 0.0

            # -------------------------------------------------------------
            # Step 14: B matrix 기반 qc 계산
            # -------------------------------------------------------------
            progress.setLabelText("Step 14: B matrix 기반 qc 계산 중...")
            QApplication.processEvents()

            edge_index_map = {e['id']: i for i, e in enumerate(self.graph_edges)}
            n_cells = len(self.mesh_cells)
            n_edges = len(self.graph_edges)

            for e in self.graph_edges:
                e['adjacent_cell_ids'] = []
                mid_pt = e['line'].interpolate(0.5, normalized=True)
                for cid, poly in enumerate(self.mesh_cells):
                    if poly.exterior.distance(mid_pt) < 2.0:
                        e['adjacent_cell_ids'].append(cid)

            def edge_unit_vector(edge):
                c = list(edge['line'].coords)
                p1 = np.array(c[0], dtype=float)
                p2 = np.array(c[-1], dtype=float)
                v = p2 - p1
                n = np.linalg.norm(v)
                return v / n if n > 1e-12 else np.array([0.0, 0.0])

            def cell_edge_sign(poly, edge):
                coords = list(poly.exterior.coords)
                ev = edge_unit_vector(edge)
                if np.linalg.norm(ev) < 1e-12:
                    return 0.0
                edge_mid = edge['line'].interpolate(0.5, normalized=True)
                best_dist = float('inf')
                best_vec = None
                for i in range(len(coords) - 1):
                    p1 = np.array(coords[i], dtype=float)
                    p2 = np.array(coords[i + 1], dtype=float)
                    seg = LineString([tuple(p1), tuple(p2)])
                    d = seg.distance(edge_mid)
                    if d < best_dist:
                        best_dist = d
                        v = p2 - p1
                        n = np.linalg.norm(v)
                        best_vec = v / n if n > 1e-12 else None
                if best_vec is None:
                    return 0.0
                return 1.0 if np.dot(ev, best_vec) >= 0 else -1.0

            B = np.zeros((n_cells, n_edges), dtype=float)
            for e in self.graph_edges:
                idx = edge_index_map[e['id']]
                for cid in e.get('adjacent_cell_ids', []):
                    B[cid, idx] = cell_edge_sign(self.mesh_cells[cid], e)
            self.B_matrix = B

            q0_edge = np.zeros(n_edges, dtype=float)
            w = np.zeros(n_edges, dtype=float)
            for e in self.graph_edges:
                idx = edge_index_map[e['id']]
                q0_edge[idx] = float(e.get('q0_mean', 0.0))
                L = float(e.get('length', 0.0))
                t = float(e.get('thickness', 0.0))
                w[idx] = L / t if t > 1e-12 else 0.0

            Wq0 = w * q0_edge
            M = B @ (w[:, None] * B.T)
            r = -B @ Wq0
            M_reg = M + 1e-9 * np.eye(n_cells)

            try:
                qc = np.linalg.solve(M_reg, r)
            except np.linalg.LinAlgError:
                qc = np.linalg.lstsq(M_reg, r, rcond=None)[0]

            self.qc = qc
            q_corr_edge = B.T @ qc
            q_final_edge = q0_edge + q_corr_edge

            for e in self.graph_edges:
                idx = edge_index_map[e['id']]
                e['q0_mean'] = float(q0_edge[idx])
                e['qc_correction'] = float(q_corr_edge[idx])
                e['q_final_mean'] = float(q_final_edge[idx])
                if 'q0' in e and e['q0'] is not None:
                    e['q_final'] = e['q0'] + q_corr_edge[idx]
                    e['q_final_geom'] = e['q_final']
                else:
                    e['q_final'] = np.array([q_final_edge[idx]])
                    e['q_final_geom'] = e['q_final']

            qc_residual = M @ qc - r
            qc_residual_norm = float(np.linalg.norm(qc_residual))
            cell_balance = B @ (w * q_final_edge)
            max_cell_balance = float(np.max(np.abs(cell_balance))) if len(cell_balance) else 0.0
            max_qc_val = float(np.max(np.abs(qc))) if len(qc) else 0.0
            max_qcorr_val = float(np.max(np.abs(q_corr_edge))) if len(q_corr_edge) else 0.0
            max_qfinal_val = float(np.max(np.abs(q_final_edge))) if len(q_final_edge) else 0.0

            # -------------------------------------------------------------
            # 전단응력 tau = q_final / t 계산 및 최대 지점 탐색
            # q [N/mm] / t [mm] = N/mm^2 = MPa
            # -------------------------------------------------------------
            max_tau_abs = 0.0
            max_tau_info = {'edge_id': None, 'tau_abs': 0.0, 'tau_signed': 0.0, 'thickness': 0.0, 'point': None,
                            'x': 0.0, 'y': 0.0}

            for e in self.graph_edges:
                t_edge = float(e.get('thickness', 0.0))
                if t_edge <= 1e-12:
                    tau_arr = np.zeros_like(e.get('q_final', np.array([0.0])), dtype=float)
                else:
                    tau_arr = np.array(e.get('q_final', np.array([0.0])), dtype=float) / t_edge
                e['tau_final'] = tau_arr
                e['tau_final_mean'] = float(np.mean(tau_arr)) if len(tau_arr) else 0.0
                if len(tau_arr) > 0:
                    idx_tau = int(np.argmax(np.abs(tau_arr)))
                    tau_signed = float(tau_arr[idx_tau])
                    tau_abs = abs(tau_signed)
                    if tau_abs > max_tau_abs:
                        max_tau_abs = tau_abs
                        pts = e.get('sample_pts', [])
                        if pts and idx_tau < len(pts):
                            pt = pts[idx_tau]
                        else:
                            mp = e['line'].interpolate(0.5, normalized=True)
                            pt = (mp.x, mp.y)
                        max_tau_info = {'edge_id': e['id'], 'tau_abs': tau_abs, 'tau_signed': tau_signed,
                                        'thickness': t_edge, 'point': pt, 'x': pt[0], 'y': pt[1]}
            self.max_tau_info = max_tau_info

            # -------------------------------------------------------------
            # 간결 검증 리포트
            # -------------------------------------------------------------
            val_warnings = []
            if missing_s_edges:
                val_warnings.append(f"[WARN] S calculation missing for edges: {missing_s_edges}")
            if max_q0_val > 10000:
                val_warnings.append(f"[WARN] Excessive q0 value detected: {max_q0_val:.2f} N/mm")
            if np.linalg.matrix_rank(M) < n_cells:
                val_warnings.append("[WARN] BWB^T matrix is rank deficient")

            visited_count = len(visited_nodes)
            total_nodes = len(self.graph_nodes)
            root_final_S = s_at_node.get(self.root_node, 0.0)

  
            # 출력창(결과) 리포트 업데이트
            # =============================================================
            res = f"--- Applied Loads ---\n"
            res += f"S.W.B.M          : {self.raw_swbm:,.2f} tm\n"
            res += f"Shear Force      : {self.raw_shear:,.2f} t\n\n"

            res += f"--- Geometric Properties (1D Line Based) ---\n"
            res += f"Total Area       : {self.calc_total_area / 100.0:>10,.2f} cm^2\n"
            res += f"I_xx(m^4)        : {self.calc_ixx * 1e-12:,.6e}\n"
            res += f"Depth(m)         : {self.calc_depth:.3f}\n"
            res += f"Position of N.A from B.L(m) : {self.calc_na_bl * 1e-3:.3f}\n"
            res += f"Zact_btm         : {self.calc_z_btm:,.4f} m^3\n"
            res += f"Zact_top         : {self.calc_z_top:,.4f} m^3\n\n"

            res += f"--- 1D Extraction & Graph Results ---\n"
            res += f"Centerlines      : {len(self.centerlines)} (Incl. Stiffeners)\n"
            res += f"Shear Edges      : {len(self.shear_flow_centerlines)} (No Stiffeners)\n"
            res += f"Detected Cells   : {len(self.mesh_cells)} (Closed Loops)\n"
            res += f"Strict Vertices  : {len(self.graph_nodes)} (Only true corners/junctions)\n"
            res += f"Macro Graph Edges: {len(self.graph_edges)} (Merged Paths)\n"
            res += f"Tree Edges       : {len([e for e in self.graph_edges if not e['is_slit']])}\n"
            res += f"Slit Edges       : {len(self.slit_edge_ids)} (BFS Outward 적용)\n\n"

            res += f"--- Validation Info ---\n"
            res += f"Root Node ID     : {self.root_node} (Slit 인접, Centerline 근처)\n"
            res += f"Root Initial S   : {root_residual:.4e} (Must be 0.0)\n"
            res += f"Visited Nodes    : {visited_count} / {total_nodes}\n"
            res += f"Missing S Edges  : {len(missing_s_edges)}\n"
            res += f"Loop Residuals   : {len(loop_residuals)} edges (for qc correction)\n"
            res += f"Max |q0|         : {max_q0_val:.2f} N/mm\n"
            res += self.qc_validation_report

            if val_warnings:
                res += "\n--- WARNINGS ---\n"
                for w in val_warnings:
                    res += f"{w}\n"
                    print(w)

            self.base_report = res
            self.result_box.setText(self.base_report)
            self.is_calculated = True
            self.btn_eval.setEnabled(True)

            progress.setLabelText("Rendering...")
            progress.setMaximum(0)
            QApplication.processEvents()
            self.refresh_ui()

            self.refresh_ui()

            # 기존 팝업 제거
            for dlg in getattr(self, "debug_dialogs", []):
                try:
                    dlg.close()
                except:
                    pass

            self.debug_dialogs = []

            # 새 팝업 생성
            if self.graph_edges:
                self.dialog_q0 = ShearFlowSingleDialog(
                    self.graph_edges, self.graph_nodes, self.root_node, mode="q0", parent=self
                )
                self.dialog_qc = ShearFlowSingleDialog(
                    self.graph_edges, self.graph_nodes, self.root_node, mode="qc", parent=self
                )
                self.dialog_qfinal = ShearFlowSingleDialog(
                    self.graph_edges, self.graph_nodes, self.root_node, mode="q_final", parent=self
                )

                self.dialog_q0.show()
                self.dialog_qc.show()
                self.dialog_qfinal.show()

                self.debug_dialogs = [self.dialog_q0, self.dialog_qc, self.dialog_qfinal]


        except UserWarning as uw:
            self.result_box.setText(f"⚠️ {str(uw)}")
        except Exception as e:
            self.result_box.setText(f"❌ Calculation Error:\n{str(e)}\n\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Fatal Error", f"예기치 못한 에러가 발생했습니다.\n\n{str(e)}")
        finally:
            progress.close()
            self.is_processing = False
            self.btn_calc.setEnabled(True)
            self.btn_load.setEnabled(True)

    def evaluate_strength(self):
        if not self.is_calculated: return
        k = float(self.txt_grade_k.text())
        self.allow_fs = 105 / k
        if self.combo_section.currentText() == "Continuous":
            self.allow_fb = 143 / k
        else:
            h_t = self.combo_hull.currentText()
            m_t = self.combo_material_type.currentText()
            table = {"S/H": {"Mild": 60, "H.T": 75, "H.T with BKT": 112},
                     "D/H": {"Mild": 112, "H.T": 150, "H.T with BKT": 157}}
            self.allow_fb = table.get(h_t, {}).get(m_t, 60)

        final_res = self.base_report + "\n--- Strength Check & Utilization ---\n"
        final_res += f"Material         : {self.combo_material_type.currentText()}\n\n"
        final_res += f"Bending Stress   : {self.act_fb:.2f} / {self.allow_fb:.2f} N/mm2 ({self.act_fb / self.allow_fb * 100:.1f}%) [{'PASS' if self.act_fb <= self.allow_fb else 'FAIL'}]\n"
        final_res += f"Shear Stress     : (Pending - 순환 전단류 q_c 산출 대기중)\n"
        self.result_box.setText(final_res)
        self.btn_save_frame.setEnabled(True)

    def save_current_frame(self):
        frame_name, ok = QInputDialog.getText(self, "Save Frame", "Enter Frame Name:",
                                              QLineEdit.EchoMode.Normal, "FR.")
        if not ok or not frame_name.strip(): return
        img_sec = io.BytesIO()
        self.fig1.savefig(img_sec, format='png', bbox_inches='tight', dpi=150)
        img_shr = io.BytesIO()
        self.fig2.savefig(img_shr, format='png', bbox_inches='tight', dpi=150)
        data = {
            "Frame": frame_name.strip(), "SWBM": self.raw_swbm, "Depth": round(self.calc_depth, 2),
            "NA": round(self.calc_na_bl * 1e-3, 2), "Ixx": round(self.calc_ixx * 1e-12, 2),
            "Grade_K": float(self.txt_grade_k.text()),
            "Z_btm": round(self.calc_z_btm, 2), "Z_top": round(self.calc_z_top, 2),
            "Act_FB": round(self.act_fb, 1), "Allow_FB": round(self.allow_fb, 1),
            "Shear": self.raw_shear,
            "Pos_Shear": f"S{self.max_shell_q_idx + 1}" if self.max_shell_q_idx != -1 else "N/A",
            "Thk": self.max_shell_thk, "Unit_q": self.q_per_v * 1e3,
            "Act_FS": round(self.act_fs, 1), "Allow_FS": round(self.allow_fs, 1),
            "ImgSec": img_sec.getvalue(), "ImgShr": img_shr.getvalue()
        }
        self.saved_frames_data.append(data)
        self.update_history_list_ui()

    def update_history_list_ui(self):
        for i in reversed(range(self.history_list_layout.count())):
            w = self.history_list_layout.itemAt(i).widget()
            if w: w.deleteLater()
        for i, data in enumerate(self.saved_frames_data):
            f = QFrame()
            f.setStyleSheet("background: white; border: 1px solid #BDC3C7; border-radius: 4px; margin-bottom: 2px;")
            l = QHBoxLayout(f)
            l.setContentsMargins(5, 5, 5, 5)
            lbl_name = QLabel(f"📝 {data['Frame']}")
            lbl_name.setStyleSheet("font-weight:bold; border:none;")
            l.addWidget(lbl_name)
            l.addStretch()
            btn_edit = QPushButton("✏️")
            btn_edit.setFixedSize(24, 24)
            btn_edit.setStyleSheet("border: none; background: transparent;")
            btn_edit.clicked.connect(lambda chk=False, idx=i: self.rename_frame(idx))
            l.addWidget(btn_edit)
            btn_up = QPushButton("⬆️")
            btn_up.setFixedSize(24, 24)
            btn_up.setStyleSheet("border: none; background: transparent;")
            btn_up.setEnabled(i > 0)
            btn_up.clicked.connect(lambda chk=False, idx=i: self.move_frame_up(idx))
            l.addWidget(btn_up)
            btn_down = QPushButton("⬇️")
            btn_down.setFixedSize(24, 24)
            btn_down.setStyleSheet("border: none; background: transparent;")
            btn_down.setEnabled(i < len(self.saved_frames_data) - 1)
            btn_down.clicked.connect(lambda chk=False, idx=i: self.move_frame_down(idx))
            l.addWidget(btn_down)
            btn_del = QPushButton("❌")
            btn_del.setFixedSize(24, 24)
            btn_del.setStyleSheet("border: none; background: transparent;")
            btn_del.clicked.connect(lambda chk=False, idx=i: self.delete_saved_frame(idx))
            l.addWidget(btn_del)
            self.history_list_layout.addWidget(f)

    def rename_frame(self, idx):
        old = self.saved_frames_data[idx]['Frame']
        n, ok = QInputDialog.getText(self, "Rename", "Frame Name:", QLineEdit.Normal, old)
        if ok and n.strip():
            self.saved_frames_data[idx]['Frame'] = n.strip()
            self.update_history_list_ui()

    def move_frame_up(self, idx):
        if idx > 0:
            self.saved_frames_data[idx - 1], self.saved_frames_data[idx] = \
                self.saved_frames_data[idx], self.saved_frames_data[idx - 1]
            self.update_history_list_ui()

    def move_frame_down(self, idx):
        if idx < len(self.saved_frames_data) - 1:
            self.saved_frames_data[idx + 1], self.saved_frames_data[idx] = \
                self.saved_frames_data[idx], self.saved_frames_data[idx + 1]
            self.update_history_list_ui()

    def delete_saved_frame(self, index):
        if 0 <= index < len(self.saved_frames_data):
            del self.saved_frames_data[index]
            self.update_history_list_ui()

    def apply_outer_border(self, ws, min_r, max_r, min_c, max_c):
        from openpyxl.styles import Border, Side
        thick = Side(border_style="medium", color="000000")
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                cell = ws.cell(row=r, column=c)
                b = cell.border
                cell.border = Border(top=thick if r == min_r else b.top,
                                     bottom=thick if r == max_r else b.bottom,
                                     left=thick if c == min_c else b.left,
                                     right=thick if c == max_c else b.right)

    def export_to_excel(self):
        if not self.saved_frames_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "Section_Analysis_Report.xlsx",
                                              "Excel Files (*.xlsx)")
        if not path: return
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.drawing.image import Image as XlImage
            from openpyxl.worksheet.pagebreak import Break
            wb = Workbook()
            ws = wb.active
            ws.title = "Strength Result"
            f_title = Font(name='돋움', size=14, bold=True, underline='double')
            f_11b = Font(name='돋움', size=11, bold=True)
            f_11n = Font(name='돋움', size=11)
            f_10n = Font(name='돋움', size=10)
            f_18b = Font(name='돋움', size=18, bold=True)
            a_c = Alignment(horizontal='center', vertical='center', wrap_text=True)
            fill_y = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')
            fill_g = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
            thin_s = Side(border_style="thin", color="000000")
            b_all = Border(top=thin_s, left=thin_s, right=thin_s, bottom=thin_s)

            ws.merge_cells('A1:I4')
            ws.cell(1, 1, "H0000 Conclusion of Scantling Check for Partial Floating Condition").font = f_title
            ws.cell(1, 1).alignment = a_c
            ws.merge_cells('J1:J4')
            ws.cell(1, 10, "검 토").alignment = a_c
            ws.cell(1, 11, "PART장").alignment = a_c
            ws.cell(1, 12, "보임과장").alignment = a_c
            ws.merge_cells('K2:K3')
            ws.merge_cells('L2:L3')
            ws.row_dimensions[2].height = ws.row_dimensions[3].height = 30
            today = datetime.datetime.now().strftime("%Y.%m.%d.")
            ws.cell(4, 11, today).alignment = a_c
            ws.cell(4, 12, today).alignment = a_c
            for r in range(1, 5):
                for c in range(1, 13): ws.cell(r, c).border = b_all

            ws.cell(5, 1, "*Allowable stress").font = f_11b
            ws.merge_cells('A6:A8');
            ws.cell(6, 1, "Bending Stress")
            ws.merge_cells('B6:C6');
            ws.cell(6, 2, "Continuous Section")
            ws.merge_cells('D6:L6');
            ws.cell(6, 4, "143/k for bending stress")
            ws.merge_cells('B7:C8');
            ws.cell(7, 2, "Discontinuous Section")
            ws.merge_cells('D7:F7');
            ws.cell(7, 4, "60 N/mm² for S/H (Mild)")
            ws.merge_cells('G7:I7');
            ws.cell(7, 7, "75 N/mm² for S/H (H.T)")
            ws.merge_cells('J7:L7');
            ws.cell(7, 10, "112 N/mm² for S/H (H.T with BKT)")
            ws.merge_cells('D8:F8');
            ws.cell(8, 4, "112 N/mm² for D/H (Mild)")
            ws.merge_cells('G8:I8');
            ws.cell(8, 7, "150 N/mm² for D/H (H.T)")
            ws.merge_cells('J8:L8');
            ws.cell(8, 10, "157 N/mm² for D/H (H.T with BKT)")
            ws.merge_cells('A9:C9');
            ws.cell(9, 1, "Shear Stress")
            ws.merge_cells('D9:L9');
            ws.cell(9, 4, "105/k for shear stress")
            for r in range(6, 10):
                for c in range(1, 13):
                    ws.cell(r, c).border = b_all
                    ws.cell(r, c).alignment = a_c

            row_idx = 11
            for data in self.saved_frames_data:
                headers = ["Position", "S.W.B.M\n( t·m )", "Depth(m)", "Position of\nN.A from B.L(m)",
                           "I_xx(m⁴)", "Grade (k)", "Zact_at btm\n(m³)", "Zact_at top\n(m³)",
                           "Bending stress_at top\n(N/mm²)", "Allowable stress\n(N/mm²)",
                           "Percentage\n(%)", "Result"]
                for i, h in enumerate(headers, 1):
                    cell = ws.cell(row_idx, i, h)
                    cell.alignment = a_c;
                    cell.font = f_10n;
                    cell.border = b_all

                r2 = [data['Frame'], data['SWBM'], data['Depth'], data['NA'], data['Ixx'],
                      data['Grade_K'], data['Z_btm'], data['Z_top'], data['Act_FB'], data['Allow_FB'],
                      data['Act_FB'] / data['Allow_FB'] if data['Allow_FB'] > 0 else 0,
                      "OK" if data['Act_FB'] <= data['Allow_FB'] else "NG"]
                for i, v in enumerate(r2, 1):
                    cell = ws.cell(row_idx + 1, i, v)
                    cell.alignment = a_c;
                    cell.border = b_all;
                    cell.font = f_11n
                    if i in [2, 3, 4, 5, 6]: cell.fill = fill_y
                    if i in [9, 10, 11]: cell.fill = fill_g
                    if i == 2:
                        cell.number_format = '#,##0'
                    elif i == 6:
                        cell.number_format = '0.00'
                    elif i == 11:
                        cell.number_format = '0%'
                    elif i in [3, 4, 5, 7, 8]:
                        cell.number_format = '#,##0.00'

                ws.cell(row_idx + 2, 2, "SHEAR ( t )").alignment = a_c
                ws.merge_cells(start_row=row_idx + 2, start_column=3, end_row=row_idx + 2, end_column=4)
                ws.cell(row_idx + 2, 3, "Position").alignment = a_c
                ws.cell(row_idx + 2, 5, "Thickness\n( mm )").alignment = a_c
                ws.cell(row_idx + 2, 6, "Grade (k)").alignment = a_c
                ws.merge_cells(start_row=row_idx + 2, start_column=7, end_row=row_idx + 2, end_column=8)
                ws.cell(row_idx + 2, 7, "Shear flow\n(N/mm) for unit load").alignment = a_c
                ws.cell(row_idx + 2, 9, "Shear stress\n(N/mm²)").alignment = a_c
                ws.cell(row_idx + 2, 10, "Allowable stress\n(N/mm²)").alignment = a_c
                ws.cell(row_idx + 2, 11, "Percentage\n(%)").alignment = a_c
                ws.cell(row_idx + 2, 12, "Result").alignment = a_c
                r4 = [data['Shear'], data['Pos_Shear'], data['Thk'], data['Grade_K'],
                      data['Unit_q'], data['Act_FS'], data['Allow_FS'],
                      data['Act_FS'] / data['Allow_FS'] if data['Allow_FS'] > 0 else 0,
                      "OK" if data['Act_FS'] <= data['Allow_FS'] else "NG"]

                ws.cell(row_idx + 3, 2, r4[0]).fill = fill_y
                ws.cell(row_idx + 3, 2).number_format = '#,##0'
                ws.merge_cells(start_row=row_idx + 3, start_column=3, end_row=row_idx + 3, end_column=4)
                ws.cell(row_idx + 3, 3, r4[1])
                ws.cell(row_idx + 3, 5, r4[2]).fill = fill_y
                ws.cell(row_idx + 3, 6, r4[3]).fill = fill_y
                ws.cell(row_idx + 3, 6).number_format = '0.00'
                ws.merge_cells(start_row=row_idx + 3, start_column=7, end_row=row_idx + 3, end_column=8)
                ws.cell(row_idx + 3, 7, r4[4]).fill = fill_y
                ws.cell(row_idx + 3, 7).number_format = '0.00E+00'
                ws.cell(row_idx + 3, 9, r4[5]).fill = fill_g
                ws.cell(row_idx + 3, 10, r4[6]).fill = fill_g
                ws.cell(row_idx + 3, 11, r4[7]).fill = fill_g
                ws.cell(row_idx + 3, 11).number_format = '0%'
                ws.cell(row_idx + 3, 12, r4[8])

                for r in range(row_idx + 2, row_idx + 4):
                    for c in range(2, 13):
                        ws.cell(r, c).border = b_all
                        ws.cell(r, c).alignment = a_c
                        ws.cell(r, c).font = f_11n

                ws.merge_cells(start_row=row_idx + 1, start_column=1, end_row=row_idx + 3, end_column=1)
                ws.cell(row_idx + 1, 1).border = b_all
                ws.cell(row_idx + 2, 1).border = b_all
                ws.cell(row_idx + 3, 1).border = b_all
                ws.cell(row_idx + 1, 1).font = f_11b
                row_idx += 4

            self.apply_outer_border(ws, 1, row_idx - 1, 1, 12)
            for i, w in enumerate([14, 13, 10, 16, 13, 10, 12, 12, 18, 16, 12, 12], 1):
                ws.column_dimensions[get_column_letter(i)].width = w
            ws.page_setup.orientation = ws.ORIENTATION_PORTRAIT
            ws.page_setup.fitToPage = True
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.sheet_view.view = 'pageBreakPreview'
            ws.print_area = f"A1:L{row_idx - 1}"
            ws.page_setup.paperSize = ws.PAPERSIZE_A4
            ws.page_margins.left = 0.3
            ws.page_margins.right = 0.3
            ws.print_options.horizontalCentered = True

            ws_v = wb.create_sheet(title="Visualizations")
            ws_v.sheet_view.view = 'pageBreakPreview'
            ws_v.page_setup.orientation = ws_v.ORIENTATION_LANDSCAPE
            ws_v.page_setup.fitToPage = True
            ws_v.page_setup.fitToWidth = 0
            ws_v.page_setup.fitToHeight = 1
            col_pos = 2
            for d in self.saved_frames_data:
                ws_v.cell(2, col_pos, f"Results: {d['Frame']}").font = f_18b
                im1 = XlImage(io.BytesIO(d['ImgSec']))
                im1.width, im1.height = int(im1.width * 0.5), int(im1.height * 0.5)
                ws_v.add_image(im1, ws_v.cell(4, col_pos).coordinate)
                im2 = XlImage(io.BytesIO(d['ImgShr']))
                im2.width, im2.height = int(im2.width * 0.5), int(im2.height * 0.5)
                ws_v.add_image(im2, ws_v.cell(22, col_pos).coordinate)
                current_page_num = (col_pos - 2) // 8 + 1
                ws_v.col_breaks.append(Break(id=current_page_num * 8))
                col_pos += 8
            ws_v.print_area = f"A1:{get_column_letter(col_pos - 1)}40"
            for i in range(1, col_pos):
                ws_v.column_dimensions[get_column_letter(i)].width = 9

            wb.save(path)
            QMessageBox.information(self, "Success", "Export Done!")

        except PermissionError:
            QMessageBox.critical(self, "Error (Errno 13)",
                                 "The file is currently open in Excel.\nPlease close it and try again.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def refresh_ui(self):
        saved = [edit.text() for edit in self.shell_thickness_inputs]
        self.fig1.clear()
        self.fig2.clear()
        ax1, ax2 = self.fig1.add_subplot(111), self.fig2.add_subplot(111)

        for i in reversed(range(self.thickness_layout.count())):
            w = self.thickness_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self.shell_thickness_inputs.clear()

        if self.is_calculated:
            if self.centerlines:
                for cl in self.centerlines:
                    lo = cl['line']
                    x, y = lo.xy
                    ct = cl.get('type', '')
                    if ct == '1999':
                        color = '#FF00FF'
                    elif ct == '157':
                        color = '#00FFFF'
                    elif ct == '1102':
                        color = '#FFA500'
                    elif ct == 'stiffener':
                        color = '#008000'
                    elif ct == 'bridge':
                        color = '#00CC00'
                    else:
                        color = '#00FF00'
                    thk = cl.get('thickness', 10.0)

                    if thk > 0:
                        try:
                            poly = lo.buffer(thk / 2.0, cap_style=2)
                            if poly.geom_type == 'Polygon':
                                ax1.fill(*poly.exterior.xy, color=color, alpha=0.3, zorder=9, edgecolor='none')
                            elif poly.geom_type == 'MultiPolygon':
                                for p in poly.geoms:
                                    ax1.fill(*p.exterior.xy, color=color, alpha=0.3, zorder=9, edgecolor='none')
                        except:
                            pass
                    ax1.plot(x, y, color=color, linewidth=2.5, alpha=0.9, zorder=10, linestyle='-')

            if hasattr(self, 'mesh_cells') and self.mesh_cells:
                cmap = matplotlib.colormaps.get_cmap('tab20')
                for idx, poly in enumerate(self.mesh_cells):
                    color = cmap(idx % 20)
                    ax2.fill(*poly.exterior.xy, color=color, alpha=0.4)
                    ax2.plot(*poly.exterior.xy, color=color, linewidth=1.5)

            if hasattr(self, 'cell_points') and self.cell_points:
                ax2.plot([p[0] for p in self.cell_points], [p[1] for p in self.cell_points], 'ro', markersize=4,
                         markeredgecolor='white', markeredgewidth=0.5,
                         label=f'Points ({len(self.cell_points)})', zorder=11)

            h, l = ax2.get_legend_handles_labels()
            if h: ax2.legend(loc="upper right", fontsize=8)

        else:
            if self.raw_1999_lines:
                for ls in self.raw_1999_lines:
                    ax1.plot(*ls.xy, color='black', lw=1.5)
            for i, l in enumerate(self.left_1999_segments):
                ax1.text(l.centroid.x, l.centroid.y, f"S{i + 1}", fontsize=10, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        for i in range(len(self.left_1999_segments)):
            f = QFrame()
            r = QHBoxLayout(f)
            f.setStyleSheet("border-left: 4px solid #2E86C1; background: #FDFEFE; margin-bottom: 2px;")
            r.addWidget(QLabel(f"S{i + 1}:"))
            r.addStretch()
            edit = QLineEdit(saved[i] if i < len(saved) else "10")
            edit.setFixedWidth(self.field_width)
            edit.setStyleSheet(self.input_style)
            r.addWidget(edit)
            self.thickness_layout.addWidget(f)
            self.shell_thickness_inputs.append(edit)

        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.grid(True, lw=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{-x:g}"))
        self.can1.draw()
        self.can2.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UltimateShipAnalyzer()
    win.show()
    sys.exit(app.exec())
            
