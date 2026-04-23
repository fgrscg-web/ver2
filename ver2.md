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
class ShearFlowPathDialog(QDialog):
    def __init__(self, graph_edges, graph_nodes, root_node, parent=None):
        super().__init__(parent)
        self.setWindowTitle("디버깅: 가상 슬릿 배치 및 q₀(정정 전단류) 분포")
        self.resize(1200, 900)
        layout = QVBoxLayout(self)
        lbl = QLabel("<b>[시각화 가이드]</b><br>"
                     "컬러 라인: 기하 방향 기준 |q₀| 분포 (jet colormap)<br>"
                     "<font color='red'><b>컬러 점선(--) 및 '||' 마크: 가상 슬릿(Slit) / 가상 뱅크 (S=0 시작점)</b></font><br>"
                     "검은색 화살표(▷): Water-flow 기반 S 누적 방향 (합류 노드에서 자동 합산)<br>"
                     "별(★): 최대 |q₀| 지점")
        layout.addWidget(lbl)

        self.fig = Figure()
        canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, stretch=1)

        btn = QPushButton("닫기")
        btn.setFixedHeight(40)
        btn.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; font-size: 14px;")
        btn.clicked.connect(self.close)
        layout.addWidget(btn)

        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)

        all_q0 = [abs(v) for e in graph_edges for v in e.get('q0_geom', [0])]
        max_q0_global = max(all_q0) if all_q0 else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_q0_global)
        cmap = matplotlib.colormaps.get_cmap('jet')

        max_q0_val = -1.0
        max_q0_pt = None

        for e in graph_edges:
            pts = np.array(e['sample_pts'])
            q0_geom = np.abs(e.get('q0_geom', np.zeros(len(pts))))

            for i in range(len(pts) - 1):
                color = cmap(norm((q0_geom[i] + q0_geom[i + 1]) / 2))
                if e['is_slit']:
                    ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]], color=color, linestyle='--',
                            linewidth=2.5, zorder=3)
                else:
                    ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]], color=color, linewidth=2.5,
                            zorder=2)

            local_max_idx = np.argmax(q0_geom)
            if q0_geom[local_max_idx] > max_q0_val:
                max_q0_val = q0_geom[local_max_idx]
                max_q0_pt = pts[local_max_idx]

            if e['is_slit']:
                geom = e['line']
                L = geom.length
                dist = max(0, L - min(50.0, L * 0.05))
                slit_pt = geom.interpolate(dist)

                p_before = geom.interpolate(max(0, dist - 5.0))
                p_after = geom.interpolate(min(L, dist + 5.0))
                angle = np.degrees(np.arctan2(p_after.y - p_before.y, p_after.x - p_before.x))

                ax.text(slit_pt.x, slit_pt.y, '||', color='red', fontsize=14, fontweight='bold',
                        ha='center', va='center', rotation=angle, zorder=5)

            if len(pts) >= 2:
                mid_idx = (len(pts) - 1) // 2
                if e.get('bfs_direction') == 'reverse':
                    p1, p2 = pts[mid_idx + 1], pts[mid_idx]
                else:
                    p1, p2 = pts[mid_idx], pts[mid_idx + 1]

                arrow = patches.FancyArrowPatch(
                    (p1[0], p1[1]), (p2[0], p2[1]),
                    mutation_scale=15, color='black', arrowstyle='-|>', zorder=4
                )
                ax.add_patch(arrow)

        for nid, node in graph_nodes.items():
            rx, ry = node['coord']
            ax.plot(rx, ry, 'o', markerfacecolor='gray', markeredgecolor='white', markersize=6, zorder=1)

        if max_q0_pt is not None:
            ax.plot(max_q0_pt[0], max_q0_pt[1], 'k*', markersize=15, label=f"Max |q₀|: {max_q0_val:.2f} N/mm", zorder=7)

        ax.legend(loc='upper right')
        canvas.draw()

    def closeEvent(self, event: QCloseEvent):
        plt.close(self.fig)
        super().closeEvent(event)


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
            self.mesh_cells = [poly for poly in raw_loops if poly.area >= 100.0]

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
            # Step 12: 가상 슬릿 배치 및 초기 전단류(q0) 계산
            # =============================================================
            progress.setLabelText("Step 12: 가상 슬릿 배치 및 전단류(q0) 산출 중...")
            QApplication.processEvents()

            # --- Identify thin loops, extract left half-section ---
            thin_polys = []
            for poly in self.mesh_cells:
                minx, miny, maxx, maxy = poly.bounds
                if minx < -1e-3 and maxx > 1e-3:
                    thks = [cl['thickness'] for cl in self.shear_flow_centerlines if cl['line'].distance(poly) < 1.0]
                    max_thk = max(thks) if thks else 20.0
                    if abs(minx) <= max_thk and abs(maxx) <= max_thk:
                        thin_polys.append(poly)

            clip_box = box(-999999, -999999, 1e-3, 999999)
            left_shear_lines = []
            for cl in self.shear_flow_centerlines:
                line = cl['line']
                is_thin_loop_edge = any(line.distance(tp.exterior) < 1e-3 for tp in thin_polys)
                if is_thin_loop_edge:
                    left_shear_lines.append(cl)
                else:
                    try:
                        res = line.intersection(clip_box)
                        if not res.is_empty:
                            if res.geom_type == 'LineString':
                                left_shear_lines.append({'line': res, 'thickness': cl['thickness'], 'type': cl.get('type')})
                            elif res.geom_type == 'MultiLineString':
                                for g in res.geoms:
                                    left_shear_lines.append({'line': g, 'thickness': cl['thickness'], 'type': cl.get('type')})
                    except:
                        pass

            # --- 1. 맹장(자유단, degree=1) 반복 가지치기 ---
            raw_edges = []
            for i, cl in enumerate(left_shear_lines):
                c = list(cl['line'].coords)
                if len(c) < 2: continue
                p1 = (round(c[0][0], 2), round(c[0][1], 2))
                p2 = (round(c[-1][0], 2), round(c[-1][1], 2))
                if p1 == p2: continue
                
                thk = cl.get('thickness', 10.0)
                is_protected = False
                
                # [수정] 보호 조건 1: 1D선 최대 x좌표 + 두께/2 값이 0보다 크다면 무조건 살리기 (두께를 고려해 대칭축에 걸치는 선분 보호)
                if max(p1[0], p2[0]) + (thk / 2.0) > 0:
                    is_protected = True
                
                # [수정] 보호 조건 2: 수평 방향으로 겹치는 미세 단차 보호 (교차/겹침 현행 유지)
                length = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if abs(p1[1] - p2[1]) < 1e-2 and length <= thk * 1.5:
                    is_protected = True

                raw_edges.append({'p1': p1, 'p2': p2, 'line': cl['line'], 'thk': thk, 'is_protected': is_protected})

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
                    if e['is_protected']:
                        survivors.append(e)
                    else:
                        if node_deg[e['p1']] <= 1 or node_deg[e['p2']] <= 1:
                            changed = True
                        else:
                            survivors.append(e)

                if not survivors and raw_edges:
                    survivors = raw_edges
                    break
                raw_edges = survivors

            # --- 2. 엄격한 진짜 꼭짓점 선별 ---
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

            # --- 3. Macro Edges 생성 (일직선 분할 노드 병합) 및 Cost 할당 ---
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
                    cell_count = sum(1 for poly in self.mesh_cells if poly.exterior.distance(mid_pt) < 2.0)

                    if cell_count >= 2:
                        cost = -1000
                    elif cell_count == 1:
                        cost = 100 - (10.0 / (abs(mid_pt.x) + 1.0))
                    else:
                        cost = -2000

                    sn = pt_to_nid[start_pt]
                    en = pt_to_nid.get(current_pt, sn)

                    if sn != en:
                        edge_dict = {
                            'id': current_eid, 'start_node': sn, 'end_node': en,
                            'line': macro_line, 'length': macro_line.length,
                            'thickness': thk, 'cell_count': cell_count, 'cost': cost
                        }
                        macro_edges.append(edge_dict)
                        node_edges[sn].append(current_eid)
                        node_edges[en].append(current_eid)
                        current_eid += 1

            self.graph_edges = macro_edges

            # --- 단계 C: Kruskal 최소 비용 스패닝 트리 (1셀 1슬릿 확정) ---
            parent = {i: i for i in range(len(self.cell_points))}

            def find(i):
                if parent[i] == i: return i
                parent[i] = find(parent[i])
                return parent[i]

            def union(i, j):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_i] = root_j
                    return True
                return False

            sorted_edge_indices = sorted(range(len(self.graph_edges)), key=lambda i: self.graph_edges[i]['cost'])

            self.slit_edge_ids = set()
            tree_edge_ids = set()

            for eid in sorted_edge_indices:
                e = self.graph_edges[eid]
                if union(e['start_node'], e['end_node']):
                    tree_edge_ids.add(eid)
                else:
                    self.slit_edge_ids.add(eid)

            for e in self.graph_edges:
                e['is_slit'] = e['id'] in self.slit_edge_ids

            # --- 단계 D: 위상 정렬 기반 S 누적 ---
            left_all_lines = []
            for cl in self.centerlines:
                line = cl['line']
                is_thin = any(line.distance(tp.exterior) < 1e-3 for tp in thin_polys)
                if is_thin:
                    left_all_lines.append(cl)
                else:
                    try:
                        res = line.intersection(clip_box)
                        if not res.is_empty:
                            if res.geom_type == 'LineString':
                                left_all_lines.append({'line': res, 'thickness': cl['thickness']})
                            elif res.geom_type == 'MultiLineString':
                                for g in res.geoms:
                                    left_all_lines.append({'line': g, 'thickness': cl['thickness']})
                    except:
                        pass

            for edge in self.graph_edges:
                L = edge['length']
                n_samples = max(2, int(L / 50.0) + 1)
                edge['sample_s'] = np.linspace(0, L, n_samples)
                edge['sample_pts'] = []
                for s in edge['sample_s']:
                    pt = edge['line'].interpolate(s)
                    edge['sample_pts'].append((pt.x, pt.y))
                edge['increments'] = np.zeros(n_samples)

            # Map the true stiffened area to the shear flow paths
            for cl in left_all_lines:
                line = cl['line']
                thk = cl.get('thickness', 10.0)
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    p1 = np.array(coords[i])
                    p2 = np.array(coords[i+1])
                    L_seg = np.linalg.norm(p2 - p1)
                    if L_seg < 1e-6: continue
                    n_chunks = max(1, int(L_seg / 50.0))
                    chunk_L = L_seg / n_chunks
                    dA = chunk_L * thk
                    
                    for c_idx in range(n_chunks):
                        f1 = c_idx / n_chunks
                        f2 = (c_idx + 1) / n_chunks
                        mid = p1 + (p2 - p1) * ((f1 + f2) / 2.0)
                        dQ = dA * (mid[1] - self.calc_na_bl)
                        
                        mid_pt = Point(mid)
                        best_edge = None
                        best_dist = float('inf')
                        best_s = 0.0
                        
                        for edge in self.graph_edges:
                            dist = edge['line'].distance(mid_pt)
                            if dist < best_dist:
                                best_dist = dist
                                best_edge = edge
                                best_s = edge['line'].project(mid_pt)
                                
                        if best_edge is not None:
                            s_array = best_edge['sample_s']
                            if best_s <= s_array[1]:
                                best_edge['increments'][1] += dQ
                            else:
                                idx = np.searchsorted(s_array, best_s)
                                if idx >= len(s_array): idx = len(s_array) - 1
                                best_edge['increments'][idx] += dQ

            for edge in self.graph_edges:
                edge['S_total'] = np.sum(edge['increments'])

            # 2. Water-Flow를 위한 위상 정렬 준비
            unresolved_degree = defaultdict(int)
            for eid in tree_edge_ids:
                e = self.graph_edges[eid]
                unresolved_degree[e['start_node']] += 1
                unresolved_degree[e['end_node']] += 1

            S_sum_at_node = defaultdict(float)

            # 3. 슬릿 엣지를 가상 뱅크(Virtual Bank, S=0 시작점)로 취급하여 흐름 주입
            for eid in self.slit_edge_ids:
                edge = self.graph_edges[eid]
                edge['bfs_direction'] = 'reverse'  

                n_samples = len(edge['sample_s'])
                S_acc = np.zeros(n_samples)
                rev_incs = edge['increments'][::-1]
                current_S = 0.0  

                for k in range(1, n_samples):
                    current_S += rev_incs[k - 1]
                    S_acc[k] = current_S

                edge['S_accumulated'] = S_acc
                S_sum_at_node[edge['start_node']] += edge['S_total']

            # 4. Leaf 노드부터 차례대로 Water-flow 누적 (위상 정렬 Queue)
            queue = deque([nid for nid in self.graph_nodes if unresolved_degree[nid] <= 1])

            loop_safeguard_queue = 0
            while queue:
                loop_safeguard_queue += 1
                if loop_safeguard_queue % 50 == 0: QApplication.processEvents()

                u = queue.popleft()
                if unresolved_degree[u] == 0:
                    continue  

                edge_idx = -1
                for eid in node_edges[u]:
                    if eid in tree_edge_ids:
                        edge_idx = eid
                        break

                if edge_idx == -1: continue
                edge = self.graph_edges[edge_idx]
                tree_edge_ids.remove(edge_idx)

                v = edge['start_node'] if edge['end_node'] == u else edge['end_node']

                start_S = S_sum_at_node[u]
                n_samples = len(edge['sample_s'])
                S_acc = np.zeros(n_samples)
                current_S = start_S

                if edge['start_node'] == u:
                    edge['bfs_direction'] = 'forward'
                    incs = edge['increments']
                    S_acc[0] = start_S
                    for k in range(1, n_samples):
                        current_S += incs[k]
                        S_acc[k] = current_S
                else:
                    edge['bfs_direction'] = 'reverse'
                    rev_incs = edge['increments'][::-1]
                    S_acc[0] = start_S
                    for k in range(1, n_samples):
                        current_S += rev_incs[k - 1]
                        S_acc[k] = current_S

                edge['S_accumulated'] = S_acc
                S_sum_at_node[v] += current_S

                unresolved_degree[u] -= 1
                unresolved_degree[v] -= 1

                if unresolved_degree[v] == 1:
                    queue.append(v)

            # 5. 최종 정정 전단류 q0 산출
            V = abs(self.raw_shear) * 1000 * 9.80665
            Ixx = self.calc_ixx

            for edge in self.graph_edges:
                if 'S_accumulated' in edge:
                    if Ixx > 1e-6:
                        q0 = -V * edge['S_accumulated'] / Ixx
                    else:
                        q0 = np.zeros(len(edge['sample_s']))

                    edge['q0'] = q0
                    if edge.get('bfs_direction') == 'reverse':
                        edge['q0_geom'] = q0[::-1]
                    else:
                        edge['q0_geom'] = q0

            # =============================================================
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
            res += f"Slit Edges       : {len(self.slit_edge_ids)} (1 Cut per Cell)\n"

            self.base_report = res
            self.result_box.setText(self.base_report)
            self.is_calculated = True
            self.btn_eval.setEnabled(True)

            progress.setLabelText("Rendering...")
            progress.setMaximum(0)
            QApplication.processEvents()
            self.refresh_ui()

            if self.graph_edges:
                self.dialog_shear_flow = ShearFlowPathDialog(self.graph_edges, self.graph_nodes, self.root_node, self)
                self.dialog_shear_flow.show()
                self.debug_dialogs.append(self.dialog_shear_flow)

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
