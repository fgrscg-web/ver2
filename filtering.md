import sys
import os
import math
import ezdxf
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QTextEdit, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal


# --- [Worker Thread] 백그라운드에서 DXF 처리를 수행하는 스레드 ---
class CleanWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(str)

    def __init__(self, input_path, threshold):
        super().__init__()
        self.input_path = input_path
        self.threshold = threshold

    def run(self):
        try:
            self.log.emit(f"📂 파일 로딩 중: {os.path.basename(self.input_path)}")

            # 인코딩 처리 (cp949 -> utf-8 순서)
            try:
                doc = ezdxf.readfile(self.input_path, encoding='cp949')
            except:
                try:
                    doc = ezdxf.readfile(self.input_path, encoding='utf-8')
                except:
                    doc = ezdxf.readfile(self.input_path)

            msp = doc.modelspace()
            entities = list(msp)
            total = len(entities)
            delete_count = 0

            # 레이어별 기본 속성(선종류, 색상) 미리 수집
            layer_props = {}
            for layer in doc.layers:
                layer_props[layer.dxf.name] = {
                    'linetype': layer.dxf.linetype.upper(),
                    'color': layer.dxf.color  # 기본 ACI 색상 코드
                }

            for i, entity in enumerate(entities):
                is_trash = False
                e_type = entity.dxftype()
                layer_name = entity.dxf.layer

                # ==========================================================
                # 🌟 1. [레이어 및 색상 필터링] (요청 사항 반영)
                # ==========================================================

                # 규칙 A: 28001 레이어는 무조건 전체 제거
                if layer_name in ('28001','3001', '4001', '5001', '-1131'):
                    is_trash = True

                if not is_trash:
                    # 객체의 실제 색상 추출 (256은 레이어 설정을 따르는 BYLAYER 상태)
                    ent_color = entity.dxf.color if entity.dxf.hasattr('color') else 256
                    if ent_color == 256:
                        resolved_color = layer_props.get(layer_name, {}).get('color', 7)
                    else:
                        resolved_color = ent_color

                    # 규칙 B: 8001 레이어 중 하늘색(Cyan, ACI Code: 4) 선분 제거
                    if layer_name == '8001' and resolved_color == 4:
                        is_trash = True

                    # 규칙 C: 6001 레이어 중 선홍색(Magenta, 6) 또는 흰색(White, 7) 선분 제거
                    elif layer_name == '6001' and resolved_color in [7]:
                        is_trash = True
                    elif layer_name == '-1102' and resolved_color in [1, 2]:
                        is_trash = True

                # ==========================================================
                # 🌟 3. [선(Line) 제거] 점선 및 미세 파편 지우기
                # ==========================================================
                if not is_trash:
                    l_type = entity.dxf.linetype.upper() if entity.dxf.hasattr('linetype') else 'BYLAYER'
                    if l_type == 'BYLAYER':
                        l_type = layer_props.get(layer_name, {}).get('linetype', 'CONTINUOUS')

                    bad_linetypes = [
                        'AVEVASHORTDASHED', 'AVEVADASHED', 'AVEVAHIDDEN',
                        'DASH', 'HIDD', 'CENT', 'DOT', 'PHANTOM'
                    ]

                    if any(bad_word in l_type for bad_word in bad_linetypes):
                        is_trash = True

                    # 파편(Length) 필터링 기준 (0.5mm/m 미만 조각선 제거)
                    if not is_trash and e_type == 'LINE':
                        start, end = entity.dxf.start, entity.dxf.end
                        length = math.hypot(end.x - start.x, end.y - start.y)
                        if length < 3:
                            is_trash = True

                # ==========================================================
                # 최종 삭제 처리
                # ==========================================================
                if is_trash:
                    msp.delete_entity(entity)
                    delete_count += 1

                if i % 100 == 0:
                    self.progress.emit(int((i / total) * 100))

            self.progress.emit(100)

            # 파일 쓰기 잠김 에러 방지 안전 저장 로직
            base_path = self.input_path.replace(".dxf", "_cleaned")
            output_path = f"{base_path}.dxf"
            counter = 1

            while os.path.exists(output_path):
                try:
                    with open(output_path, 'a'):
                        pass
                    break
                except PermissionError:
                    output_path = f"{base_path}_{counter}.dxf"
                    counter += 1

            doc.saveas(output_path)

            self.log.emit(f"✅ 필터링 완료! 총 {delete_count}개의 요소가 정밀 제거되었습니다.")
            self.finished.emit(output_path)

        except Exception as e:
            self.log.emit(f"❌ 에러 발생: {str(e)}")


# --- [UI Main Window] ---
class DXFCleanerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DXF Geometry Optimizer - 맞춤형 레이어/색상 필터 적용기")
        self.resize(700, 500)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #E0E0E0; font-size: 13px; font-family: 'Malgun Gothic'; }
            QPushButton { 
                background-color: #00AD1D; color: white; border-radius: 6px; 
                padding: 10px; font-weight: bold; border: none; font-family: 'Malgun Gothic';
            }
            QPushButton:hover { background-color: #009619; }
            QPushButton:disabled { background-color: #404040; color: #808080; }
            QTextEdit { background-color: #1E1E1E; color: #00FF41; border: 1px solid #333; font-family: 'Consolas'; font-size: 13px; }
            QProgressBar { border: 1px solid #333; border-radius: 5px; text-align: center; color: white; font-weight: bold;}
            QProgressBar::chunk { background-color: #00AD1D; border-radius: 4px;}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        desc = ("<b>[도면 전처리 필터 가동 중]</b><br>"
                "- 점선, 파편, 원, 호, 삼각형 기호 제거<br>"
                "- 28001 레이어 전체 제거<br>"
                "- 8001 레이어 내 하늘색(Cyan) 선분 제거<br>"
                "- 6001 레이어 내 선홍색(Magenta) 및 흰색(White) 선분 제거")
        lbl_desc = QLabel(desc)
        lbl_desc.setTextFormat(Qt.RichText)
        layout.addWidget(lbl_desc)

        file_box = QHBoxLayout()
        self.lbl_path = QLabel("파일을 선택해주세요...")
        self.lbl_path.setStyleSheet(
            "background-color: #1E1E1E; padding: 5px; border: 1px solid #333; border-radius: 4px;")

        btn_browse = QPushButton("파일 선택 📂")
        btn_browse.setFixedWidth(120)
        btn_browse.clicked.connect(self.browse_file)

        file_box.addWidget(self.lbl_path, stretch=1)
        file_box.addWidget(btn_browse)
        layout.addLayout(file_box)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)

        self.btn_run = QPushButton("도면 최적화 및 필터링 시작 🚀")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.start_cleaning)
        layout.addWidget(self.btn_run)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "DXF 파일 선택", "", "DXF Files (*.dxf)")
        if path:
            self.lbl_path.setText(path)
            self.btn_run.setEnabled(True)
            self.log_output.append(f"📍 대상 파일 선택됨: {path}")

    def start_cleaning(self):
        self.btn_run.setEnabled(False)
        self.pbar.setValue(0)
        self.worker = CleanWorker(self.lbl_path.text(), threshold=0.5)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.log.connect(self.log_output.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, output_path):
        QMessageBox.information(self, "정제 완료", f"지정된 조건의 레이어와 색상이 제거된 파일이 생성되었습니다:\n{output_path}")
        self.btn_run.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DXFCleanerApp()
    window.show()
    sys.exit(app.exec())
