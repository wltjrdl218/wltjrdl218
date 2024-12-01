import os
import sys
if not hasattr(sys, 'stderr') or sys.stderr is None:
    sys.stderr = open('stderr.log', 'w')

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
    QComboBox, QProgressBar, QLineEdit, QTableWidget, QTableWidgetItem, 
    QTabWidget, QMainWindow, QSizePolicy, QLabel, QSpacerItem, QTextEdit, 
    QHeaderView, QScrollArea, QLineEdit, QMessageBox, QDialog, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
import torch
import time

### 한글처리
plt.rc('font', family='Malgun Gothic')
# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

import cx_Oracle


from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account

import subprocess
import psutil


##################################################################################################################################
### 팝업창 페이지
# ActionWindow 클래스
class ActionWindow(QMainWindow):
    def __init__(self, title, message, action_label, parent=None, selected_row_data=None, current_tab=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.selected_row_data = selected_row_data  # 선택된 행의 데이터 저장
        self.current_tab = current_tab  # 현재 탭 정보 저장
        self.setFixedSize(528, 500)
        
        # print(self.current_tab)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; margin-top: -40px;")
        layout.addWidget(label)
        
        # 버튼 레이아웃 추가 전 간격 추가
        layout.addSpacing(40)  # 버튼과 메시지 사이의 간격

        button_layout = QHBoxLayout()
        action_button = QPushButton(action_label)
        back_button = QPushButton("뒤로가기")

        action_button.setFixedSize(200, 70)
        back_button.setFixedSize(200, 70)

        action_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;  
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)

        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)

        action_button.clicked.connect(self.handle_action)
        back_button.clicked.connect(self.handle_back)

        button_layout.addWidget(action_button)
        button_layout.addWidget(back_button)
        layout.addLayout(button_layout)

        layout.addSpacing(20)
        central_widget.setLayout(layout)

    def handle_action(self):
        # 로그아웃 처리
        if self.windowTitle() == "로그아웃":
            print("로그아웃 완료!")
            QApplication.quit()  # 프로그램 종료
            return
        
        # 수정하기 일때
        if self.windowTitle() == "수정":
            print("수정완료!")            
            return
        
        # 데이터 선택하지 않고 삭제누를 때
        if not self.selected_row_data:
            QMessageBox.warning(self, "삭제 오류", "삭제할 데이터가 없습니다.")
            return

        row_id = self.selected_row_data[0]

        # 테이블 삭제 SQL 쿼리 결정
        if self.current_tab == "경험공유":
            delete_sql = "DELETE FROM post WHERE PostID = :row_id"
        elif self.current_tab == "공지사항":
            delete_sql = "DELETE FROM notice WHERE NoticeID = :row_id"
        elif self.current_tab == "AI 예측 결과":
            delete_sql = "DELETE FROM predictions WHERE Prediction_ID = :row_id"
        elif self.current_tab == "사용자 피드백":
            delete_sql = "DELETE FROM feedback WHERE Feedback_ID = :row_id"
        elif self.current_tab == None:
            delete_sql = 'DELETE FROM "File" WHERE FILE_NAME = :row_id'   
        else:
            QMessageBox.warning(self, "삭제 오류", "올바르지 않은 탭에서 삭제를 시도했습니다.")
            return

        # 데이터베이스에 연결하고 쿼리 실행
        try:
            dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com", 1521, "orcl")
            conn = cx_Oracle.connect("admin", "1emddlqslek", dsn)
            cursor = conn.cursor()

            cursor.execute(delete_sql, row_id=row_id)
            conn.commit()

            
            self.parent().update_community_table()  # 삭제 후 테이블 업데이트
            self.close()

        except cx_Oracle.DatabaseError as e:
            QMessageBox.warning(self, "삭제 오류", f"데이터베이스 삭제 중 오류가 발생했습니다: {str(e)}")

        finally:
            cursor.close()
            conn.close()
    
    def handle_back(self):
        self.close()        

##################################################################################################################################
## 삭제 실패
class WarningWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("삭제 오류")
        self.setFixedSize(528, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        label = QLabel("삭제할 데이터를 선택해 주세요.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; margin-top: 10px;")
        layout.addWidget(label)
        
        layout.addSpacing(40)

        back_button = QPushButton("뒤로가기")
        back_button.setFixedSize(200, 70)
        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        back_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(back_button)
        layout.addLayout(button_layout)

        layout.addSpacing(20)
        central_widget.setLayout(layout)

##################################################################################################################################
### 로그인 실패 페이지
class NoLoginWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("로그인 실패")
        self.setFixedSize(528, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        label = QLabel("로그인 되지 않았습니다.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; margin-top: -40px;")
        layout.addWidget(label)

        back_button = QPushButton("뒤로가기")
        back_button.setFixedSize(200, 70)
        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        back_button.clicked.connect(self.handle_back)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        central_widget.setLayout(layout)

    def handle_back(self):
        self.close()

##################################################################################################################################

### 로그인 페이지
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()       
        
        
        self.setWindowTitle("관리자 Login")
        self.setFixedSize(1925, 990)

        self.background_label = QLabel(self)
        self.background_pixmap = QPixmap("./image/universe.jpg")
        self.background_pixmap = self.background_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setGeometry(0, 0, self.width(), self.height())

        main_layout = QVBoxLayout(self)

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setSpacing(20)

        login_label = QLabel()
        login_pixmap = QPixmap("./image/logo.png")
        login_pixmap = login_pixmap.scaled(300, 100, Qt.KeepAspectRatio)
        login_label.setPixmap(login_pixmap)
        login_label.setAlignment(Qt.AlignCenter)
        central_layout.addWidget(login_label, alignment=Qt.AlignCenter)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("아이디를 입력하세요")
        self.id_input.setFixedSize(300, 40)
        self.id_input.setStyleSheet("""
            QLineEdit {
                font-family: 'Malgun Gothic';
                background-color: #2b2b2b;
                color: white;
                border: 0.7px black;
                border-radius: 10px;
                padding-left: 10px;
            }
        """)
        central_layout.addWidget(self.id_input, alignment=Qt.AlignCenter)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("비밀번호를 입력하세요")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedSize(300, 40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                font-family: 'Malgun Gothic';
                background-color: #2b2b2b;
                color: white;
                border: 0.7px black;
                border-radius: 10px;
                padding-left: 10px;
            }
        """)
        central_layout.addWidget(self.password_input, alignment=Qt.AlignCenter)

        login_button = QPushButton("로그인")
        login_button.setFixedSize(300, 40)
        login_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #FF8800;
                border-radius: 10px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #FFA500;
            }
        """)
        login_button.clicked.connect(self.handle_login)
        central_layout.addWidget(login_button, alignment=Qt.AlignCenter)

        self.id_input.returnPressed.connect(self.handle_login)
        self.password_input.returnPressed.connect(self.handle_login)

        main_layout.addWidget(central_widget, alignment=Qt.AlignCenter)
        self.setLayout(main_layout)

        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com", 1521, "orcl")
        conn = cx_Oracle.connect("admin", "1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            Select adminid, password
            From admin
        """
        cursor.execute(sql) 
        admin_rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        

        self.id_list = [row[0] for row in admin_rows]
        self.pw_list = [row[1] for row in admin_rows]

        # YOLO 모델 로드
        # YOLO 얼굴 인식 모델을 사용
        model_path = os.path.join(os.getcwd(), 'models', 'yolov5s.pt')
        torch.hub.set_dir('./data/cashe')  # 사용자 정의 캐시 경로 설정
        
        # base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        # cache_dir = os.path.join(base_path, 'data', 'cache')
        # torch.hub.set_dir(cache_dir)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.face_detected = False



    def handle_login(self):
        id_text = self.id_input.text()
        password_text = self.password_input.text()
        # print(f"ID: {id_text}, PW: {password_text}")
        
        if (id_text in self.id_list) and (password_text in self.pw_list):
            ### 로그인 아이디 전역변수에 담기
            self.logged_in_user_id = id_text
            
            # # 로그인 성공 시 MainWindow로 이동
            # self.main_window = MainWindow(logged_in_user_id=id_text)
            
            # 1차 인증 통과 -> 얼굴 인식 팝업 창 열기
            self.show_face_recognition_popup()
        else:
            # 로그인 실패 시 NoLoginWindow 창 띄우기
            self.nologin_window = NoLoginWindow(self)
            self.nologin_window.show()

    def show_face_recognition_popup(self):
        # 얼굴 인식 팝업 창 열기
        self.face_popup = QDialog(self)
        self.face_popup.setWindowTitle("얼굴 인증")
        self.face_popup.setFixedSize(800, 600)
        self.face_popup.setStyleSheet("background-color: #2b2b2b;")  # 배경색 설정

        # 레이아웃 설정
        layout = QVBoxLayout()

        # 카메라 피드를 표시할 QLabel 생성
        self.camera_label = QLabel(self)
        layout.addWidget(self.camera_label)

        # 카메라를 통해 얼굴을 인증 중이라는 안내 메시지
        self.info_label = QLabel("카메라를 통해 얼굴을 인증 중입니다...")
        self.info_label.setStyleSheet("""
            font-family: 'Malgun Gothic';
            color: white;
            font-size: 22px;
        """)  # 글씨 색상, 볼드체 및 크기 설정
        self.info_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬
        layout.addWidget(self.info_label)

        # 닫기 버튼 추가
        self.close_button = QPushButton("닫기")
        self.close_button.setFixedHeight(40)
        self.close_button.setFixedWidth(100)
        self.close_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                background-color: orange;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #ff9500;
            }
        """)
        self.close_button.clicked.connect(self.face_popup.close)  # 버튼 클릭 시 팝업 닫기
        layout.addWidget(self.close_button, alignment=Qt.AlignCenter)

        self.face_popup.setLayout(layout)

        # 팝업 창이 닫힐 때 타이머 중지 및 카메라 해제 처리
        self.face_popup.finished.connect(self.stop_camera)

        self.face_popup.show()

        # 카메라 피드를 지속적으로 업데이트하기 위해 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.cap = cv2.VideoCapture(0)  # 카메라 열기
        self.timer.start(30)  # 30ms마다 업데이트 (약 33FPS)


    def stop_camera(self):
        # 타이머 중지 및 카메라 해제
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    
    def update_camera_feed(self):
        ret, frame = self.cap.read()  # 카메라에서 프레임 읽기
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('./face_dataset/face_model/face_trainer.yml')  # 학습된 얼굴 인식 모델을 불러옴

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                label, confidence = recognizer.predict(face_roi)
                if confidence < 100:  # confidence가 50 이하일 때만 당신의 얼굴로 판단
                    if not hasattr(self, 'face_detected_start_time'):  # 얼굴이 처음 감지된 시간 기록
                        self.face_detected_start_time = time.time()

                    if time.time() - self.face_detected_start_time >= 1:
                        print("1초 동안 관리자의 얼굴이 감지되었습니다. MainWindow로 이동합니다.")
                        self.timer.stop()
                        self.cap.release()
                        self.face_popup.accept()  # 얼굴 인증 완료 후 팝업 닫기
                        self.handle_face_recognition_success()
                else:
                    if hasattr(self, 'face_detected_start_time'):
                        del self.face_detected_start_time

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)  # 주황색 박스
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.camera_label.setPixmap(pixmap)
            self.camera_label.setScaledContents(True)    
    

    def start_face_recognition(self):
        cap = cv2.VideoCapture(0)  # 카메라 열기
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO로 얼굴 감지
            results = self.model(frame)
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            frame = self.plot_boxes(results, frame)

            # 얼굴이 감지되면 2차 인증 통과
            if len(cords) > 0:
                self.face_detected = True
                print("얼굴 인식 성공! 2차 인증 완료")
                cap.release()
                cv2.destroyAllWindows()
                self.face_popup.accept()  # 얼굴 인증 완료 후 팝업 닫기
                self.handle_face_recognition_success()
                break

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def plot_boxes(self, results, frame):
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        for i in range(n):
            row = cords[i]
            if row[4] >= 0.5:  # Confidence threshold
                x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame

    def handle_face_recognition_success(self):
        
        # MainWindow로 이동 (2차 인증 성공 시)
        current_position = self.geometry()
        self.main_window = MainWindow(self.logged_in_user_id)
        self.main_window.setGeometry(current_position)
        self.main_window.show()
        self.close()

##################################################################################################################################

class ClickableLabel(QLabel):
    clicked = pyqtSignal()  # 클릭 시 발생하는 시그널 생성

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()  # 클릭 시 clicked 시그널 방출



##################################################################################################################################
### 메인 페이지
class MainWindow(QMainWindow):    
    def __init__(self, logged_in_user_id=None):
        super().__init__()
        self.logged_in_user_id = logged_in_user_id
        self.setup_ui()  # 초기화 시 UI 설정
        
        
    def setup_ui(self):        
               
        self.setWindowTitle("관리자 페이지")
        self.setFixedSize(1925, 990)

        widget = QWidget()
        self.setCentralWidget(widget)
        
        self.tab_widget = QTabWidget()  # 참조를 유지
        

        layout = QHBoxLayout()
        
        # 삭제 버튼 설정
        self.selected_row_data = None  # 선택된 행의 데이터를 저장하는 변수
        
        self.current_tab_name = None

        # 왼쪽 레이아웃
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(20)

        logo_label = ClickableLabel()  # ClickableLabel 인스턴스 사용
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(380, 100, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignLeft)
        
        # 로고 클릭 시 테이블 새로고침 기능 연결
        logo_label.clicked.connect(self.refresh_main_window)       
        
        
        
        left_layout.addWidget(logo_label)

        input_buttons_widget = QWidget()
        input_buttons_layout = QVBoxLayout(input_buttons_widget)
        input_buttons_layout.setSpacing(15)
        input_buttons_layout.setContentsMargins(10, 10, 10, 10)

        
        # 경험 공유 버튼
        self.community_button = QPushButton("경험공유")
        self.community_button.setFixedHeight(50)
        self.community_button.setStyleSheet(self.get_default_button_style())
        self.community_button.clicked.connect(lambda: self.handle_button_click(self.community_button, "경험공유"))

        # 공지사항 버튼
        self.notice_button = QPushButton("공지사항")
        self.notice_button.setFixedHeight(50)
        self.notice_button.setStyleSheet(self.get_default_button_style())
        self.notice_button.clicked.connect(lambda: self.handle_button_click(self.notice_button, "공지사항"))

        # AI 예측 결과 버튼
        self.aiPredict_button = QPushButton("AI 예측 결과")
        self.aiPredict_button.setFixedHeight(50)
        self.aiPredict_button.setStyleSheet(self.get_default_button_style())
        self.aiPredict_button.clicked.connect(lambda: self.handle_button_click(self.aiPredict_button, "AI 예측 결과"))

        # 사용자 피드백 버튼
        self.user_feedback_button = QPushButton("사용자 피드백")
        self.user_feedback_button.setFixedHeight(50)
        self.user_feedback_button.setStyleSheet(self.get_default_button_style())
        self.user_feedback_button.clicked.connect(lambda: self.handle_button_click(self.user_feedback_button, "사용자 피드백"))

        
        
        input_buttons_layout.addWidget(self.community_button)
        input_buttons_layout.addWidget(self.notice_button)
        input_buttons_layout.addWidget(self.aiPredict_button)
        input_buttons_layout.addWidget(self.user_feedback_button)
        input_buttons_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        input_buttons_widget.setStyleSheet("""
            QWidget {
                font-family: 'Malgun Gothic';
                border: 2px solid #3c3c3c;
                border-radius: 15px;
                padding: 10px;
                background-color: #2b2b2b;
            }
        """)
        left_layout.addWidget(input_buttons_widget)

        # 콤보박스 스타일 설정
        combo = QComboBox()
        combo.addItems(["전체", "딥페이크", "보이스피싱", "문자스미싱"])
        combo.setFixedHeight(45)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: white;
                border-radius: 15px;
                padding-left: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow_icon.png);
                width: 14px;
                height: 14px;
            }
        """)
        input_buttons_layout.addWidget(combo)

        # 검색창 스타일 설정
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("검색어를 입력하세요...")
        line_edit.setFixedHeight(45)
        line_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                color: white;
                border-radius: 15px;
                padding-left: 10px;
                font-size: 14px;
            }
        """)
        input_buttons_layout.addWidget(line_edit)

        # '검색하기' 버튼 스타일 설정
        search_button = QPushButton("검색하기")
        search_button.setFixedHeight(45)
        search_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        input_buttons_layout.addWidget(search_button)

        # 프로그레스바 설정
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setSpacing(15)
        progress_layout.setContentsMargins(15, 15, 15, 15)

        progress_widget.setStyleSheet("""
            QWidget {
                border: 1px solid lightgray;
                border-radius: 15px;
                padding: 10px;
                background-color: #2b2b2b;
            }
        """)

        # AI 모델 재학습 버튼 추가
        model_retrain = QPushButton("AI 모델 재학습")
        model_retrain.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        model_retrain.setFixedHeight(60)
        model_retrain.clicked.connect(self.show_retrain_model_window)
        progress_layout.addWidget(model_retrain)



        # ProgressBar 설정       
        ### DB 연동 SQL        
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            SELECT 
                COUNT(*) AS Total_Rows
            FROM 
                "File" f
            LEFT JOIN 
                Predictions p ON f.File_name = p.File_name
            LEFT JOIN 
                Experience_File_Management efm ON f.File_name = efm.File_name
            LEFT JOIN 
                Post po ON efm.PostID = po.PostID
            WHERE 
                (p.CategoryID = 1 OR po.CategoryID = 1)
                AND (
                    (p.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND p.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                    OR 
                    (po.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND po.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                )      
        """

        cursor.execute(sql) 

        deepfake_file_nm = cursor.fetchall()[0][0] 

        sql = """
            SELECT 
                COUNT(*) AS Total_Rows
            FROM 
                "File" f
            LEFT JOIN 
                Predictions p ON f.File_name = p.File_name
            LEFT JOIN 
                Experience_File_Management efm ON f.File_name = efm.File_name
            LEFT JOIN 
                Post po ON efm.PostID = po.PostID
            WHERE 
                (p.CategoryID = 2 OR po.CategoryID = 2)
                AND (
                    (p.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND p.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                    OR 
                    (po.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND po.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                )     
        """

        cursor.execute(sql) 

        voice_file_nm = cursor.fetchall()[0][0] 
        

        sql = """
            SELECT 
                COUNT(*) AS Total_Rows
            FROM 
                "File" f
            LEFT JOIN 
                Predictions p ON f.File_name = p.File_name
            LEFT JOIN 
                Experience_File_Management efm ON f.File_name = efm.File_name
            LEFT JOIN 
                Post po ON efm.PostID = po.PostID
            WHERE 
                (p.CategoryID = 3 OR po.CategoryID = 3)
                AND (
                    (p.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND p.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                    OR 
                    (po.CreatedAt >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM') 
                    AND po.CreatedAt < ADD_MONTHS(TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE), 'MM'), 1))
                )      
        """

        cursor.execute(sql) 

        smishing_file_nm = cursor.fetchall()[0][0] 

        cursor.close()
        conn.close()
        
        ### 100개가 되면 모델 재학습 기준으로 퍼센테이지 구성
        
        ### 목표갯수 설정
        goal_value = 100
               
        # 딥페이크 프로그래스 바
        progress1 = QProgressBar()
        progress1.setMaximum(goal_value)  # 최대값을 고정된 100으로 설정
        progress1.setValue(min(deepfake_file_nm, goal_value))  # 100을 넘으면 100으로 설정
        progress1.setFormat(f"{deepfake_file_nm}% (딥페이크)")  # 실제 값을 포맷에 표시
        progress1.setStyleSheet("""
            QProgressBar {
                font-family: 'Malgun Gothic';
                border: 1px solid lightgray;
                border-radius: 15px;
                text-align: center;
                color: white;
                background-color: #2b2b2b;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #6495ED, stop: 1 #1E90FF); /* 그라데이션 적용 */
                border-radius: 10px;
            }
        """)
        progress1.setFixedHeight(50)

        # 보이스피싱 프로그래스 바
        progress2 = QProgressBar()
        progress2.setMaximum(goal_value)  # 최대값을 고정된 100으로 설정
        progress2.setValue(min(voice_file_nm, goal_value))  # 100을 넘으면 100으로 설정
        progress2.setFormat(f"{voice_file_nm}% (보이스피싱)")
        progress2.setStyleSheet("""
            QProgressBar {
                font-family: 'Malgun Gothic';
                border: 1px solid lightgray;
                border-radius: 15px;
                text-align: center;
                color: white;
                background-color: #2b2b2b;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #E9967A, stop: 1 #FF6347); /* 그라데이션 적용 */
                border-radius: 10px;
            }
        """)
        progress2.setFixedHeight(50)

        # 문자스미싱 프로그래스 바
        progress3 = QProgressBar()
        progress3.setMaximum(goal_value)  # 최대값을 고정된 100으로 설정
        progress3.setValue(min(smishing_file_nm, goal_value))  # 100을 넘으면 100으로 설정
        progress3.setFormat(f"{smishing_file_nm}% (문자스미싱)")
        progress3.setStyleSheet("""
            QProgressBar {
                font-family: 'Malgun Gothic';
                border: 1px solid lightgray;
                border-radius: 15px;
                text-align: center;
                color: white;
                background-color: #2b2b2b;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #48D1CC, stop: 1 #20B2AA); /* 그라데이션 적용 */
                border-radius: 10px;
            }
        """)
        progress3.setFixedHeight(50)

        # ProgressBar를 레이아웃에 추가
        progress_layout.addWidget(progress1)
        progress_layout.addWidget(progress2)
        progress_layout.addWidget(progress3)

        left_layout.addWidget(progress_widget)

        
        # 로그아웃 버튼 설정
        logout_button = QPushButton("로그아웃")
        logout_button.setFixedHeight(70)
        logout_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        logout_button.clicked.connect(self.show_logout_confirmation)
        left_layout.addWidget(logout_button)

    
        # 중앙 및 오른쪽 레이아웃
        # 첫 화면 시각화
        self.central_layout = QVBoxLayout()

        self.tabs = QTabWidget()         


        # 첫 번째 그래프 
        chart_view1 = self.create_page_conversion_bounce_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')     
                        
        # 두 번째 그래프
        chart_view2 = self.create_traffic_source_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        # 세 번째 그래프        
        chart_view3 = self.create_new_vs_returning_users_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        # 네 번째 그래프         
        chart_view4 = self.create_top_pages_views_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        # 다섯 번째 그래프
        chart_view5 = self.create_user_engagement_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        # 여섯 번째 그래프
        chart_view6 = self.create_top_pages_engagement_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        # 일곱 번째 그래프
        chart_view7 = self.create_city_chart(service_account_file='./data/Quickstart-de2a8ce16450.json', property_id='462860018')
        
        
        
        self.tabs.addTab(chart_view1, "페이지별 전환수 및 이탈률")
        self.tabs.addTab(chart_view2, "트래픽 소스별 사용자 비율")
        self.tabs.addTab(chart_view3, "신규 사용자와 재방문 사용자 비율")
        self.tabs.addTab(chart_view4, "상위 10개 페이지 조회수")
        self.tabs.addTab(chart_view5, "시간대별 사용자 참여도")
        self.tabs.addTab(chart_view6, "상위 10개 페이지별 사용자 참여 시간")
        self.tabs.addTab(chart_view7, "사용자 접속 도시 상위 TOP 10")        
        
        
        
        self.central_layout.addWidget(self.tabs)
        
        # 스타일 시트 설정: QTabWidget 테두리 색상을 회색으로 변경
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3c3c3c;  /* 검정에 가까운 진회색 테두리 */
            }
            QTabBar::tab {
                font-family: 'Malgun Gothic';
                color: white;
                padding: 10px;
                font-size: 15px;
            }
        """)                    
    
        right_layout = QVBoxLayout()

        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
                SELECT
                F.File_name,
                F.File_path,
                P.PostID,
                P.CategoryID AS PostCategoryID,
                P.Title,
                P.CreatedAt AS PostCreatedAt,
                P.UpdatedAt AS PostUpdatedAt,
                PR.prediction_id,
                PR.CategoryID AS PredictionCategoryID,
                PR.prediction_result,
                PR.CreatedAt AS PredictionCreatedAt
            FROM
                "File" F
            LEFT JOIN Experience_File_Management EFM ON F.File_name = EFM.File_name
            LEFT JOIN Post P ON EFM.PostID = P.PostID
            LEFT JOIN Predictions PR ON F.File_name = PR.File_name
            ORDER BY
                F.File_name ASC    
        """ 
        
        cursor.execute(sql)
        file_rows = cursor.fetchall()

        self.file_col_num = len(file_rows[0])  ### 컬럼 갯수
        self.file_rows_num = len(file_rows)  ### 행 갯수

        self.file_list = []
        for i in range(0, len(file_rows), 1):
            file_mk_list = list(file_rows[i])
            self.file_list.append(file_mk_list)

        cursor.close()
        conn.close()

        ### 테이블에 데이터 넣기
        # 스크롤 가능하게 설정한 테이블
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.table = QTableWidget()
        self.table.setRowCount(self.file_rows_num)
        self.table.setColumnCount(self.file_col_num)
        self.table.setHorizontalHeaderLabels(["파일명", "파일경로", "경험공유 글번호", "범죄유형", "경험공유 글제목", "경험공유 생성일", "경험공유 수정일", "예측번호", "범죄유형", "예측결과", "AI예측 생성일"])

        data = self.file_list

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
                
        # 여기서 각 열의 너비를 설정
        self.table.setColumnWidth(0, 250)  # 파일명 열 너비
        self.table.setColumnWidth(1, 250)  # 파일경로 열 너비
        self.table.setColumnWidth(2, 160)  # 경험공유 글번호 열 너비
        self.table.setColumnWidth(3, 150)  # 범죄유형 열 너비         
        self.table.setColumnWidth(4, 150)  # 경험공유 글제목 열 너비         
        self.table.setColumnWidth(5, 150)  # 경험공유 생성일 열 너비         
        self.table.setColumnWidth(6, 150)  # 경험공유 수정일 열 너비            
        self.table.setColumnWidth(7, 150)  # 예측번호 열 너비            
        self.table.setColumnWidth(8, 150)  # 범죄유형 열 너비            
        self.table.setColumnWidth(9, 150)  # 예측결과 열 너비            
        self.table.setColumnWidth(10, 150)  # AI예측 생성일 열 너비            
                

        self.table.setFixedHeight(800)


        # 테이블 인덱스 배경색 변경
        self.table.setStyleSheet("""
        QTableWidget {
            background-color: #2b2b2b;
            color: white;
            border: 2px solid #3c3c3c;
            border-radius: 15px;
            gridline-color: #5a5a5a;
            font-size: 14px;
        }
        QTableWidget::item {
            padding: 10px;
            border: none;
        }
        QHeaderView::section {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #5a5a5a;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        QTableWidget::item:selected {
            background-color: #FFA500;
            color: black;
        }
        QTableCornerButton::section {
            background-color: #3c3c3c;
            border: 1px solid #5a5a5a;
        }
        QHeaderView {
            background-color: #3c3c3c;
        }
        QHeaderView::section:horizontal, QHeaderView::section:vertical {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #5a5a5a;
        }
        QTableWidget::viewport {
            background-color: #2b2b2b;
        }
    """)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # 사용자 인터랙션을 통해 간격 조절 가능
        header.setStretchLastSection(False)  # 마지막 컬럼이 자동으로 확장되지 않도록 설정
        header.setFixedHeight(100)  ### 테이블 컬럼명 세로 크기(높이)
        right_layout.addWidget(self.table)

        # 수정, 삭제 버튼 추가
        button_layout = QHBoxLayout()
        update_button = QPushButton("수정")
        update_button.setFixedHeight(70)
        update_button.setFixedWidth(250)
        update_button.clicked.connect(lambda: self.show_action_window("수정", "수정하시겠습니까?", "수정"))
        
        

        delete_button = QPushButton("삭제")
        delete_button.setFixedHeight(70)
        delete_button.setFixedWidth(250)
        delete_button.clicked.connect(self.delete_selected_row)
        
        update_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        delete_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        button_layout.addWidget(update_button)
        button_layout.addWidget(delete_button)
        
        # 테이블에서 선택된 행의 데이터를 추출
        self.table.itemSelectionChanged.connect(self.get_selected_row_data)
        
        right_layout.addLayout(button_layout)

        layout.addWidget(left_widget)
        layout.addLayout(self.central_layout)
        layout.addLayout(right_layout)

        layout.setStretch(0, 2)
        layout.setStretch(1, 4)
        layout.setStretch(2, 4)

        widget.setLayout(layout)
        
        
    def refresh_main_window(self):
        self.setup_ui()  # setup_ui를 호출하여 새로고침 효과를 적용   
        
    
    
    # 버튼을 클릭할 때 다른 버튼들의 스타일을 초기화하고, 클릭된 버튼의 스타일을 변경하는 함수
    def handle_button_click(self, clicked_button, tab_name):
        # 모든 버튼의 스타일을 초기화
        buttons = [self.community_button, self.notice_button, self.aiPredict_button, self.user_feedback_button]
        for button in buttons:
            button.setStyleSheet(self.get_default_button_style())

        # 클릭된 버튼의 스타일을 활성화 상태로 변경
        clicked_button.setStyleSheet(self.get_selected_button_style())

        # 현재 탭 이름을 설정
        self.current_tab_name = tab_name  # 현재 클릭된 탭 이름을 저장

        # 클릭된 버튼에 따라 레이아웃 업데이트
        if tab_name == "경험공유":
            self.update_community_layout()
        elif tab_name == "공지사항":
            self.update_notice_layout()
            self.notice_button.setStyleSheet(self.get_selected_button_style())
        elif tab_name == "AI 예측 결과":
            self.update_aiPredict_layout()
        elif tab_name == "사용자 피드백":
            self.update_user_feedback_layout()
        
        # QTabWidget의 스타일을 설정하여 테두리 색상을 회색(#3c3c3c)으로 유지
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-family: 'Malgun Gothic';
            }    
            QTabWidget::pane {
                border: 2px solid #3c3c3c;  /* 진한 회색 테두리 */
            }
             QWidget {
                font-family: 'Malgun Gothic';  # 탭 안에 포함된 위젯의 글씨체 적용
            }
        """)


    # 기본 버튼 스타일 정의
    def get_default_button_style(self):
        return """
            QPushButton {
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """

    # 선택된 버튼 스타일 정의
    def get_selected_button_style(self):
        return """
            QPushButton {
                background-color: orange;
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
        """ 
        
               
        
    # 공지사항 레이아웃을 업데이트하는 메소드
    def update_notice_layout(self):
        self.clear_central_layout()

        # 공지사항 폼을 중앙에 배치
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("공지사항 제목")
        self.title_input.setFixedHeight(50)
        self.title_input.setFixedWidth(600)
        self.title_input.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
                padding-left: 10px;
            }
        """)

        self.content_input = QTextEdit()
        self.content_input.setPlaceholderText("공지사항 내용을 입력하세요")
        self.content_input.setFixedHeight(670)
        self.content_input.setFixedWidth(600)
        self.content_input.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
                padding: 10px;
            }
        """)

        submit_button = QPushButton("입력하기")
        submit_button.setFixedHeight(70)
        submit_button.setFixedWidth(600)
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        submit_button.clicked.connect(lambda: self.submit_notice(self.title_input.text(), self.content_input.toPlainText()))

        central_layout.addWidget(self.title_input, alignment=Qt.AlignCenter)
        central_layout.addWidget(self.content_input, alignment=Qt.AlignCenter)
        central_layout.addWidget(submit_button, alignment=Qt.AlignCenter)

        self.clear_central_layout()
        self.central_layout.addWidget(central_widget)
        
    
    def submit_notice(self, title, content):
        
        # 오라클 데이터베이스에 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com", 1521, "orcl")
        conn = cx_Oracle.connect("admin", "1emddlqslek", dsn)
        cursor = conn.cursor()

        # 로그인한 사용자의 아이디
        user_id = self.logged_in_user_id    
        

        # 데이터베이스에 삽입할 SQL 쿼리 작성
        sql = """
            INSERT INTO notice (ADMINID, TITLE, CONTENT, CREATEDDATE)
            VALUES (:user_id, :title, :content, CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE))
        """

        try:
            cursor.execute(sql, user_id=user_id, title=title, content=content)
            conn.commit()  # 변경 사항을 커밋하여 데이터베이스에 적용

            # 성공 팝업 호출
            self.success_window = NoticeSuccessWindow(self)
            self.success_window.show()

        except cx_Oracle.DatabaseError as e:
            # 오류 팝업 호출
            error_window = NoticeErrorWindow(str(e), self)
            error_window.show()
        finally:
            cursor.close()
            conn.close()

        # 입력 폼 초기화
        self.clear_notice_form()       
            

    def clear_notice_form(self):
        # 제목과 내용 입력 필드를 초기화
        self.title_input.clear()
        self.content_input.clear()
        
        
    def show_logout_confirmation(self):
        self.show_action_window("로그아웃", "로그아웃 하시겠습니까?", "로그아웃")


    def get_selected_row_data(self):
        """
        테이블에서 선택된 행의 데이터를 저장합니다.
        """
        selected_row = self.table.currentRow()
        self.selected_row_data = []
        
        for col in range(self.table.columnCount()):
            item = self.table.item(selected_row, col)
            if item is not None:
                self.selected_row_data.append(item.text())


    def delete_selected_row(self):
        """
        선택된 행의 데이터를 가져와서 show_action_window 함수로 전달합니다.
        """
        # print(self.selected_row_data)
    
        if not self.selected_row_data:
            warning_window = WarningWindow(self)  # self를 전달하여 부모로 설정
            warning_window.show()
            return
        
        # 현재 탭 이름을 가져옵니다
        current_tab = self.current_tab_name  # handle_button_click에서 설정한 current_tab_name 사용
        
        # show_action_window 호출에 선택한 행 데이터 전달
        self.show_action_window("삭제", "삭제하시겠습니까?", "삭제", self.selected_row_data, current_tab)
        
        # 삭제 후에도 현재 탭 유지하기 위한 업데이트
        if current_tab == "공지사항":
            self.update_notice_layout()
        elif current_tab == "경험공유":
            self.update_community_layout()
        elif current_tab == "AI 예측 결과":
            self.update_aiPredict_layout()
        elif current_tab == "사용자 피드백":
            self.update_user_feedback_layout()
            
        # 다시 그려질 때 테두리가 생기지 않도록 중간 레이아웃 스타일 초기화
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3c3c3c;  /* 검정에 가까운 진회색 테두리 */
            }
            QTabBar::tab {
                font-family: 'Malgun Gothic';
                color: white;
                padding: 10px;
                font-size: 15px;
            }
        """)


############################################################################################################################################


    def clear_central_layout(self):
        """
        중간 레이아웃의 위젯을 모두 제거합니다.
        """
        for i in reversed(range(self.central_layout.count())):
            widget = self.central_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
                
    def update_community_layout(self):
        """
        경험공유 버튼 클릭 시 중간 레이아웃에 경험공유 시각화를 보여줍니다.
        오른쪽 테이블에는 경험공유 데이터를 표시합니다.
        """
        self.show_community_visualization()
        self.update_community_table()            
                
                
    def update_notice_layout(self):
        """
        공지사항 버튼 클릭 시 호출되어 중간 레이아웃을 공지사항 입력 폼으로 업데이트하고,
        오른쪽 테이블은 공지사항 데이터로 업데이트합니다.
        """
        # 중간 레이아웃을 공지사항 입력 폼으로 변경
        self.show_notice_layout()
        # 오른쪽 테이블을 공지사항 데이터로 업데이트
        self.update_notice_table()


    def update_aiPredict_layout(self):
        """
        AI 예측 결과 버튼 클릭 시 중간 레이아웃에 AI 예측 결과 시각화를 보여줍니다.
        오른쪽 테이블에는 AI 예측 결과 데이터를 표시합니다.
        """
        self.show_aiPredict_visualization()
        self.update_aiPredict_table()


    def update_user_feedback_layout(self):
        """
        사용자 피드백 버튼 클릭 시 중간 레이아웃에 사용자 피드백 시각화를 보여줍니다.
        오른쪽 테이블에는 사용자 피드백 데이터를 표시합니다.
        """
        self.show_feedback_visualization()
        self.update_user_feedback_table()
        
        
               
#######################################################################################################################################
        
        
        

    def update_community_table(self):
        """
        경험공유 버튼 클릭 시 호출되어 테이블의 데이터를 경험공유 데이터로 업데이트합니다.
        """
        
        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            Select *
            From post
            ORDER BY PostID ASC
        """
        cursor.execute(sql)
        community_rows = cursor.fetchall()

        self.community_col_num = len(community_rows[0])  ### 컬럼 갯수
        self.community_rows_num = len(community_rows)    ### 행 갯수

        self.community_list = []
        for i in range(0, len(community_rows), 1):
            community_mk_list = list(community_rows[i])
            self.community_list.append(community_mk_list)

        cursor.close()
        conn.close()          
        
        
        community_data = self.community_list
        self.table.setRowCount(self.community_rows_num)
        self.table.setColumnCount(self.community_col_num)
        self.table.setHorizontalHeaderLabels(["글 번호", "카테고리", "제목", "내용", "비밀번호", "생성일", "수정일"])

        for i, row in enumerate(community_data):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
                
                
        # 여기서 각 열의 너비를 설정
        self.table.setColumnWidth(0, 80)  # 글 번호 열 너비
        self.table.setColumnWidth(1, 90)  # 카테고리 열 너비
        self.table.setColumnWidth(2, 160)  # 제목 열 너비
        self.table.setColumnWidth(3, 250)  # 내용 열 너비         
        self.table.setColumnWidth(4, 90)  # 비밀번호 열 너비         
        self.table.setColumnWidth(5, 100)  # 생성일 열 너비         
        self.table.setColumnWidth(6, 100)  # 수정일 열 너비         
                
    
    def update_notice_table(self):
        """
        공지사항 버튼 클릭 시 호출되어 테이블의 데이터를 공지사항으로 업데이트합니다.
        """
        
        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            Select *
            From notice
            ORDER BY NoticeID ASC
        """
        cursor.execute(sql)
        notice_rows = cursor.fetchall()

        self.notice_col_num = len(notice_rows[0])  ### 컬럼 갯수
        self.notice_rows_num = len(notice_rows)    ### 행 갯수

        self.notice_list = []
        for i in range(0, len(notice_rows), 1):
            notice_mk_list = list(notice_rows[i])
            self.notice_list.append(notice_mk_list)

        cursor.close()
        conn.close()        
        
        
        notice_data = self.notice_list
        self.table.setRowCount(self.notice_rows_num)
        self.table.setColumnCount(self.notice_col_num)
        self.table.setHorizontalHeaderLabels(["공지글 번호", "관리자ID", "제목", "내용", "생성일", "수정일"])

        for i, row in enumerate(notice_data):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
                
        # 여기서 각 열의 너비를 설정
        self.table.setColumnWidth(0, 110)  # 공지글 번호 열 너비
        self.table.setColumnWidth(1, 100)  # 관리자ID 열 너비
        self.table.setColumnWidth(2, 140)  # 제목 열 너비
        self.table.setColumnWidth(3, 170)  # 내용 열 너비                 
        self.table.setColumnWidth(4, 100)  # 생성일 열 너비         
        self.table.setColumnWidth(5, 100)  # 수정일 열 너비        
                 
                
                
    def update_aiPredict_table(self):
        """
        AI 예측 결과 버튼 클릭 시 호출되어 테이블의 데이터를 AI 예측 결과 데이터로 업데이트합니다.
        """
        
        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            SELECT p.PREDICTION_ID, 
                c.CategoryName, 
                p.PREDICTION_RESULT, 
                p.CONFIDENCE_SCORE, 
                p.FILE_NAME, 
                p.CREATEDAT
            FROM predictions p
            JOIN Category c 
                ON p.CATEGORYID = c.CategoryID
            Order By p.PREDICTION_ID ASC
        """
        cursor.execute(sql)
        aiPredict_rows = cursor.fetchall()

        self.aiPredict_col_num = len(aiPredict_rows[0])  ### 컬럼 갯수
        self.aiPredict_rows_num = len(aiPredict_rows)    ### 행 갯수

        self.aiPredict_list = []
        for i in range(0, len(aiPredict_rows), 1):
            aiPredict_mk_list = list(aiPredict_rows[i])
            self.aiPredict_list.append(aiPredict_mk_list)

        cursor.close()
        conn.close() 
        
        ai_predict_data = self.aiPredict_list
        self.table.setRowCount(self.aiPredict_rows_num)
        self.table.setColumnCount(self.aiPredict_col_num)
        self.table.setHorizontalHeaderLabels(["예측 번호", "범죄유형", "AI 예측 결과", "신뢰도", "파일이름", "생성일"])

        for i, row in enumerate(ai_predict_data):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
                
                
        # 여기서 각 열의 너비를 설정
        self.table.setColumnWidth(0, 120)  # 예측 번호 열 너비
        self.table.setColumnWidth(1, 110)  # 범죄유형 열 너비
        self.table.setColumnWidth(2, 140)  #  AI 판단 결과 열 너비
        self.table.setColumnWidth(3, 110)  # 신뢰도 열 너비                 
        self.table.setColumnWidth(4, 110)  # 파일이름 열 너비         
        self.table.setColumnWidth(5, 110)  # 생성일 열 너비           
                
                
    
    def update_user_feedback_table(self):
        """
        사용자 피드백 버튼 클릭 시 호출되어 테이블의 데이터를 사용자 피드백 데이터로 업데이트합니다.
        """
        
        # Database 연결 설정
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        sql = """
            Select *    
            From Feedback
            ORDER BY Feedback_ID ASC
        """
        cursor.execute(sql)
        feedback_rows = cursor.fetchall()

        self.feedback_col_num = len(feedback_rows[0])  ### 컬럼 갯수
        self.feedback_rows_num = len(feedback_rows)    ### 행 갯수

        self.feedback_list = []
        for i in range(0, len(feedback_rows), 1):
            feedback_mk_list = list(feedback_rows[i])
            self.feedback_list.append(feedback_mk_list)
        
        
        feedback_data = self.feedback_list
        self.table.setRowCount(self.feedback_rows_num)
        self.table.setColumnCount(self.feedback_col_num)
        self.table.setHorizontalHeaderLabels(["피드백 번호", "만족도", "내용", "생성일"])

        cursor.close()
        conn.close() 
        
        for i, row in enumerate(feedback_data):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
                
        # 여기서 각 열의 너비를 설정
        self.table.setColumnWidth(0, 110)  # 피드백 번호 열 너비
        self.table.setColumnWidth(1, 100)  # 만족도 열 너비
        self.table.setColumnWidth(2, 350)  # 내용 열 너비를 600으로 설정 (원하는 크기로 조정)
        self.table.setColumnWidth(3, 130)  # 생성일 열 너비     
                                                   
                

    def show_visualization_layout(self):
        """
        중간 레이아웃을 시각화 레이아웃으로 복원합니다.
        """
        self.clear_central_layout()

        self.community_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
            }
        """)
        self.notice_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border-radius: 15px;
                color: white;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)

        self.tabs = QTabWidget()


        # 각 그래프 생성 시 매개변수를 전달
        self.bar_chart_view = self.create_bar_chart_matplotlib(
            ['월', '화', '수', '목', '금'],  # categories
            [10, 20, 30, 40, 50],  # values
            "요일별 접속 현황"  # title
        )
        self.pie_chart_view = self.create_pie_chart_matplotlib(
            ['딥페이크', '보이스피싱', '문자스미싱'],  # labels
            [40, 35, 25],  # sizes
            "사이버 범죄별 비율"  # title
        )
        self.line_chart_view = self.create_line_chart_matplotlib(
            [1, 2, 3, 4, 5],  # x
            [10, 30, 20, 50, 40],  # y
            "시간에 따른 데이터 변화"  # title
        )

        self.tabs.addTab(self.bar_chart_view, "막대 그래프")
        self.tabs.addTab(self.pie_chart_view, "파이 그래프")
        self.tabs.addTab(self.line_chart_view, "선 그래프")

        self.clear_central_layout()
        self.central_layout.addWidget(self.tabs)
        
        
    def show_community_visualization(self):
        """
        경험공유 버튼 클릭 시 중간 레이아웃에 3개의 막대 그래프를 보여줍니다.
        """
        self.clear_central_layout()
        
        self.tabs = QTabWidget()
        
        
        ### DB 불러오기
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        ### 총합
        sql = """
            Select count(*)
            From Post    
        """

        cursor.execute(sql)
        total_cnt = cursor.fetchone()[0]


        ### 딥페이크
        sql = """
            Select count(*)
            From Post
            Where CATEGORYID = 1    
        """

        cursor.execute(sql)
        deepfake_cnt = cursor.fetchone()[0]


        ### 보이스피싱
        sql = """
            Select count(*)
            From Post
            Where CATEGORYID = 2    
        """

        cursor.execute(sql)
        voice_cnt = cursor.fetchone()[0]


        ### 문자스미싱
        sql = """
            Select count(*)
            From Post
            Where CATEGORYID = 3    
        """

        cursor.execute(sql)
        smishing_cnt = cursor.fetchone()[0]
        
        
        ### post 경험공유 게시글 평균길이
        sql = """
            SELECT CATEGORYID, ROUND(AVG(LENGTH(content)), 0) AS 평균길이
            FROM post
            GROUP BY CATEGORYID
            
        """ 

        cursor.execute(sql)
        total_data = cursor.fetchall() 
        
        ### post 경험공유 게시글 평균길이
        sql = """
            SELECT TO_CHAR(CREATEDAT, 'YYYY-MM-DD') AS post_date, COUNT(*) AS post_count
            FROM post
            WHERE CREATEDAT >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE)) - 10
            GROUP BY TO_CHAR(CREATEDAT, 'YYYY-MM-DD')
            ORDER BY post_date            
        """ 

        cursor.execute(sql)
        total_data2 = cursor.fetchall() 

        cursor.close()
        conn.close()
        
        
        ### 사이버 범죄별 게시물 수
        cnt_list= []
        categories = ["딥페이크", "보이스피싱", "문자스미싱"]

        for i in total_data:
            cnt_list.append(i[1]) 
            
        
        ### 날짜별 작성 게시물 수    
        date_list= []
        num_list = []

        for i in total_data2:
            date_list.append(i[0])
            
        for i in total_data2:
            num_list.append(i[1])              
        
               

        # 첫 번째 막대 그래프 
        chart_view1 = self.create_pie_chart_matplotlib(
                                                            ['딥페이크', '보이스피싱', '문자스미싱'], [deepfake_cnt, voice_cnt, smishing_cnt], "사이버 범죄별 경험공유 업로드 분석"
                                                        )
        # 두 번째 막대 그래프
        chart_view2 = self.create_simple_bar_chart_matplotlib(
                                                            categories, cnt_list, "사이버 범죄별 경험공유 평균 글 길이"
                                                        )
        
        # 세 번째 막대 그래프
        chart_view3 = self.create_line_chart_matplotlib(
                                                            date_list, num_list, "날짜별 경험공유 작성 수 변화"
                                                        )

        self.tabs.addTab(chart_view1, "사이버 범죄별 경험공유 업로드 분석")
        self.tabs.addTab(chart_view2, "사이버 범죄별 경험공유 평균 글 길이")
        self.tabs.addTab(chart_view3, "날짜별 경험공유 작성 수 변화")

        self.central_layout.addWidget(self.tabs)


    def show_aiPredict_visualization(self):
        """
        AI 예측 결과 버튼 클릭 시 중간 레이아웃에 3개의 파이 그래프를 보여줍니다.
        """
        self.clear_central_layout()
        
        self.tabs = QTabWidget()
        
        ### DB 불러오기
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()

        ### 딥페이크
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 1
        """

        cursor.execute(sql)
        deepfake_num = cursor.fetchone()[0]

        ### 보이스피싱
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 2
        """

        cursor.execute(sql)
        voice_num = cursor.fetchone()[0]

        ### 문자스미싱
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 3
        """

        cursor.execute(sql)
        smishing_num = cursor.fetchone()[0]
        
        
        ### 딥페이크 real 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 1 and PREDICTION_RESULT = 'real'   
        """

        cursor.execute(sql)
        df_real_cnt = cursor.fetchone()[0]

        ### 딥페이크 fake 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 1 and PREDICTION_RESULT = 'fake'    
        """

        cursor.execute(sql)
        df_fake_cnt = cursor.fetchone()[0]

        ### 보이스피싱 real 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 2 and PREDICTION_RESULT = 'real'   
        """

        cursor.execute(sql)
        voice_real_cnt = cursor.fetchone()[0]

        ### 보이스피싱 fake 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 2 and PREDICTION_RESULT = 'fake'    
        """

        cursor.execute(sql)
        voice_fake_cnt = cursor.fetchone()[0]


        ### 문자스미싱 real 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 3 and PREDICTION_RESULT = 'real'   
        """

        cursor.execute(sql)
        smishing_real_cnt = cursor.fetchone()[0]

        ### 문자스미싱 fake 카운트
        sql = """
            Select count(*)
            From Predictions
            Where CATEGORYID = 3 and PREDICTION_RESULT = 'fake'    
        """

        cursor.execute(sql)
        smishing_fake_cnt = cursor.fetchone()[0]
        
        sql = """
            SELECT TO_CHAR(CREATEDAT, 'YYYY-MM-DD') AS 탐지일, COUNT(*) AS 탐지건수
            FROM Predictions
            WHERE CREATEDAT >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE)) - 10
            GROUP BY TO_CHAR(CREATEDAT, 'YYYY-MM-DD')
            ORDER BY 탐지일
        """

        cursor.execute(sql)
        date_total = cursor.fetchall()
        
        
        ### 딥페이크 신뢰도
        sql = """
            SELECT CONFIDENCE_SCORE
            FROM Predictions
            Where CATEGORYID = 1
            
        """

        cursor.execute(sql)
        deepfake_confidence = cursor.fetchall()


        ### 보이스피싱 신뢰도
        sql = """
            SELECT CONFIDENCE_SCORE
            FROM Predictions
            Where CATEGORYID = 2
            
        """

        cursor.execute(sql)
        voice_confidence = cursor.fetchall()

        ### 문자스미싱 신뢰도
        sql = """
            SELECT CONFIDENCE_SCORE
            FROM Predictions
            Where CATEGORYID = 3
            
        """

        cursor.execute(sql)
        smishing_confidence = cursor.fetchall()

        cursor.close()
        conn.close()        
        
        # 날짜별 AI 판단 숫자
        date_list = []
        count_list = []

        for i in date_total:
            date_list.append(i[0])
            count_list.append(i[1])
            
        
        # 사이버 범죄별 신뢰도 리스트
        deepfake_list = []
        voice_list = []
        smishing_list = []

        for i in deepfake_confidence:
            deepfake_list.append(i[0])
            
        for i in voice_confidence:
            voice_list.append(i[0])
                
        for i in smishing_confidence:
            smishing_list.append(i[0])     

        total_dict = {}
        
        total_dict["딥페이크"] = deepfake_list

        total_dict["보이스피싱"] = voice_list

        total_dict["문자스미싱"] = smishing_list       
            
        
        
        # 범죄유형별 AI 판단 결과 비율 분석을 위한 데이터
        categories = ["딥페이크", "보이스피싱", "문자스미싱"]
        real_counts = [df_real_cnt, voice_real_cnt, smishing_real_cnt]  # 예시로 real 판단 횟수
        fake_counts = [df_fake_cnt, voice_fake_cnt, smishing_fake_cnt]  # 필요에 따라 fake 판단 데이터도 추가 가능
        
        

        # 첫 번째 그룹 막대 그래프
        chart_view1 = self.create_grouped_bar_chart(categories, real_counts, fake_counts, "범죄유형별 AI 판단 결과 비율 분석")
        
        
        # 두 번째 파이 그래프
        chart_view2 = self.create_scatter_plot(categories, total_dict, "사이버범죄 유형별 신뢰도 분포")
        
        # 세 번째 막대 그래프
        chart_view3 = self.create_bar_chart_matplotlib(["딥페이크", '보이스피싱', '문자스미싱'], [deepfake_num, voice_num, smishing_num], "사이버 범죄별 AI 탐지 빈도 분석")
   
        # 네 번째 선 그래프
        chart_view4 = self.create_line_chart_matplotlib(date_list, count_list, "날짜별 AI탐지 건수 시각화")

        self.tabs.addTab(chart_view1, "범죄유형별 AI 판단결과 비율 분석")
        self.tabs.addTab(chart_view2, "사이버범죄 유형별 신뢰도 분포")
        self.tabs.addTab(chart_view3, "사이버 범죄별 AI 탐지 빈도 분석")
        self.tabs.addTab(chart_view4, "날짜별 AI탐지 건수 시각화")

        self.central_layout.addWidget(self.tabs)


    def show_feedback_visualization(self):
        """
        사용자 피드백 버튼 클릭 시 중간 레이아웃에 3개의 선 그래프를 보여줍니다.
        """
        self.clear_central_layout()
        
        self.tabs = QTabWidget()
        
        
        ### DB 불러오기
        dsn = cx_Oracle.makedsn("true-db.c3awi8yo8lit.ap-northeast-2.rds.amazonaws.com",1521,"orcl")
        conn = cx_Oracle.connect("admin","1emddlqslek", dsn)
        cursor = conn.cursor()
        
        ### 총 행의 갯수
        sql = """
            Select count(*)
            From feedback
        """

        cursor.execute(sql)
        total_num = cursor.fetchone()[0]


        ### 만족도 1일 때 
        sql = """
            Select count(*)
            From feedback
            Where SATISFACTION_LEVEL = 1
        """

        cursor.execute(sql)
        satisfy_1_num = cursor.fetchone()[0]

        ### 만족도 2일 때
        sql = """
            Select count(*)
            From feedback
            Where SATISFACTION_LEVEL = 2
        """

        cursor.execute(sql)
        satisfy_2_num = cursor.fetchone()[0]


        ### 만족도 3일 때
        sql = """
            Select count(*)
            From feedback
            Where SATISFACTION_LEVEL = 3
        """

        cursor.execute(sql)
        satisfy_3_num = cursor.fetchone()[0]
        
        
        ### 작성일별 작성건수
        sql = """
            SELECT TO_CHAR(CREATEDAT, 'YYYY-MM-DD') AS 작성일, COUNT(*) AS 작성건수
            FROM feedback
            WHERE CREATEDAT >= TRUNC(CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS DATE)) - 10
            GROUP BY TO_CHAR(CREATEDAT, 'YYYY-MM-DD')
            ORDER BY 작성일
        """

        cursor.execute(sql)
        date_total = cursor.fetchall()
                
        cursor.close()
        conn.close()
        
        date_list = []
        count_list = []

        for i in date_total:
            date_list.append(i[0])
            count_list.append(i[1])
                

        # 첫 번째 선 그래프
        # line_chart_view1 = self.create_bar_chart_matplotlib([0, 1, 2, 3], [10, 20, 30, 40], "사용자 피드백 막대 그래프 1")
        
        # 두 번째 파이 그래프
        line_chart_view2 = self.create_pie_chart_matplotlib(["불만족", "보통", "만족"], [satisfy_1_num, satisfy_2_num, satisfy_3_num], "사이트 사용자 만족도 분석")
        
        # 세 번째 선 그래프
        line_chart_view3 = self.create_line_chart_matplotlib(date_list, count_list, "작성 일별 사용자 피드백 시각화")

        # self.tabs.addTab(line_chart_view1, "막대 그래프 1")
        self.tabs.addTab(line_chart_view2, "사이트 사용자 만족도 분석")
        self.tabs.addTab(line_chart_view3, "작성 일별 사용자 피드백 시각화")

        self.central_layout.addWidget(self.tabs)       
               
        
        
    def show_notice_layout(self):
        # 공지사항 입력 레이아웃
        self.clear_central_layout()

        # self.title_input을 클래스 속성으로 변경
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("공지사항 제목")
        self.title_input.setFixedHeight(50)
        self.title_input.setFixedWidth(600)
        self.title_input.setStyleSheet("""
            QLineEdit {
                font-family: 'Malgun Gothic';
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
                padding-left: 10px;
            }
        """)

        # self.content_input을 클래스 속성으로 변경
        self.content_input = QTextEdit()
        self.content_input.setPlaceholderText("공지사항 내용을 입력하세요")
        self.content_input.setFixedHeight(670)
        self.content_input.setFixedWidth(600)
        self.content_input.setStyleSheet("""
            QTextEdit {
                font-family: 'Malgun Gothic';
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
                padding: 10px;
            }
        """)

        submit_button = QPushButton("입력하기")
        submit_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 20px;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        submit_button.setFixedHeight(70)
        submit_button.setFixedWidth(600)
        
        # submit_notice에 self.title_input과 self.content_input 연결
        submit_button.clicked.connect(lambda: self.submit_notice(self.title_input.text(), self.content_input.toPlainText()))

        self.notice_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 20px;
                color: white;
            }
        """)
        self.community_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 20px;
                color: white;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)

        self.title_input.setStyleSheet("""
            QLineEdit {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 10px;
                color: white;
                padding-left: 10px;
            }
        """)
        self.content_input.setStyleSheet("""
            QTextEdit {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                border-radius: 10px;
                color: white;
                padding: 10px;
            }
        """)
        submit_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                background-color: #3c3c3c;
                color: white;
                border-radius: 15px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setAlignment(Qt.AlignCenter)
        central_layout.setSpacing(20)
        central_layout.addWidget(self.title_input, alignment=Qt.AlignCenter)  # 클래스 속성 사용
        central_layout.addWidget(self.content_input, alignment=Qt.AlignCenter)  # 클래스 속성 사용
        central_layout.addWidget(submit_button, alignment=Qt.AlignCenter)

        self.clear_central_layout()
        self.central_layout.addWidget(central_widget)

    
            

    def clear_central_layout(self):
        """
        중간 레이아웃을 비우는 함수.
        """
        for i in reversed(range(self.central_layout.count())):
            widget = self.central_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
                

    def show_retrain_model_window(self):
        self.retrain_window = RetrainModelWindow(self)
        self.retrain_window.show()


    def show_action_window(self, title, message, action_label, selected_row_data=None, current_tab=None):
    # selected_row_data 인자가 전달된 경우, 이를 사용하여 윈도우에 표시하거나 작업 수행
        # print(f"Selected Row Data for Action: {selected_row_data}")
        self.action_window = ActionWindow(title, message, action_label, self, selected_row_data, current_tab)
        self.action_window.show()

    
          
    #######################################################################################################################
    ### 시각화
    
    # 공통 스타일 설정 함수 정의
    def apply_common_style(self, fig, ax, title, xlabel, ylabel):
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        ax.set_title(title, fontsize=18, weight='bold', color='white')
        ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='white')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='white')
        ax.tick_params(axis='x', colors='white', labelsize=12)
        ax.tick_params(axis='y', colors='white', labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#2b2b2b')
        ax.spines['right'].set_color('#2b2b2b')
        fig.subplots_adjust(top=0.85, bottom=0.2)
        
        
        
    # matplotlib을 사용하여 막대 그래프 생성
    def create_bar_chart_matplotlib(self, categories, values, title):
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        COLORS = ['#A29BFE', '#FF7675', '#55EFC4']
        self.apply_common_style(fig, ax, title, "범죄 종류", "AI 탐지 횟수")

        bars = ax.bar(categories, values, color=[COLORS[i % len(COLORS)] for i in range(len(values))], edgecolor='#3c3c3c', linewidth=1.2)
        ax.set_ylim(0, max(values) * 1.3)

        total = sum(values)
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100 if total != 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, height * 1.02, f'{height:.0f}', ha='center', va='bottom', fontsize=13, color='white', weight='bold')
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{percentage:.1f}%', ha='center', va='center', fontsize=14, color='white', weight='bold')

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)
    
    
    
    # 퍼센트 없이 값만 표시하는 막대 그래프 생성
    def create_simple_bar_chart_matplotlib(self, categories, values, title):
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        self.apply_common_style(fig, ax, title, "범죄 종류", "글자 수")
        COLORS = ['#A29BFE', '#FF7675', '#55EFC4']

        bars = ax.bar(categories, values, color=[COLORS[i % len(COLORS)] for i in range(len(values))], edgecolor='#3c3c3c', linewidth=1.2)
        ax.set_ylim(0, max(values) * 1.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height * 1.02, f'{height:.0f}글자', ha='center', va='bottom', fontsize=13, color='white', weight='bold')

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)
    
    
    # matplotlib을 사용하여 파이 그래프 생성
    def create_pie_chart_matplotlib(self, labels, sizes, title):
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')

        explode = [0.05] * len(sizes)
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']

        def func(pct, allvals):
            absolute = int(pct / 100. * sum(allvals))
            return f"{pct:.1f}%\n({absolute}건)"

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), startangle=140,
                                        colors=colors, explode=explode, wedgeprops=dict(edgecolor='#3c3c3c', linewidth=1.5))

        for text in texts:
            text.set_fontsize(12)
            text.set_color('white')
            text.set_weight('bold')
        for autotext in autotexts:
            lines = autotext.get_text().split('\n')
            autotext.set_text(f"{lines[0]}\n{lines[1]}")
            autotext.set_color('white')
            autotext.set_fontsize(19 if '\n' in autotext.get_text() else 12)
            autotext.set_weight('bold')

        ax.set_title(title, fontsize=20, weight='bold', color='white')
        fig.subplots_adjust(top=0.85, bottom=0.15)

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)
    
    
    # matplotlib을 사용하여 선 그래프 생성
    def create_line_chart_matplotlib(self, x, y, title):
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        self.apply_common_style(fig, ax, title, "날짜", "건수")

        ax.plot(x, y, marker='o', color='#FF7043', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)

        for i in range(len(x)):
            ax.text(x[i], y[i] + 0.008, f'{y[i]}건', ha='center', va='bottom', fontsize=14, color='white', weight='bold')

        # x축 레이블 회전 설정
        plt.xticks(rotation=45, ha='right')

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)
    
    # matplotlib을 사용하여 그룹화된 막대 그래프 생성
    def create_grouped_bar_chart(self, categories, real_counts, fake_counts, title):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#2b2b2b')
        self.apply_common_style(fig, ax, title, "범죄유형", "횟수")

        colors = ['#C3A6FF', '#C9E4C5']
        bar_width = 0.35
        index = range(len(categories))

        bars_real = ax.bar([i - bar_width / 2 for i in index], real_counts, bar_width, color=colors[0], label="Real", edgecolor='#3c3c3c', linewidth=1.2)
        bars_fake = ax.bar([i + bar_width / 2 for i in index], fake_counts, bar_width, color=colors[1], label="Fake", edgecolor='#3c3c3c', linewidth=1.2)
        ax.set_ylim(0, max(max(real_counts), max(fake_counts)) * 1.2)

        total_counts = [real + fake for real, fake in zip(real_counts, fake_counts)]

        for i, bars in enumerate([bars_real, bars_fake]):
            counts = real_counts if i == 0 else fake_counts
            for j, bar in enumerate(bars):
                height = bar.get_height()
                percentage = (height / total_counts[j]) * 100 if total_counts[j] != 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{percentage:.1f}%', ha='center', va='center', fontsize=12, color='white', weight='bold')
                ax.text(bar.get_x() + bar.get_width() / 2, height * 1.02, f'{height:.0f}', ha='center', va='bottom', fontsize=12, color='white')

        ax.legend(facecolor='#2b2b2b', edgecolor='white', fontsize=10, loc="upper right", labelcolor='white')

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)
    
    
    # matplotlib을 사용하여 산점도 생성
    def create_scatter_plot(self, categories, confidence_scores, title):
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        self.apply_common_style(fig, ax, title, "범죄 유형", "신뢰도")
        COLORS = ['#A29BFE', '#FF7675', '#55EFC4']

        for i, category in enumerate(categories):
            x_vals = [i + (0.05 * (j - len(confidence_scores[category]) / 2)) for j in range(len(confidence_scores[category]))]
            y_vals = confidence_scores[category]
            ax.scatter(x_vals, y_vals, color=COLORS[i % len(COLORS)], label=category, alpha=1, edgecolor='white', s=110, linewidth=1.5)

        ax.legend(facecolor='#2b2b2b', edgecolor='white', fontsize=10, loc="upper left", bbox_to_anchor=(1, 1), labelcolor='white')
        fig.subplots_adjust(top=0.85, bottom=0.25, right=0.8)

        plt.close(fig)  # 그래프 닫기
        return FigureCanvas(fig)    
    
    
    ###########################################################################################################################
    ###########################################################################################################################
    
    ### GA4 함수
    
    # GA4 보고서 실행 함수
    def run_ga4_report(self, service_account_file, property_id, dimensions, metrics):
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        client = BetaAnalyticsDataClient(credentials=credentials)

        request = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=dimensions,
            metrics=metrics,
            date_ranges=[{"start_date": "30daysAgo", "end_date": "today"}]
        )

        try:
            response = client.run_report(request)
        except Exception as e:
            print(f"보고서 가져오기 오류: {e}")
            return None

        data = []
        for row in response.rows:
            row_data = [value.value for value in row.dimension_values] + [float(value.value) for value in row.metric_values]
            data.append(row_data)

        return pd.DataFrame(data, columns=[dim['name'] for dim in dimensions] + [met['name'] for met in metrics])


    # 공통 스타일 적용 함수
    def apply_common_styling(self, fig, ax, title):
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        ax.set_title(title, fontsize=20, weight='bold', color='white', pad=20)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')   
    
    
    
    
    ### 접속자 도시 시각화 함수
    def create_city_chart(self, service_account_file, property_id, top_n=10):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "city"}],
            metrics=[{"name": "activeUsers"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        # 사용자 수가 많은 상위 top_n개 도시 선택
        top_cities = df.sort_values(by="activeUsers", ascending=False).head(top_n)

        # 시각화: 상위 top_n 대한민국 지역별 사용자 수 막대 그래프
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')

        # 파스텔 색상 리스트 생성
        pastel_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ffb3e6', 
                        '#c2c2f0', '#ffb3b3', '#c2f0c2', '#ffccff', '#ffda99']

        bars = ax.barh(top_cities["city"], top_cities["activeUsers"], color=pastel_colors, zorder=2)
        ax.set_xlabel("사용자 수(명)", fontsize=10, color='white')
        ax.set_ylabel("City", fontsize=8, color='white')
        ax.invert_yaxis()  # 막대 그래프 상위부터 표시
        ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=1)

        # 각 막대에 사용자 수 숫자 표시
        for bar in bars:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{int(bar.get_width())}', va='center', ha='left', color='white', fontsize=12)

        # 공통 스타일 적용
        self.apply_common_styling(fig, ax, "사용자 접속 도시 상위 TOP 10")

        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas
    
    

    ### 트래픽 소스별 사용자 비율 시각화
    def create_traffic_source_chart(self, service_account_file, property_id):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "sessionSource"}],
            metrics=[{"name": "activeUsers"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        # 트래픽 소스 시각화 (파이 차트)
        fig, ax = plt.subplots(facecolor='#2b2b2b')  # 전체 배경색 설정
        fig.patch.set_facecolor('#2b2b2b')

        # 색상 팔레트 정의
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        explode = [0.05] * len(df)  # 모든 조각을 약간 강조

        # 사용자 정의 함수로 퍼센트와 개수를 개별 스타일로 표시
        def func(pct, allvals):
            absolute = int(pct / 100. * sum(allvals))
            return f"{pct:.1f}%\n({absolute}명)"

        wedges, texts, autotexts = ax.pie(
            df["activeUsers"], labels=df["sessionSource"], autopct=lambda pct: func(pct, df["activeUsers"]), 
            startangle=140, colors=colors, explode=explode, wedgeprops=dict(edgecolor='#3c3c3c', linewidth=1.5)
        )

        # 텍스트 스타일 설정
        for text in texts:
            text.set_fontsize(12)
            text.set_color('white')  # 라벨을 흰색으로 설정하여 가독성 향상
            text.set_weight('bold')
        for autotext in autotexts:
            lines = autotext.get_text().split('\n')
            autotext.set_text(f"{lines[0]}\n{lines[1]}")
            autotext.set_color('white')
            autotext.set_fontsize(13)  # 퍼센트 글씨 크기 증가
            autotext.set_weight('bold')

        # 타이틀 스타일
        ax.set_title("트래픽 소스별 사용자 비율", fontsize=20, weight='bold', color='white')

        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas

    
    
    
    ### 신규 사용자와 재방문 사용자 비율 시각화
    def create_new_vs_returning_users_chart(self, service_account_file, property_id):
        # GA4 리포트 데이터 가져오기
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "newVsReturning"}],
            metrics=[{"name": "activeUsers"}]
        )

        # 데이터 유효성 검사
        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        # NaN 값 제거
        df = df.dropna(subset=['activeUsers', 'newVsReturning'])

        # 컬럼명 한글로 변경
        if len(df) > 0:
            df.loc[0, "newVsReturning"] = "신규방문자"
        if len(df) > 1:
            df.loc[1, "newVsReturning"] = "재방문자"
        if len(df) > 2:
            df.loc[2, "newVsReturning"] = "기타"

        # 차트 설정
        fig, ax = plt.subplots(facecolor='#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')

        colors = ['#a1cfff', '#ffb7b2', '#b8e4c9']
        explode = [0.05] * len(df)

        # 퍼센티지 및 사용자 수 표시 함수
        def func(pct, allvals):
            absolute = int(pct / 100. * sum(allvals))
            return f"{pct:.1f}%\n({absolute}명)"

        # 파이 차트 생성
        wedges, texts, autotexts = ax.pie(
            df["activeUsers"], labels=df["newVsReturning"], autopct=lambda pct: func(pct, df["activeUsers"]),
            startangle=140, colors=colors, explode=explode, wedgeprops=dict(edgecolor='#3c3c3c', linewidth=1.5)
        )

        # 텍스트 스타일 설정
        for text in texts:
            text.set_fontsize(15)
            text.set_color('white')
            text.set_weight('bold')
        for autotext in autotexts:
            lines = autotext.get_text().split('\n')
            autotext.set_text(f"{lines[0]}\n{lines[1]}")
            autotext.set_color('white')
            autotext.set_fontsize(19)
            autotext.set_weight('bold')

        # 차트 제목 설정
        ax.set_title("신규 사용자와 재방문 사용자 비율", fontsize=20, weight='bold', color='white')

        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas


    
    
    ### 페이지별 전환수 및 이탈률
    def create_page_conversion_bounce_chart(self, service_account_file, property_id):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "pagePath"}],
            metrics=[{"name": "conversions"}, {"name": "bounceRate"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')

        # 전환수 막대 그래프
        bars = ax1.bar(df["pagePath"], df["conversions"], color='#66b3ff', edgecolor='black', label='전환수')
        ax1.set_xlabel("페이지 경로", fontsize=12, color='white')
        ax1.set_ylabel("전환수", fontsize=12, color='#66b3ff')
        ax1.tick_params(axis='y', labelcolor='#66b3ff')
        ax1.tick_params(axis='x', colors='white')
        ax1.set_facecolor('#2b2b2b')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df["pagePath"], rotation=45, ha='right', fontsize=10, color='white')

        # 각 막대 위에 전환수 숫자 표시
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + max(df["conversions"]) * 0.02,
                    f'{int(height)}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

        # 이탈률 꺾은선 그래프
        ax2 = ax1.twinx()
        line, = ax2.plot(df["pagePath"], df["bounceRate"], color='#ff7f0e', marker='o', linestyle='-', linewidth=2, label='이탈률 (%)')
        ax2.set_ylabel("이탈률 (%)", fontsize=12, color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        ax2.tick_params(axis='x', colors='white')

        # 각 점 위에 이탈률 표시
        for i, (x, y) in enumerate(zip(df["pagePath"], df["bounceRate"])):
            ax2.text(i, y + max(df["bounceRate"]) * 0.02, f'{y:.1f}%', ha='center', va='bottom', color='#ff7f0e', fontsize=10, fontweight='bold')

        # y=0인 지점에 굵은 흰색 가로선 추가
        ax1.axhline(y=0, color='white', linewidth=2)

        # 제목 및 배경색 설정
        plt.title("페이지별 전환수 및 이탈률", fontsize=16, weight='bold', color='white', pad=20)

        # 범례 추가 (그래프 왼쪽 위)
        legend1 = ax1.legend(loc="upper left", facecolor='#2b2b2b', edgecolor='white')
        plt.setp(legend1.get_texts(), color='white')
        legend2 = ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.9), facecolor='#2b2b2b', edgecolor='white')
        plt.setp(legend2.get_texts(), color='white')

        # x축, y축 테두리선을 흰색으로 변경
        for spine in ax1.spines.values():
            spine.set_edgecolor('white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('white')

        # y축 0 부분의 점선을 흰색으로 변경
        ax1.grid(axis="y", linestyle="--", alpha=0.7, color='white')
        plt.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas

    
    
    ### 시간대별 사용자 참여도
    def create_user_engagement_chart(self, service_account_file, property_id):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "hour"}],
            metrics=[{"name": "userEngagementDuration"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        # 시간대 순서대로 정렬 (0-23시)
        df["hour"] = pd.to_numeric(df["hour"], errors='coerce')
        df = df.sort_values(by="hour").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ['#ff6347' if value == df["userEngagementDuration"].max() else '#99ccff' for value in df["userEngagementDuration"]]
        ax.bar(df["hour"], df["userEngagementDuration"], color=colors, edgecolor='black')
        ax.set_xlabel("시간대 (시)", fontsize=12, color='white')
        ax.set_ylabel("참여 시간 (초)", fontsize=12, color='white')
        ax.set_xticks(range(0, 24, 1))  # X축을 1시간 단위로 표시
        self.apply_common_styling(fig, ax, "시간대별 사용자 참여도")

        max_value = df["userEngagementDuration"].max()
        max_hour = df[df["userEngagementDuration"] == max_value]["hour"].values[0]
        ax.text(max_hour, max_value, f'Max: {max_value:.2f}초', ha='center', va='bottom', fontsize=10, color='white', weight='bold')

        ax.grid(axis="y", linestyle="--", alpha=0.7, color='white')
        plt.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas

    
 

    ### 상위 10개 페이지별 사용자 참여 시간
    def create_top_pages_engagement_chart(self, service_account_file, property_id):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "pagePath"}],
            metrics=[{"name": "userEngagementDuration"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        top_pages = df.sort_values(by="userEngagementDuration", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(top_pages["pagePath"], top_pages["userEngagementDuration"], marker='o', linestyle='-', color='#ff6347')
        ax.set_xticklabels(top_pages["pagePath"], rotation=45, ha='right', fontsize=10, color='white')
        ax.set_xlabel("페이지 경로", fontsize=12, color='white')
        ax.set_ylabel("사용자 참여 시간 (초)", fontsize=12, color='white')
        self.apply_common_styling(fig, ax, "상위 10개 페이지별 사용자 참여 시간")

        # 사용자 참여 시간을 시간 단위로 변환하여 굵고 크게 위쪽에 표시
        for i, (x, y) in enumerate(zip(top_pages["pagePath"], top_pages["userEngagementDuration"])):
            hours = y / 3600  # 초를 시간으로 변환
            ax.text(i, y + max(top_pages["userEngagementDuration"]) * 0.015,  # y값을 약간 위로 이동
                    f'{hours:.1f}h', color='white', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')

        ax.grid(axis="y", linestyle="--", alpha=0.7, color='white')
        plt.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15)
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas

    
 
    ### 상위 10개 페이지 조회수
    def create_top_pages_views_chart(self, service_account_file, property_id):
        df = self.run_ga4_report(
            service_account_file,
            property_id,
            dimensions=[{"name": "pagePath"}],
            metrics=[{"name": "screenPageViews"}]
        )

        if df is None or df.empty:
            print("해당 날짜 범위에 사용할 수 있는 데이터가 없습니다.")
            return None

        # 상위 10개 페이지 선택
        top_pages = df.sort_values(by="screenPageViews", ascending=False).head(10)

        # 시각화: 상위 페이지 조회수 막대 그래프
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')

        pastel_colors = plt.get_cmap('Pastel1', len(top_pages))
        bars = ax.bar(top_pages["pagePath"], top_pages["screenPageViews"],
                    color=[pastel_colors(i) for i in range(len(top_pages))], zorder=2)
        ax.set_xlabel("페이지 경로", fontsize=12, color='white')
        ax.set_ylabel("조회수", fontsize=12, color='white')
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=1)

        # 각 막대 위에 조회수 숫자 표시
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,  # 막대 바로 위에 숫자 위치
                    f'{int(bar.get_height())}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

        # 공통 스타일 적용
        self.apply_common_styling(fig, ax, "상위 10개 페이지 조회수")

        plt.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.15)  # 그래프와 타이틀을 아래로 조정하고 윗부분에 공간 추가
        canvas = FigureCanvas(fig)
        plt.close(fig)  # 그래프 닫기
        return canvas



    


##################################################################################################################################

class NoticeSuccessWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("공지사항 작성 완료")
        self.setFixedSize(528, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # 성공 메시지 표시
        label = QLabel("공지사항이 성공적으로 저장되었습니다.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; margin-top: -20px;")
        layout.addWidget(label)

        # 뒤로가기 버튼 추가
        back_button = QPushButton("뒤로가기")
        back_button.setFixedSize(200, 70)
        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        back_button.clicked.connect(self.handle_back)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        central_widget.setLayout(layout)

    def handle_back(self):
        self.close()
        
        
#########################################################################################################################

class NoticeErrorWindow(QMainWindow):
    def __init__(self, error_message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("공지사항 작성 오류")
        self.setFixedSize(528, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # 오류 메시지 표시
        label = QLabel(f"공지사항 \n 내용을 작성해주세요.")
        label.setAlignment(Qt.AlignCenter)
        
        # QLabel의 폰트 크기 설정
        font = label.font()
        font.setPointSize(17)  # 원하는 폰트 크기로 설정
        label.setFont(font)
        
        
        label.setStyleSheet("font-family: 'Malgun Gothic;  margin-top: -20px;")
        layout.addWidget(label)

        # 뒤로가기 버튼 추가
        back_button = QPushButton("뒤로가기")
        back_button.setFixedSize(200, 70)
        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        back_button.clicked.connect(self.handle_back)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        central_widget.setLayout(layout)

    def handle_back(self):
        self.close()


##################################################################################################################################

class ModelTrainingThread(QThread):
    finished = pyqtSignal()  # 학습 완료 신호
    process = None  # 프로세스 객체를 저장
    stopped = False  # 중단 여부 플래그

    def __init__(self, script_path, parent=None):
        super().__init__(parent)
        self.script_path = script_path

    def run(self):
        # 외부 Python 스크립트 실행
        self.process = subprocess.Popen(["python", self.script_path], shell=True)
        self.process.wait()  # 프로세스가 완료될 때까지 대기
        if not self.stopped:  # 중단되지 않았을 때만 완료 신호 방출
            self.finished.emit()

    def stop(self):
        self.stopped = True  # 중단 플래그 설정
        if self.process and self.process.poll() is None:  # 프로세스가 실행 중인지 확인
            # psutil을 사용하여 자식 프로세스도 모두 종료
            process = psutil.Process(self.process.pid)
            for child in process.children(recursive=True):  # 자식 프로세스 검색
                try:
                    child.kill()  # 자식 프로세스 강제 종료
                except psutil.NoSuchProcess:
                    pass  # 이미 종료된 프로세스는 무시
            self.process.kill()  # 부모 프로세스 강제 종료
            self.process = None


class TrainingProgressDialog(QDialog):
    def __init__(self, model_name, parent=None, training_thread=None):
        super().__init__(parent)
        self.setWindowTitle(f"{model_name} 모델 재학습 진행 중")
        self.setFixedSize(528, 500)
        self.setStyleSheet("background-color: #2b2b2b;")
        self.training_thread = training_thread

        layout = QVBoxLayout()

        # 로고 추가
        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # 안내 라벨
        label = QLabel(f"{model_name} 모델 재학습이 진행 중입니다.\n잠시만 기다려주세요...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 20px; color: white;")
        layout.addWidget(label)

        # 진행률 표시를 위한 ProgressBar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # 진행 상태를 애매하게 표시
        progress_bar.setTextVisible(False)
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #525252;
            }
            QProgressBar::chunk {
                background-color: orange;
                width: 20px;
            }
        """)
        layout.addWidget(progress_bar)
        
        # 프로그래스바와 닫기 버튼 사이의 간격 추가
        layout.addSpacing(40)  # 원하는 픽셀 값으로 조정

        # 닫기 버튼 추가
        close_button = QPushButton("닫기")
        close_button.setFixedSize(100, 50)
        close_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        close_button.clicked.connect(self.close_and_stop)

        # 버튼을 레이아웃에 추가
        button_layout = QHBoxLayout()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def close_and_stop(self):
        # 창 닫을 때 학습 중지
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
        self.close()




#################################################################################################################################

### 새로운 AI 모델 재학습 팝업창 페이지
class RetrainModelWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 모델 재학습")
        self.setFixedSize(528, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        label = QLabel("재학습 할 AI모델을 선택해주세요.")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; margin-top: -20px;")
        layout.addWidget(label)

        # 3개의 버튼 생성 및 간격 조정
        button_layout = QHBoxLayout()
        deepfake_button = QPushButton("딥페이크\n모델")
        voice_button = QPushButton("보이스피싱\n모델")
        sms_button = QPushButton("문자스미싱\n모델")

        for button in [deepfake_button, voice_button, sms_button]:
            button.setFixedSize(130, 80)
            button.setStyleSheet("""
                QPushButton {
                    font-family: 'Malgun Gothic';
                    border: none;
                    border-radius: 20px;
                    color: white;
                    background-color: #3c3c3c;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: orange;
                }
            """)
            button_layout.addWidget(button)
            button_layout.addSpacing(8)  # 버튼 간의 간격 추가

        layout.addLayout(button_layout)

        # 버튼 클릭 시 실행할 함수 연결
        deepfake_button.clicked.connect(lambda: self.run_model_retrain("딥페이크", "./deepfake_retrain_model/model_retrain.py"))
        voice_button.clicked.connect(lambda: self.run_model_retrain("보이스피싱", "./sms_retrain_model/model_retrain.py"))
        sms_button.clicked.connect(lambda: self.run_model_retrain("문자스미싱", "./sms_retrain_model/model_retrain.py"))

        # 뒤로가기 버튼 추가
        back_button = QPushButton("뒤로가기")
        back_button.setFixedSize(180, 55)
        back_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 20px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        back_button.clicked.connect(self.close)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        layout.addSpacing(8)  # 밑부분 공간 간격

        central_widget.setLayout(layout)

    def run_model_retrain(self, model_name, script_path):
        # 진행률 팝업 창 생성 및 표시
        self.progress_dialog = TrainingProgressDialog(model_name, self)
        self.progress_dialog.show()

        # 학습 스레드 생성 및 시작
        self.training_thread = ModelTrainingThread(script_path)
        self.progress_dialog.training_thread = self.training_thread
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()

    def on_training_finished(self):
        # 학습이 끝나면 진행률 팝업 창 닫기
        if self.progress_dialog and not self.training_thread.stopped:
            self.progress_dialog.close()
            self.progress_dialog = None

            # 완료 메시지 또는 다른 후속 작업 추가 가능
            self.show_training_completed_message()

    ## 여기다
    def show_training_completed_message(self):
        # 학습 완료 알림 메시지
        completion_dialog = QDialog(self)
        completion_dialog.setWindowTitle("재학습 완료")
        completion_dialog.setFixedSize(528, 500)
        completion_dialog.setStyleSheet("background-color: #2b2b2b;")

        layout = QVBoxLayout(completion_dialog)

        # 로고 추가
        logo_label = QLabel()
        pixmap = QPixmap("./image/logo.png")
        pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # 안내 라벨
        message_label = QLabel("모델 재학습이 완료되었습니다.")
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 25px; color: white;")
        layout.addWidget(message_label)

        
        # 닫기 버튼 추가
        close_button = QPushButton("닫기")
        close_button.setFixedSize(200, 70)
        close_button.setStyleSheet("""
            QPushButton {
                font-family: 'Malgun Gothic';
                border: none;
                border-radius: 15px;
                color: white;
                background-color: #3c3c3c;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: orange;
            }
        """)
        
        
        close_button.clicked.connect(completion_dialog.close)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        completion_dialog.setLayout(layout)
        completion_dialog.exec_()
    
    

#########################################################################################################################################           
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
    QMainWindow {
        background-color: #2b2b2b;
        color: white;
        font-family: 'Malgun Gothic';
    }
    QPushButton, QCheckBox, QRadioButton, QLineEdit, QComboBox, QSpinBox {
        color: white;
        background-color: #3c3c3c;
        border: none;
        padding: 10px;
        font-family: 'Malgun Gothic';
    }
    QProgressBar {
        color: white;
        background-color: #3c3c3c;
        font-family: 'Malgun Gothic';
    }
    QLabel {
        color: white;
        font-family: 'Malgun Gothic';
    }
    QTabWidget::pane {
        border: 1px solid lightgray;
    }
    QTabBar::tab {
        font-family: 'Malgun Gothic';
        background-color: #3c3c3c;
        color: white;
        padding: 10px;
    }
    QTabBar::tab:selected {
        background-color: #3c3c3c;;
    }
    QTabBar::tab:hover {
        background-color: orange;
    }
""")

    ## 처음에 로그인 윈도우 부터 하려면
    login_window = LoginWindow()
    login_window.show()
    
    # main_window = MainWindow()
    # main_window.show()

    sys.exit(app.exec_())
