import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import tempfile
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

# Import libraries for text-to-speech
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa

# Constants for UI and gestures
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)
UI_BG_COLOR = (50, 50, 50, 150)  # Semi-transparent background
EXIT_BTN_COLOR_NORMAL = (0, 0, 255)
EXIT_BTN_COLOR_HOVER = (0, 0, 150)
EXIT_TEXT_COLOR = (255, 255, 255)
SMOOTHING_BUFFER_SIZE = 5
WINDOW_NAME = "Hand Gesture Recognizer"
AUDIO_COOLDOWN = 2  # Cooldown time in seconds for audio feedback

# Hand Detection Classes
# This class handles all core hand tracking using Mediapipe.
class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, max_num_hands=self.maxHands,
            model_complexity=modelComplexity, min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # Landmarks for fingertips

    # Finds hands and draws landmarks on the image.
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img, self.results

    # Gets landmark positions as a list of coordinates.
    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    # Returns if the hand is Left or Right.
    def get_handedness(self, handNo=0):
        if self.results.multi_handedness:
            return self.results.multi_handedness[handNo].classification[0].label
        return None

    # Logic to determine which fingers are up based on landmark positions.
    def fingersUp(self, lmList, handedness):
        fingers = []
        if not lmList:
            return fingers

        # Thumb logic (special case for left/right hands)
        if handedness == "Right":
            fingers.append(1 if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1] else 0)
        elif handedness == "Left":
            fingers.append(1 if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1] else 0)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            fingers.append(1 if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2] else 0)

        return fingers

# Gesture Management Classes
# Represents a single gesture.
class Gesture:
    def __init__(self, name, finger_pattern, direction="Neutral"):
        self.name = name
        self.finger_pattern = finger_pattern
        self.direction = direction

# Manages and recognizes all gestures. Easy to add new gestures here.
class GestureRecognizer:
    def __init__(self):
        self.gestures = {}
        self.add_predefined_gestures()

    def add_gesture(self, gesture: Gesture):
        self.gestures[tuple(gesture.finger_pattern)] = gesture

    def add_predefined_gestures(self):
        self.add_gesture(Gesture("Fist", [0, 0, 0, 0, 0]))
        self.add_gesture(Gesture("Open Hand", [1, 1, 1, 1, 1]))
        self.add_gesture(Gesture("Peace", [0, 1, 1, 0, 0]))
        self.add_gesture(Gesture("Thumbs Up", [1, 0, 0, 0, 0]))
        self.add_gesture(Gesture("Pointing", [0, 1, 0, 0, 0]))
        self.add_gesture(Gesture("Party", [1, 0, 0, 1, 1]))
        self.add_gesture(Gesture("Party", [1, 1, 0, 0, 1]))
        self.add_gesture(Gesture("Pinky", [1, 0, 0, 0, 1]))
        self.add_gesture(Gesture("Pinky", [0, 0, 0, 0, 1]))

    def recognize(self, finger_pattern):
        return self.gestures.get(tuple(finger_pattern), None)

# Holds data for a detected hand
class HandInfo:
    def __init__(self, handedness, lmList):
        self.handedness = handedness
        self.lmList = lmList
        self.fingers_up = []
        self.num_fingers = 0
        self.gesture_name = "Unknown"
        self.direction = "Neutral"

# Voice Functionality
is_playing_audio = False # Flag to prevent overlapping audio

def say_gesture(text):
    global is_playing_audio
    if is_playing_audio:
        return

    def run_voice():
        global is_playing_audio
        is_playing_audio = True
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                mp3_path = os.path.join(temp_dir, "gesture.mp3")
                wav_path = os.path.join(temp_dir, "gesture.wav")

                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(mp3_path)
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(wav_path, format="wav")

                wave_obj = sa.WaveObject.from_wave_file(wav_path)
                play_obj = wave_obj.play()
                play_obj.wait_done()
        except Exception as e:
            print(f"Error playing sound: {e}")
        finally:
            is_playing_audio = False

    threading.Thread(target=run_voice).start()


# Multithreading for video processing
# This thread handles all the heavy lifting: video capture and gesture detection.
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage, list) # Signal to send processed frames and data
    gesture_detected_signal = pyqtSignal(str) # Signal to trigger voice feedback

    def __init__(self):
        super().__init__()
        self.running = True
        self.detector = HandDetector(maxHands=2)
        self.gesture_recognizer = GestureRecognizer()
        self.cap = cv2.VideoCapture(0)
        self.last_gesture_name = ""
        self.gesture_buffer = deque(maxlen=3) # Buffer to stabilize gesture detection
        self.last_gesture_time = 0
        self.audio_cooldown = AUDIO_COOLDOWN

    def run(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1)
                img, results = self.detector.findHands(img, draw=True)
                hands_data = []
                current_gesture_name = "No Hand Detected"

                if results.multi_hand_landmarks:
                    for i, handLms in enumerate(results.multi_hand_landmarks):
                        lmList = self.detector.findPosition(img, handNo=i)
                        handedness = self.detector.get_handedness(handNo=i)
                        
                        if lmList and handedness:
                            fingers = self.detector.fingersUp(lmList, handedness)
                            hand_info = HandInfo(handedness, lmList)
                            hand_info.fingers_up = fingers
                            hand_info.num_fingers = sum(fingers)
                            
                            # Calculate hand direction
                            if len(lmList) > 8:
                                wrist_x = lmList[0][1]
                                index_tip_x = lmList[8][1]
                                if abs(wrist_x - index_tip_x) > 50:
                                    hand_info.direction = "Right" if index_tip_x > wrist_x else "Left"
                                else:
                                    hand_info.direction = "Neutral"

                            # Use the recognizer to find the gesture
                            recognized_gesture = self.gesture_recognizer.recognize(fingers)
                            hand_info.gesture_name = recognized_gesture.name if recognized_gesture else "Unknown"
                            current_gesture_name = hand_info.gesture_name
                            hands_data.append(hand_info)

                # Stabilize gesture detection with a buffer
                self.gesture_buffer.append(current_gesture_name)
                if self.gesture_buffer.count(current_gesture_name) >= len(self.gesture_buffer):
                    if current_gesture_name != self.last_gesture_name and current_gesture_name not in ["Unknown", "No Hand Detected"]:
                        current_time = time.time()
                        if (current_time - self.last_gesture_time) > self.audio_cooldown:
                            self.gesture_detected_signal.emit(current_gesture_name)
                            self.last_gesture_name = current_gesture_name
                            self.last_gesture_time = current_time
                
                # Convert image for display in PyQt
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image, hands_data)
                
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

# PyQt5 GUI Application
# The main application window and UI
class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_NAME)
        
        self.setFixedSize(1400, 800)
        self.setStyleSheet("background-color: #323232;")
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #505050;")
        self.video_label.setFixedSize(960, 720)
        
        self.exit_button = QPushButton("EXIT")
        self.exit_button.setStyleSheet(f"background-color: rgb{EXIT_BTN_COLOR_NORMAL}; color: white; font-size: 16px; border-radius: 5px; padding: 10px;")
        self.exit_button.setFixedSize(150, 50)
        self.exit_button.clicked.connect(self.close)

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.exit_button, alignment=Qt.AlignCenter)
        self.setLayout(main_layout)

        # Start the video processing thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_ui)
        self.video_thread.gesture_detected_signal.connect(self.handle_gesture_voice)
        self.video_thread.start()
        
        # FPS tracking
        self.start_time = time.time()
        self.frame_count = 0

    # Updates the UI with a new frame and hand data.
    def update_ui(self, qt_image, hands_data):
        img = qt_image.copy()
        q_format = QImage.Format_RGB888
        h, w = img.height(), img.width()
        
        ptr = img.constBits()
        ptr.setsize(h * w * 3)
        np_img = np.array(ptr).reshape((h, w, 3))
        
        self.draw_overlays(np_img, hands_data)
        
        final_img = QImage(np_img.data, w, h, q_format)
        pixmap = QPixmap.fromImage(final_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.frame_count += 1
        
    # Draws UI elements directly onto the video frame.
    def draw_overlays(self, img, hands_data):
        h, w, _ = img.shape
        
        # 1. Main Title and Description
        main_title = "Gesture Detection Model"
        description = "Show your palms to the camera"
        
        title_font = cv2.FONT_HERSHEY_DUPLEX
        desc_font = cv2.FONT_HERSHEY_PLAIN
        title_scale = 1.5
        desc_scale = 1.2
        title_thickness = 2
        desc_thickness = 1

        title_size = cv2.getTextSize(main_title, title_font, title_scale, title_thickness)[0]
        desc_size = cv2.getTextSize(description, desc_font, desc_scale, desc_thickness)[0]
        
        overlay_padding = 20
        overlay_width = max(title_size[0], desc_size[0]) + 2 * overlay_padding
        overlay_height = title_size[1] + desc_size[1] + 2 * overlay_padding + 10
        
        # Fix for UnboundLocalError
        overlay_x1 = (w - overlay_width) // 2
        overlay_y1 = 20
        overlay_x2 = overlay_x1 + overlay_width
        overlay_y2 = overlay_y1 + overlay_height
        
        sub_img = img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        overlay = np.zeros_like(sub_img)
        overlay[:, :, :] = (0, 0, 0) # Black background
        cv2.addWeighted(sub_img, 1 - 0.7, overlay, 0.7, 0, sub_img)

        title_x = overlay_x1 + (overlay_width - title_size[0]) // 2
        title_y = overlay_y1 + overlay_padding + title_size[1]
        desc_x = overlay_x1 + (overlay_width - desc_size[0]) // 2
        desc_y = title_y + overlay_padding + desc_size[1] + 5

        cv2.putText(img, main_title, (title_x, title_y), title_font, title_scale, TEXT_COLOR, title_thickness, cv2.LINE_AA)
        cv2.putText(img, description, (desc_x, desc_y), desc_font, desc_scale, TEXT_COLOR, desc_thickness, cv2.LINE_AA)

        # 2. Hand-specific info (only if hand is detected)
        for hand_info in hands_data:
            label_y_start = h - 200
            x_pos = 10 if hand_info.handedness == "Left" else w - 260
            self.draw_hand_label(img, x_pos, label_y_start, hand_info)
        
        # 3. FPS Counter
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.start_time, self.frame_count = end_time, 0
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(img, fps_text, (w - 100, 40), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

    # Helper function to draw the hand information labels
    def draw_hand_label(self, img, x_pos, y_pos, hand_info):
        label_width, label_height = 250, 150
        sub_img = img[y_pos:y_pos + label_height, x_pos:x_pos + label_width]
        overlay = np.zeros(sub_img.shape, dtype=np.uint8)
        overlay[:, :, :] = (50, 50, 50)
        cv2.addWeighted(sub_img, 0.7, overlay, 0.3, 0, sub_img)

        cv2.putText(img, f"Hand: {hand_info.handedness}", (x_pos + 10, y_pos + 30), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, f"Gesture: {hand_info.gesture_name}", (x_pos + 10, y_pos + 60), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, f"Fingers: {hand_info.num_fingers}", (x_pos + 10, y_pos + 90), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, f"Direction: {hand_info.direction}", (x_pos + 10, y_pos + 120), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

    def handle_gesture_voice(self, gesture_name):
        say_gesture(gesture_name)

    def closeEvent(self, event):
        self.video_thread.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureApp()
    window.show()
    sys.exit(app.exec_())