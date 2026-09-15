#!/usr/bin/env python3
"""
Combined Camera (nhan dien vat the bang chip AI imx500 + nhan dien khuon mat bang OpenCV
+ nhan dien trai pickleball bang YOLO) + RPLIDAR web viewer cho Raspberry Pi.
Chay: python3 combined_stream.py
Xem:  http://<IP_CUA_PI>:8000/
"""

import json
import logging
import os
import socketserver
import time
from http import server
from threading import Condition, Lock, Thread

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from rplidar import RPLidar
from ultralytics import YOLO

# ---------- Cau hinh ----------
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUDRATE = 115200
CAM_SIZE = (640, 480)
MAX_DIST_MM = 6000
HTTP_PORT = 8000
JPEG_QUALITY = 80

# Model nhan dien vat the chinh thuc cua Raspberry Pi, cai qua "sudo apt install imx500-all"
MODEL_PATH = '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
OBJ_THRESHOLD = 0.55  # do tin cay toi thieu de ve khung vat the (imx500)
PERSON_LABEL_NAME = 'person'  # ten nhan trong tap COCO ma model imx500 tra ve

# Model YOLO rieng nhan dien trai pickleball (dat file .pt cung thu muc voi script nay)
PICKLEBALL_MODEL_PATH = 'test_11_9.pt'
PICKLEBALL_THRESHOLD = 0.5  # do tin cay toi thieu de ve khung pickleball

# Model nhan dien khuon mat (OpenCV YuNet, dnn module) - dat file .onnx cung thu muc voi script nay
# QUAN TRONG: ban OpenCV 5.x tren Raspberry Pi OS (apt) can dung ban "2026may" (dynamic input shape).
# Ban "2023mar" (fixed 640x640) CHI tuong thich OpenCV 4.x va se bi loi tren may nay.
# Tai file bang lenh sau (chay tren Pi, can mang):
#   wget -O face_detection_yunet_2026may.onnx \
#     "https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/resolve/main/face_detection_yunet_2026may.onnx"
FACE_MODEL_PATH = 'face_detection_yunet_2026may.onnx'
FACE_THRESHOLD = 0.6
FACE_INPUT_STRIDE = 32  # model 2026may yeu cau chieu rong/cao dau vao la boi so cua 32

logging.basicConfig(level=logging.INFO)

# ---------- HTML trang chinh ----------
PAGE = """\
<html>
<head>
<title>Pi Camera (AI Object + Face + Pickleball) + LIDAR</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif; text-align:center; }
  .wrap { display:flex; justify-content:center; gap:30px; flex-wrap:wrap; margin-top:20px; }
  canvas, img { background:#000; border:1px solid #444; }
  .legend { font-size:13px; margin-top:8px; color:#aaa; }
  .legend span { padding:2px 8px; border-radius:4px; margin:0 4px; }
</style>
</head>
<body>
<h2>Raspberry Pi - Camera (Vat the qua AI Camera + Khuon mat + Pickleball qua YOLO) + LIDAR</h2>
<div class="wrap">
  <div>
    <h3>Camera</h3>
    <img src="stream.mjpg" width="640" height="480" />
    <div class="legend">
      <span style="background:#00f;color:#fff;">Vat the (imx500)</span>
      <span style="background:#f00;color:#fff;">Nguoi (imx500)</span>
      <span style="background:#0f0;color:#000;">Khuon mat</span>
      <span style="background:#ff0;color:#000;">Pickleball (YOLO)</span>
    </div>
  </div>
  <div>
    <h3>LIDAR (radar view)</h3>
    <canvas id="radar" width="500" height="500"></canvas>
  </div>
</div>
<script>
const canvas = document.getElementById('radar');
const ctx = canvas.getContext('2d');
const CX = 250, CY = 250, MAXR = 240;
const MAXDIST = %(maxdist)s;

function drawGrid() {
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  for (let r = 1; r <= 4; r++) {
    ctx.moveTo(CX + r * (MAXR/4), CY);
    ctx.arc(CX, CY, r * (MAXR/4), 0, 2 * Math.PI);
  }
  ctx.moveTo(CX - MAXR, CY); ctx.lineTo(CX + MAXR, CY);
  ctx.moveTo(CX, CY - MAXR); ctx.lineTo(CX, CY + MAXR);
  ctx.stroke();
}

async function update() {
  try {
    const res = await fetch('/lidar_data');
    const data = await res.json();
    ctx.clearRect(0, 0, 500, 500);
    drawGrid();
    ctx.fillStyle = '#0f0';
    data.forEach(function (p) {
      const angleDeg = p[1];
      const dist = Math.min(p[2], MAXDIST);
      const angleRad = angleDeg * Math.PI / 180;
      const scale = MAXR / MAXDIST;
      const x = CX + dist * scale * Math.sin(angleRad);
      const y = CY - dist * scale * Math.cos(angleRad);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  } catch (e) {
    console.error(e);
  }
  setTimeout(update, 200);
}
update();
</script>
</body>
</html>
""" % {"maxdist": MAX_DIST_MM}

# ---------- Bo nho dung frame moi nhat de stream ----------
class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()


output = StreamingOutput()


def find_haarcascade():
    """Tim file haarcascade_frontalface_default.xml o nhieu vi tri co the co.
    Luu y: mot so ban OpenCV toi gian (vd apt python3-opencv tren Raspberry Pi OS)
    KHONG co module objdetect -> cv2.CascadeClassifier se khong ton tai du tim
    ra file. Ham nay chi tim FILE, khong dam bao cv2.CascadeClassifier dung duoc."""
    candidates = []
    try:
        candidates.append(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except AttributeError:
        pass
    candidates += [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_face_detector():
    """
    Uu tien YuNet (module dnn - hoat dong tot tren ban OpenCV rut gon cua Raspberry Pi OS).
    Neu khong tai duoc (thieu file / sai phien ban) thi thu Haar Cascade (module objdetect
    -> co the KHONG co tren may nay, se tu dong bo qua chu khong lam sap chuong trinh).
    Tra ve tuple (face_detector_yunet_or_None, haar_cascade_or_None).
    """
    # --- Thu YuNet truoc (khuyen nghi tren Raspberry Pi OS) ---
    try:
        detector = cv2.FaceDetectorYN_create(
            FACE_MODEL_PATH, "", (320, 320), score_threshold=FACE_THRESHOLD,
        )
        logging.info('Da tai xong model khuon mat YuNet: %s', FACE_MODEL_PATH)
        return detector, None
    except Exception as e:
        logging.warning('Khong tai duoc model YuNet (%s). Thu Haar Cascade...', e)

    # --- Fallback: Haar Cascade (co the KHONG co tren Raspberry Pi OS) ---
    try:
        cascade_path = find_haarcascade()
        if not cascade_path:
            logging.warning('Khong tim thay file Haar Cascade. TAT tinh nang nhan dien khuon mat.')
            return None, None
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logging.warning('File Haar Cascade rong/khong hop le. TAT tinh nang nhan dien khuon mat.')
            return None, None
        logging.info('Dung Haar Cascade cho nhan dien khuon mat (do chinh xac thap hon YuNet).')
        return None, cascade
    except AttributeError as e:
        logging.warning(
            'OpenCV tren may nay khong co module objdetect (%s). TAT tinh nang nhan dien khuon mat. '
            'Object detection (imx500) va pickleball detection (YOLO) van chay binh thuong.',
            e,
        )
        return None, None
    except Exception as e:
        logging.warning('Loi khong xac dinh khi tai Haar Cascade (%s). TAT tinh nang nhan dien khuon mat.', e)
        return None, None


def detect_faces_yunet(frame, face_detector):
    """Chay YuNet tren toan bo frame. Model '2026may' can chieu rong/cao la
    boi so cua 32 -> pad them vien phai/duoi (khong lech goc toa do)."""
    fh, fw = frame.shape[:2]
    pad_w = (-fw) % FACE_INPUT_STRIDE
    pad_h = (-fh) % FACE_INPUT_STRIDE
    if pad_w or pad_h:
        padded = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded = frame

    face_detector.setInputSize((padded.shape[1], padded.shape[0]))
    _, results = face_detector.detect(padded)
    faces = []
    if results is not None:
        for f in results:
            x, y, w, h = map(int, f[:4])
            conf = float(f[-1])
            faces.append((x, y, w, h, conf))
    return faces


# ---------- Camera: object detection (imx500) + face detection (OpenCV CPU) + pickleball (YOLO CPU) ----------
def camera_worker():
    # IMX500 phai duoc khoi tao TRUOC khi tao Picamera2
    imx500 = IMX500(MODEL_PATH)
    intrinsics = getattr(imx500, 'network_intrinsics', None)
    labels = getattr(intrinsics, 'labels', None) if intrinsics is not None else None

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"}, buffer_count=12
    )
    try:
        imx500.show_network_fw_progress_bar()
    except Exception:
        pass
    picam2.start(config)

    logging.info('IMX500 labels (%d): %s', len(labels) if labels else 0, labels)
    if not labels:
        logging.warning(
            'CANH BAO: khong doc duoc labels tu model imx500! '
            'Nhan dien "nguoi" se KHONG hoat dong vi ten nhan luon la "classN".'
        )

    # --- Face detection (uu tien YuNet, fallback Haar Cascade neu co) ---
    face_detector, face_cascade = load_face_detector()

    # --- Pickleball detection (YOLO, CPU) ---
    logging.info('Dang tai model pickleball tu %s ...', PICKLEBALL_MODEL_PATH)
    pickleball_model = YOLO(PICKLEBALL_MODEL_PATH)
    logging.info('Da tai xong model pickleball')

    logging.info(
        'Camera san sang - AI object detection (imx500): ON, '
        'face detection: %s, pickleball detection (YOLO): ON',
        'YuNet' if face_detector is not None else ('Haar Cascade' if face_cascade is not None else 'OFF')
    )

    last_debug_time = 0.0
    DEBUG_LOG_INTERVAL = 2.0  # giay - log tinh trang object detection dinh ky de theo doi lien tuc

    # --- Bien theo doi FPS ---
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    while True:
        request = None
        try:
            request = picam2.capture_request()
            metadata = request.get_metadata()
            frame = request.make_array("main")  # BGR order

            # --- Nhan dien vat the ngay tren chip AI cua camera (khong ton CPU Pi) ---
            try:
                outputs = imx500.get_outputs(metadata)
            except Exception as e:
                outputs = None
                logging.warning('imx500.get_outputs() bi loi ngay tu dau: %s', e)

            now = time.time()
            should_log_debug = (now - last_debug_time) > DEBUG_LOG_INTERVAL

            if outputs is None:
                if should_log_debug:
                    last_debug_time = now
                    logging.info(
                        'DEBUG dinh ky: get_outputs()=None (chip AI chua san sang hoac '
                        'khong tra ket qua cho frame nay)'
                    )
            else:
                try:
                    boxes, scores, classes = outputs[0], outputs[1], outputs[2]
                    scores_arr = np.asarray(scores)

                    if should_log_debug:
                        last_debug_time = now
                        n_total = len(scores_arr)
                        n_above = int((scores_arr >= OBJ_THRESHOLD).sum()) if n_total else 0
                        top_score = float(scores_arr.max()) if n_total else 0.0
                        logging.info(
                            'DEBUG dinh ky: %d detection tho, diem cao nhat=%.3f, '
                            '%d/%d qua nguong %.2f (shapes boxes:%s scores:%s classes:%s)',
                            n_total, top_score, n_above, n_total, OBJ_THRESHOLD,
                            np.asarray(boxes).shape, scores_arr.shape, np.asarray(classes).shape,
                        )

                    for box, score, cls in zip(boxes, scores, classes):
                        if score < OBJ_THRESHOLD:
                            continue
                        obj = imx500.convert_inference_coords(box, metadata, picam2)
                        # Tuy phien ban picamera2, convert_inference_coords co the tra ve
                        # object co thuoc tinh .x/.y/.width/.height HOAC mot tuple/list (x, y, w, h).
                        if hasattr(obj, 'x'):
                            x, y, w, h = obj.x, obj.y, obj.width, obj.height
                        else:
                            x, y, w, h = obj
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cls_idx = int(cls)
                        name = labels[cls_idx] if labels and cls_idx < len(labels) else f"class{cls_idx}"

                        if name.strip().lower() == PERSON_LABEL_NAME:
                            # Khung rieng cho nguoi: mau do, day hon de de phan biet voi vat the khac
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(
                                frame, f"Nguoi {score:.2f}", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                            )
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(
                                frame, f"{name} {score:.2f}", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                            )
                except Exception as e:
                    if should_log_debug:
                        last_debug_time = now
                        logging.warning('Loi parse object detection: %s', e)

            # --- Nhan dien khuon mat (YuNet uu tien, Haar Cascade la fallback) ---
            if face_detector is not None:
                try:
                    faces = detect_faces_yunet(frame, face_detector)
                    for (x, y, w, h, conf) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame, f"Face {conf:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                except Exception as e:
                    logging.warning('Loi nhan dien khuon mat (YuNet): %s', e)
            elif face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                    )
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame, 'Face', (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                except Exception as e:
                    logging.warning('Loi nhan dien khuon mat (Haar Cascade): %s', e)

            # --- Nhan dien trai pickleball bang model YOLO rieng (chay tren CPU) ---
            try:
                results = pickleball_model(frame, verbose=False)[0]
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < PICKLEBALL_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_idx = int(box.cls[0])
                    if isinstance(pickleball_model.names, dict):
                        name = pickleball_model.names.get(cls_idx, f"class{cls_idx}")
                    else:
                        name = pickleball_model.names[cls_idx]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
            except Exception as e:
                logging.warning('Loi nhan dien pickleball: %s', e)

            # --- Tinh va ve FPS ---
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            ok, jpeg = cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                with output.condition:
                    output.frame = jpeg.tobytes()
                    output.condition.notify_all()
        except Exception as e:
            logging.warning('Loi xu ly camera: %s', e)
            time.sleep(1)
        finally:
            if request is not None:
                request.release()


# ---------- LIDAR background thread ----------
lidar_points = []
lidar_lock = Lock()


def lidar_worker():
    global lidar_points
    while True:
        lidar = None
        try:
            lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
            time.sleep(0.5)
            lidar.clean_input()
            for scan in lidar.iter_scans():
                pts = [[q, angle, dist] for (q, angle, dist) in scan]
                with lidar_lock:
                    lidar_points = pts
        except Exception as e:
            logging.warning('Loi LIDAR: %s - thu lai sau 5s', e)
            time.sleep(5)
        finally:
            if lidar is not None:
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except Exception:
                    pass


# ---------- HTTP handler ----------
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame is None:
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Client camera ngat: %s', e)

        elif self.path == '/lidar_data':
            with lidar_lock:
                data = json.dumps(lidar_points).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    t_cam = Thread(target=camera_worker, daemon=True)
    t_cam.start()

    t_lidar = Thread(target=lidar_worker, daemon=True)
    t_lidar.start()

    try:
        srv = StreamingServer(('', HTTP_PORT), StreamingHandler)
        print(f"Server dang chay -> http://<IP_CUA_PI>:{HTTP_PORT}")
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Dang dung server...")
