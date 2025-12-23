from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from math import sqrt
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'csi_activity_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================
# LOAD MODEL + ENCODER + SCALERS
# ============================

model = tf.keras.models.load_model("csi_activity_model_final.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("per_channel_scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

WINDOW_LEN = 200

# ============================
# PARSE CSI STRING INTO AMPLITUDES
# ============================

def parse_csi_row(csi_string):
    try:
        inner = re.findall(r"\[(.*)\]", csi_string)[0]
        nums = [int(x) for x in inner.split() if x.strip() != ""]
        imag = nums[0::2]
        real = nums[1::2]
        return [sqrt(i*i + r*r) for i,r in zip(imag, real)]
    except:
        return None

# ============================
# SCALE USING SAVED SCALERS
# ============================

def scale_segment(seg, scalers):
    seg_scaled = np.empty_like(seg, dtype=np.float32)
    for ch in range(seg.shape[1]):
        flat = seg[:, ch].reshape(-1, 1)
        seg_scaled[:, ch] = scalers[ch].transform(flat).reshape(-1)
    return seg_scaled

# ============================
# CSI PROCESSING THREAD
# ============================

def csi_processing_thread():
    CSV_PATH = r"D:\sem 8\reseach\activity----raw\sitting\sitting_01.csv"
    buffer = []
    
    print("✅ CSI Processing Thread Started. Monitoring CSI file...")
    
    while True:
        try:
            df = pd.read_csv(CSV_PATH)
            
            buffer = []
            
            for _, row in df.iterrows():
                amps = parse_csi_row(row["CSI_DATA"])
                if amps is not None:
                    buffer.append(amps)
            
            if len(buffer) >= WINDOW_LEN:
                seg = np.array(buffer[-WINDOW_LEN:])    # last 200 rows
                
                seg_scaled = scale_segment(seg, scalers)
                seg_input = seg_scaled.reshape(1, WINDOW_LEN, seg.shape[1], 1)
                
                probs = model.predict(seg_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_name = label_encoder.inverse_transform([pred_idx])[0]
                
                # Emit the activity update to all connected clients
                socketio.emit('activity_update', {
                    'activity': pred_name,
                    'confidence': float(probs[pred_idx])
                })
                
                print(f"Activity: {pred_name}   (conf: {probs[pred_idx]:.2f})")
            
            time.sleep(0.5)
            
        except Exception as e:
            print("Error in CSI processing:", e)
            time.sleep(1)

@app.route('/')
def index():
    return render_template('csi_dashboard.html')

if __name__ == '__main__':
    # Start CSI processing in a separate thread
    thread = threading.Thread(target=csi_processing_thread)
    thread.daemon = True
    thread.start()
    
    print("🚀 Starting CSI Activity Detection Server...")
    print("📊 Web Dashboard available at: http://localhost:5001")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)