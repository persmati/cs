import torch
from flask import Flask, render_template, Response, jsonify
from PIL import Image
import torchvision.transforms as transforms
from MyCNN import ImprovedCNN
from itertools import zip_longest
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import threading
import time
import numpy as np
from collections import deque

# Flask app init
app = Flask(__name__)

#setup cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_path = r"match_1.webm"

#initialize thread-safe buffer for video frames
frame_buffer = []
buffer_lock = threading.Lock()
process_thread = None
#set up circular buffer with max size
MIN_BUFFER_SIZE = 30
frame_buffer = deque(maxlen=MIN_BUFFER_SIZE)

def load_model(path, model):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

class_labels_weapon = ['ak', 'awp', 'dgl', 'dual', 'five_seven', 'galil', 'glock', 'm4-s', 'm4a4', 'mac', 'mp9', 'p2000', 'p250', 'scout', 'tec9', 'usp']
class_labels_hp = ['0', '1', '10', '100', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
class_labels_timer = ['0', '0_01', '0_02', '0_03', '0_04', '0_05', '0_06', '0_07', '0_08', '0_09', '0_10', '0_11', '0_12', '0_13', '0_14', '0_15', '0_16', '0_17', '0_18', '0_19', '0_20', '0_21', '0_22', '0_23', '0_24', '0_25', '0_26', '0_27', '0_28', '0_29', '0_30', '0_31', '0_32', '0_33', '0_34', '0_35', '0_36', '0_37', '0_38', '0_39', '0_40', '0_41', '0_42', '0_43', '0_44', '0_45', '0_46', '0_47', '0_48', '0_49', '0_50', '0_51', '0_52', '0_53', '0_54', '0_55', '0_56', '0_57', '0_58', '0_59', '1_00', '1_01', '1_02', '1_03', '1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12', '1_13', '1_14', '1_15', '1_16', '1_17', '1_18', '1_19', '1_20', '1_21', '1_22', '1_23', '1_24', '1_25', '1_26', '1_27', '1_28', '1_29', '1_30', '1_31', '1_32', '1_33', '1_34', '1_35', '1_36', '1_37', '1_38', '1_39', '1_40', '1_41', '1_42', '1_43', '1_44', '1_45', '1_46', '1_47', '1_48', '1_49', '1_50', '1_51', '1_52', '1_53', '1_54', '1_55', 'bomb']
class_label_hud = ['0', '1'] # hud off / on

cnn_model = ImprovedCNN(input_shape=1, hidden_units=64, output_shape=len(class_labels_weapon), dropout_rate=0.2, H=20, W=70 ).to(device)
hp_model = ImprovedCNN(input_shape=1, hidden_units=64, output_shape=len(class_labels_hp), dropout_rate=0.2, H=20, W=70 ).to(device)
timer_model = ImprovedCNN(input_shape=1, hidden_units=64, output_shape=len(class_labels_timer), dropout_rate=0.2, H=20, W=70 ).to(device)
hud_model = ImprovedCNN(input_shape=1, hidden_units=64, output_shape=len(class_label_hud), dropout_rate=0.2, H=100, W=75 ).to(device)

model_weapon = load_model('weapons_d.pth', cnn_model)
model_hp = load_model('hp_d.pth', hp_model)
model_timer = load_model('timer_d.pth', timer_model)
model_hud = load_model('replay_d.pth', hud_model)

#coordinates for different elements detection
boxes = [
    (217, 595, 115, 30), (217, 685, 115, 30), (217, 775, 115, 30),
    (217, 865, 115, 30), (217, 955, 115, 30),

    (1590, 595, 115, 30),(1590, 685, 115, 30), (1589, 775, 115, 30), (1590, 865, 115, 30),
    (1591, 955, 115, 30)
]
boxes_hp = [
    (51, 599, 33, 21), (50, 690, 33, 19), (49, 779, 33, 21),
    (51, 870, 30, 20), (50, 960, 35, 21), (1835, 598, 33, 22),
    (1837, 689, 33, 22), (1837, 778, 32, 22), (1837, 868, 33, 22),
    (1836, 957, 34, 23)
]
box_timer = (938, 66, 44, 28)

box_score = [(825, 50, 28, 37), (1070, 50, 28, 37)]

box_hud = [ (36, 593, 295, 453), (1587, 595, 292, 448)]

data = {
    'team1_odds': 50,
    'team2_odds': 50,
    'team_assignment': 'Unknown',
    'timer': '0',
    'player_statuses': ['Unknown'] * 10,
    'weapons': ['Unknown'] * 10,
    'hp': ['hp'] * 10,
    'frame_number': 0,
    'team_1_score': '0',
    'team_2_score': '0',
    'frame' : 0
}

last_detected_scores = ['0','0']

#detect if team is T / CT base on hud color
def detect_team_colors(frame):
    roi_0 = frame[44:44+52, 657:657+147]  # Team 1 ROI
    roi_1 = frame[44:44+52, 1115:1115+145]  # Team 2 ROI

    # Define color ranges for T (yellow) and CT (blue)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Convert ROIs to HSV
    hsv_roi_0 = cv2.cvtColor(roi_0, cv2.COLOR_BGR2HSV)
    hsv_roi_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2HSV)

    # Create masks for each color in each ROI
    mask_yellow_0 = cv2.inRange(hsv_roi_0, lower_yellow, upper_yellow)
    mask_blue_0 = cv2.inRange(hsv_roi_0, lower_blue, upper_blue)
    mask_yellow_1 = cv2.inRange(hsv_roi_1, lower_yellow, upper_yellow)
    mask_blue_1 = cv2.inRange(hsv_roi_1, lower_blue, upper_blue)

    # Count non-zero pixels for each mask
    yellow_count_0 = cv2.countNonZero(mask_yellow_0)
    blue_count_0 = cv2.countNonZero(mask_blue_0)
    yellow_count_1 = cv2.countNonZero(mask_yellow_1)
    blue_count_1 = cv2.countNonZero(mask_blue_1)

    # Determine which team is T and which is CT
    if yellow_count_0 > blue_count_0 and blue_count_1 > yellow_count_1:
        team_assignment = "Team 0 is T, Team 1 is CT"
        is_team0_ct = False
    elif blue_count_0 > yellow_count_0 and yellow_count_1 > blue_count_1:
        team_assignment = "Team 0 is CT, Team 1 is T"
        is_team0_ct = True
    else:
        team_assignment = "Unable to determine teams"
        is_team0_ct = None

    return team_assignment, is_team0_ct, frame

#calculate odds base on team weapons and hp
def calculate_odds(weapons, hp_values, timer_label, is_team0_ct):

    team0_power = 0
    team1_power = 0
    team0_alive = 0
    team1_alive = 0

    current_time = timer_label[-1]
    if current_time == '0':
        return 50
    else:
        for i in range(5):  # Team 0
            if hp_values[i] != '0':  # Player is alive
                team0_power += weapon_power(weapons[i]) + int(hp_values[i])
                team0_alive += 1

        for i in range(5, 10):  # Team 1
            if hp_values[i] != '0':  # Player is alive
                team1_power += weapon_power(weapons[i]) + int(hp_values[i])
                team1_alive += 1

        # Bomb advantage
        bomb_factor = 1
        if timer_label == 'bomb':
            bomb_factor = 2  #increase advantage for T when bomb is planted
            if is_team0_ct == False:
                team0_power *= bomb_factor
            else:
                team1_power *= bomb_factor

        total_power = team0_power + team1_power
        if total_power == 0:
            return 50

        team0_odds = (team0_power / total_power) * 100

        return round(team0_odds, 2)

def weapon_power(weapon):
    # weapons power
    power_dict = {
        'ak': 95, 'awp': 100, 'dgl': 50, 'dual': 30, 'five_seven': 45,
        'galil': 80, 'glock': 15, 'm4-s': 90, 'm4a4': 90, 'mac': 60,
        'mp9': 65, 'p250': 30, 'scout': 80, 'tec9': 30, 'usp': 20, 'p2000': 20
    }

    return power_dict.get(weapon, 0)  # Return 0 if weapon not recognized

#function that process specific frame regions
def process_box(frame, box, model, transform, class_labels):
    x, y, w, h = box
    
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return None
    
    roi = frame[y:y+h, x:x+w]
    
    if roi.size == 0:
        return None
    
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_roi)
    input_tensor = transform(pil_image).unsqueeze(0)

    #move input tensor to the same device as model
    input_tensor = input_tensor.to(next(model.parameters()).device)

    try:
        with torch.inference_mode():
            prediction = model(input_tensor)
        
        predicted_class_index = torch.argmax(prediction, dim=1).item()
        predicted_class_label = class_labels[predicted_class_index]
        score = torch.softmax(prediction, dim=1).max().item()
        
        return (x, y, w, h), predicted_class_label, score
    except Exception as e:
        print(f"Error processing box {box}: {str(e)}")
        return None

def get_video_stream(video_path):
    return cv2.VideoCapture(video_path)

#
def update_score(detected_score, current_score):
    try:
        detected_score_int = int(detected_score)
        current_score_int = int(current_score)
        
        if detected_score_int - (current_score_int + 1) == 0:
            return str(detected_score_int)
        else:
            return current_score
    except ValueError:
        # If conversion fails, return the current score
        return current_score

#video capture function
def capture_frames():
    global frame_buffer

    #init video capture from the video
    cap = get_video_stream(video_path)
    #get the original video's frames per second
    target_fps = cap.get(cv2.CAP_PROP_FPS)

    #calculatee how long each frame should be displayed in sec
    frame_interval = 1 / target_fps

    while True:
        start_time = time.time()

        #read a frame | success: bool if frame was successfully read | frame: image data
        success, frame = cap.read()
        #reset video 
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        #lock
        with buffer_lock:
            frame_buffer.append(frame)
        
        #calculate how long the frame processing took
        processing_time = time.time() - start_time

        #calculate how long to sleep to maintain target FPS
        #if processing took longer than frame_interval, sleep will be 0
        sleep_time = max(0, frame_interval - processing_time)

        #maintaining consistent frame rate
        time.sleep(sleep_time)

#transforms to desired size
def transformers(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def process_frames():
    global data, last_detected_scores
    frame_count = 0

    #set up image transformation for different size regions
    transform_small = transformers((20,70))
    transform_big = transformers((100,75))

    while True:
        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.001) #wait if buffer is empty
                continue
            frame = frame_buffer[-1]  # Get the latest frame
        
        frame_count += 1
        
        #check match timer and detect if showing replay
        timer_result = process_box(frame, box_timer, model_timer, transform_small, class_labels_timer)
        timer_value = timer_result[1].replace("_", ":") if timer_result else '0'
        timer_replay = True if timer_value == '0' else False

        #check hud elements to detect replay
        hud_replay = False
        for box in box_hud:
            hud_result = process_box(frame, box, model_hud, transform_big, class_label_hud)
            hud_replay = True if hud_result[1] == '1' else False

        #determine if current frame is replay
        is_replay = (hud_replay or timer_replay == True)

        #update data
        data['timer'] = timer_value
        data['is_replay'] = is_replay
        data['frame'] = frame_count

        #process if not replay
        if not is_replay:
            # detect team sides (ct or t)
            team_assignment, is_team0_ct, processed_frame = detect_team_colors(frame)
            
            results_weapon = []
            results_hp = []

            #process weapon and HP boxes for the players
            for box_weapon, box_hp in zip_longest(boxes, boxes_hp, fillvalue=None):
                #detect weapons
                if box_weapon is not None:
                    result = process_box(processed_frame, box_weapon, model_weapon, transform_small, class_labels_weapon)

                    if result:
                        results_weapon.append(result)

                #detect hp
                if box_hp is not None:
                    result = process_box(processed_frame, box_hp, model_hp, transform_small, class_labels_hp)
                    if result:
                        results_hp.append(result)

            #process score boxes using OCR
            for i, score_box in enumerate(box_score):
                x, y, w, h = score_box
                roi = processed_frame[y:y+h, x:x+w]
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                try:
                    #use tesseract to read score number
                    wynik = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                    if wynik.strip():
                        last_detected_scores[i] = wynik.strip()
                except pytesseract.TesseractError as e:
                    print(f"Tesseract error: {e}")
                    wynik = last_detected_scores[i][-1]

            #extract weapon and hp
            weapons = [r[1] for r in results_weapon]
            hp_values = [r[1] for r in results_hp]

            odds = calculate_odds(weapons, hp_values, [timer_value], is_team0_ct)

            data.update({
                'team1_odds': round(odds, 2),
                'team2_odds': round(100 - odds, 2),
                'team_assignment': team_assignment,
                'player_statuses': ['Alive' if hp != '0' else 'Dead' for hp in hp_values],
                'weapons': ['none' if hp == '0' else weapon for hp, weapon in zip(hp_values, weapons)],
                'hp': ['none' if hp == '0' else hp for hp in hp_values],
                'frame_number': frame_count,
                'team_1_score': update_score(last_detected_scores[0], data['team_1_score']),
                'team_2_score': update_score(last_detected_scores[1], data['team_2_score'])
            })

        time.sleep(0.001)  # Small sleep to prevent this thread from consuming too much CPU

def generate_frames():
    while True:
        #lock for safety
        with buffer_lock:
            if not frame_buffer:
                time.sleep(0.001) #wait if buffer is empty
                continue
            frame = frame_buffer[0] #get oldest frame

        #image enhancement
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.addWeighted(frame, 1.2, frame, 0, 0)

        #convert to jpg
        _, buffer = cv2.imencode('.jpg', frame)

        #convert jpg to bytes
        frame_bytes = buffer.tobytes()
        #yield the frame in multipart http response format to stream video over http
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    global data
    return jsonify(data)

@app.route('/video_feed')
def video_feed():
    #stream processed frames
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the background threads
def init_app():
    global capture_thread, process_thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    process_thread.start()

if __name__ == '__main__':
    init_app()
    app.run(host='0.0.0.0', port=8000, threaded=True)
