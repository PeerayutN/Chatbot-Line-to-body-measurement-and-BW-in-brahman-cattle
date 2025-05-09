from __future__ import unicode_literals

import errno
import os
import sys
import tempfile
from dotenv import load_dotenv

from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, PostbackEvent, StickerMessage, StickerSendMessage, 
    LocationMessage, LocationSendMessage, ImageMessage, ImageSendMessage)

import time
from pathlib import Path

import cv2
from ultralytics import YOLO 
from utils.plots import Annotator

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# reads the key-value pair from .env file and adds them to environment variable.
load_dotenv()

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')


### YOLOv11 ##
# Directories
save_dir = 'static/tmp/'

# Load YOLOv11 model
model = YOLO('best_HD.pt')

# function for create tmp dir for download content
def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise

@app.route("/", methods=['GET'])
def home():
    return "Object Detection API"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    if text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' + profile.display_name),
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))
    else:
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.message.text))


@handler.add(MessageEvent, message=LocationMessage)
def handle_location_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        LocationSendMessage(
            title='Location', address=event.message.address,
            latitude=event.message.latitude, longitude=event.message.longitude
        )
    )


@handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        StickerSendMessage(
            package_id=event.message.package_id,
            sticker_id=event.message.sticker_id)
    )


@handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.' + ext
    os.rename(tempfile_path, dist_path)

    im_file = open(dist_path, "rb")
    im = cv2.imread(im_file)

    # Make a copy of the image to draw on it
    im0 = im.copy()

    # Step 1: Convert the input image to grayscale
    gray_im = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("frame.png",gray_im)
    h, w, _ = im0.shape
    
    # Step 2: Use YOLOv11 to predict objects in the grayscale image
    results = model.predict(source="frame.png", imgsz=640, conf=0.75, iou=0.50, show=False, verbose=False)
    print(results)

    annotator = Annotator(im0)  # Continue using the original colored image for annotation

    # Step 3: Create an overlay for transparency
    overlay = im0.copy()

    alpha = 0.2  # Transparency level

    for r in results:
        # Coordinates for the box (xmin, ymin, xmax, ymax)
        for box in r.boxes.xyxy:
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))

            # Calculate width and height of the box
            w = int(box[2]) - int(box[0])  # width
            h = int(box[3]) - int(box[1])  # height
            aspect_ratio = w / h
            print(aspect_ratio)

            BW = round((7.52*h*0.213877921041242)+(4.12*w*0.213877921041242)-461.05)

            label = f'BL/HD: {aspect_ratio:.2f}, BW: {BW} kg'

            # Step 4: Draw semi-transparent box on the original colored image
            cv2.rectangle(overlay, top_left, bottom_right, color=(0, 255, 255), thickness=-1)
            cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0, im0)

            # Draw label text on the original colored image
            annotator.box_label((box[0], box[1], box[2], box[3]), label, color=(0, 0, 0))

    # Step 5: Save the final image with detections in color
    save_path = str(save_dir + os.path.basename(tempfile_path) + '_result.' + ext) 
    cv2.imwrite(save_path, im0)

    url = request.url_root + '/' + save_path

    line_bot_api.reply_message(
        event.reply_token, [
            TextSendMessage(text=f'BL/HD: {aspect_ratio:.2f}, BW: {BW} kg'),
            ImageSendMessage(url, url)
        ])


@app.route('/static/<path:path>')
def send_static_content(path):
    return send_from_directory('static', path)

# create tmp dir for download content
make_static_tmp_dir()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
