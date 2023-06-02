from flask import Flask, render_template, request, json
from os import environ
import importlib
import cv2
import numpy as np
import base64
import sys
from datetime import datetime

app = Flask(__name__)

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.SIFT_create()
        self.smoothing_window_size=800

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

@app.route('/')
@app.route('/home')
def home():
    return render_template(
        'home.html',
        title='Главная',
        year=datetime.now().year,
    )

@app.route('/test')
def test():
    return render_template(
        'test.html',
        title='Тест',
    )

@app.route('/api/gereration', methods = ['POST'])
def gereration():
    data = request.get_json()

    if data['file1'] and data['file2']:
        decoded_data1 = base64.b64decode(data['file1'])
        np_data1 = np.fromstring(decoded_data1 ,np.uint8)
        img1 = cv2.imdecode(np_data1, cv2.IMREAD_UNCHANGED)

        decoded_data2 = base64.b64decode(data['file2'])
        np_data2 = np.fromstring(decoded_data2 ,np.uint8)
        img2 = cv2.imdecode(np_data2, cv2.IMREAD_UNCHANGED)

        final = Image_Stitching().blending(img1,img2)
        cv2.imwrite('panorama.jpg', final)

        with open("matching.jpg", "rb") as img_file:
            matching = base64.b64encode(img_file.read()).decode('utf-8')

        with open("panorama.jpg", "rb") as img_file:
            panorama = base64.b64encode(img_file.read()).decode('utf-8')


        response = app.response_class(
            response= json.dumps({
                "matching" : str( matching ),
                "panorama" : str( panorama )
            }),
            status=202,
            mimetype='application/json'
        )
        return response

    else:
        return {
            "error" : "file not found"
        }, 404

if __name__ == "__main__":
    HOST = environ.get('SERVER_HOST', '0.0.0.0')
    try:
        PORT = int(environ.get('SERVER_PORT', '8000'))
    except ValueError:
        PORT = 5000
    app.run(HOST, PORT)