# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
print(1)
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
print(2)
import tools.infer.utility as utility
print(2.5)
from ppocr.utils.utility import initial_logger
print(3)
logger = initial_logger()
print(4)
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy

print(5)
import numpy as np
import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
print(6)
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt
###
#from socket import *
import time
import socket
import threading
from multiprocessing import Process


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)
    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

data_text = 0



def main(args):
    ##
    start = time.clock()

    print("1")
    openfile = data_text + "_result.txt"
    f = open(openfile, 'w+')
    print("2")
    image_file_list = get_image_file_list(args.image_dir)
    print("3")
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    print("4")
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            if image_file == data_text:
                img = cv2.imread(image_file)
                print("dddddddddddd",image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            logger.info("error in loading image:{}".format(data_text))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("####")
        print("Predict time of %s: %.3fs" % (image_file, elapse))
        print("####")

        drop_score = 0.5
        dt_num = len(dt_boxes)
        for dno in range(dt_num):
            text, score = rec_res[dno]
            if score >= drop_score:
                text_str = "%s, %.3f \n" % (text, score)
                text = "%s\n" % (text)
                print(text_str)
                ###
                f.writelines(text)
                
        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))
            ####
            """f.close()
            sockfd = socket()
            server_addr = ("192.168.143.14",8888)
            sockfd.connect(server_addr)
            print("connect success")
            #time.sleep(2)

            try:
                while True:
                    sockfd.send(b'hello')
            except IOError as e:
                print(e.strerror)
            sockfd.close()"""
    end = time.clock()    
    print('Running time: %s Seconds'%(end-start))

def handle_client(client_socket):
    global data_text
    request_data = client_socket.recv(1024)
    print("request data:", request_data)
    data_text = request_data
    #data_text = str(request_data,'utf-8').split(':')
    data_text = data_text.decode('utf-8')
    print("data_text:",data_text)
    #print("[%s]********************" % data_text)
    # 构造响应数据
    response_start_line = "HTTP/1.1 200 OK\r\n"
    response_headers = "Server: My server\r\n"
    response_body = "<h1>"+"Python HTTP1 Test"+"</h1>"
    response = response_start_line + response_headers + "\r\n" + response_body

    # 向客户端返回响应数据
    client_socket.send(bytes(response, "utf-8"))
    
    # 关闭客户端连接
    client_socket.close()

if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", 8000))
    server_socket.listen(128)
    while True:
        client_socket,client_address = server_socket.accept()
        print("[%s, %s]用户连接上了" % client_address)
        #handle_client_process = Process(target=handle_client, args=(client_socket,))
        #handle_client_process.start()
        #time.sleep(3)
        handle_client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        handle_client_thread.start()
        #handle_client_thread.join()
        time.sleep(0.2)
        main(utility.parse_args())
        handle_client_thread.join()
        client_socket.close()   
 
