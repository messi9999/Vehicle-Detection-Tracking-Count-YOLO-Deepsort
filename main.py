import os
import cv2
import time
import datetime
import argparse
import torch
import warnings
import numpy as np
import sys
from ultralytics import YOLO
from save_data import *
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
import openpyxl
from dotenv import load_dotenv

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        self.classNames = []
        self.idx_object = [2, 3, 5, 7]
        self.currenttime = datetime.datetime.now()
        self.direct = [] 

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        self.temp_a_list = []
        self.temp_b_list = []
        self.temp_c_list = []
        self.temp_d_list = []
        self.ab_list = [0, 0, 0, 0]
        self.ac_list = [0, 0, 0, 0]
        self.ad_list = [0, 0, 0, 0]
        self.bc_list = [0, 0, 0, 0]
        self.bd_list = [0, 0, 0, 0]
        self.cd_list = [0, 0, 0, 0]
        self.ba_list = [0, 0, 0, 0]
        self.ca_list = [0, 0, 0, 0]
        self.da_list = [0, 0, 0, 0]
        self.cb_list = [0, 0, 0, 0]
        self.db_list = [0, 0, 0, 0]
        self.dc_list = [0, 0, 0, 0]     

        load_dotenv()

        self.save_per_time = int(os.getenv('SAVE_PER_TIME'))

        self.xa1 = int(os.getenv('ENV_XA1'))
        self.xa2 = int(os.getenv('ENV_XA2'))
        self.xb1 = int(os.getenv('ENV_XB1'))
        self.xb2 = int(os.getenv('ENV_XB2'))
        self.xc1 = int(os.getenv('ENV_XC1'))
        self.xc2 = int(os.getenv('ENV_XC2'))
        self.xd1 = int(os.getenv('ENV_XD1'))
        self.xd2 = int(os.getenv('ENV_XD2'))
        self.ya1 = int(os.getenv('ENV_YA1'))
        self.ya2 = int(os.getenv('ENV_YA2'))
        self.yb1 = int(os.getenv('ENV_YB1'))
        self.yb2 = int(os.getenv('ENV_YB2'))
        self.yc1 = int(os.getenv('ENV_YC1'))
        self.yc2 = int(os.getenv('ENV_YC2'))
        self.yd1 = int(os.getenv('ENV_YD1'))
        self.yd2 = int(os.getenv('ENV_YD2'))

        self.font_color = (0, 0, 255)
        self.font_size = 0.5
        self.font_thickness = 2

        self.path = os.getcwd()
        self.filename = './ExcelOutput/data_car.xlsx'
        self.isFile = os.path.isfile(self.path + '/' + self.filename)
        if self.isFile == False:
            self.wb = openpyxl.Workbook()
            self.ws = self.wb.active
            self.row = ["Direction", "Type of Car", "Timestamp"]
            self.ws.append(self.row)
        else:
            self.wb = openpyxl.load_workbook(self.filename)
            self.ws = self.wb.active


    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()


        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)    

    def find_center(self, x, y, h):
        cx1 = x
        cy1=y+h//2
        return cx1, cy1

    def insert_row(self, index):
        self.direct.insert(1, self.classNames[self.idx_object[index]])
        self.direct.insert(2, self.currenttime)
        self.ws.append(self.direct)
        self.wb.save(self.filename)
    
    def count_vehicle(self, box_id, img):
        self.ws = self.wb.active
        self.currenttime = datetime.datetime.now()
        x, y, w, h, id, index = box_id
        # Find the center of the rectangle for detection
        center = self.find_center(x, y, h)
        ix, iy = center
        self.direct = []    
        # Draw circle in the middle of the rectangle
        if ix < self.xa1 and ix > self.xa2 and iy < self.ya2 and iy > self.ya1:

            if id in self.temp_b_list:
                self.ba_list[index] += 1
                self.direct.insert(0, "Direction B -> A")
                self.insert_row(index)
                self.temp_b_list.remove(id) 

            elif id in self.temp_c_list:
                self.ca_list[index] += 1
                self.direct.insert(0, "Direction C -> A")
                self.insert_row(index)
                self.temp_c_list.remove(id)

            elif id in self.temp_d_list:
                self.da_list[index] += 1
                self.direct.insert(0, "Direction D -> A")
                self.insert_row(index)
                self.temp_d_list.remove(id)

            else:
                self.temp_a_list.append(id)

        elif ix > self.xb1 and ix < self.xb2 and iy < self.yb2 and iy > self.yb1:

            if id in self.temp_a_list:
                self.ab_list[index] += 1
                self.direct.insert(0, "Direction A -> B")
                self.insert_row(index)
                self.temp_a_list.remove(id)

            elif id in self.temp_c_list:
                self.cb_list[index] += 1
                self.direct.insert(0, "Direction C -> B")
                self.insert_row(index)
                self.temp_c_list.remove(id)

            elif id in self.temp_d_list:
                self.db_list[index] += 1
                self.direct.insert(0, "Direction D -> B")
                self.insert_row(index)
                self.temp_d_list.remove(id)

            else:
                self.temp_b_list.append(id)

        elif ix > self.xc1 and ix < self.xc2 and iy > self.yc1 and iy < self.yc2:

            if id in self.temp_a_list:
                self.ac_list[index] += 1
                self.direct.insert(0, "Direction A -> C")
                self.insert_row(index)
                self.temp_a_list.remove(id)

            elif id in self.temp_b_list:
                self.bc_list[index] += 1
                self.direct.insert(0, "Direction B -> C")
                self.insert_row(index)
                self.temp_b_list.remove(id)

            elif id in self.temp_d_list:
                self.dc_list[index] += 1
                self.direct.insert(0, "Direction D -> C")
                self.insert_row(index)
                self.temp_d_list.remove(id)

            else:
                self.temp_c_list.append(id)

        elif ix < self.xd1 and ix > self.xd2 and iy > self.yd2 and iy < self.yd1:

            if id in self.temp_a_list:
                self.ad_list[index] += 1
                self.direct.insert(0, "Direction A -> D")
                self.insert_row(index)
                self.temp_a_list.remove(id)

            elif id in self.temp_b_list:
                self.bd_list[index] += 1
                self.direct.insert(0, "Direction B -> D")
                self.insert_row(index)
                self.temp_b_list.remove(id)

            elif id in self.temp_c_list:
                self.cd_list[index] += 1
                self.direct.insert(0, "Direction C -> D")
                self.insert_row(index)
                self.temp_c_list.remove(id)

            else:
                self.temp_d_list.append(id)
            


        # cv2.line(img, (self.xa1, self.ya1), (self.xa2, self.ya2), (255, 0, 255), 2)
        # cv2.line(img, (self.xb1, self.yb1), (self.xb2, self.yb2), (255, 0, 255), 2)
        # cv2.line(img, (self.xc1, self.yc1), (self.xc2, self.yc2), (255, 0, 255), 2)
        # cv2.line(img, (self.xd1, self.yd1), (self.xd2, self.yd2), (255, 0, 255), 2)

        return img

    def run(self):
        starttime = time.time()
        lasttime = starttime
        results = []
        idx_frame = 0
        
        model = YOLO('./detector/YOLOv8/weight/yolov8n.pt')
        self.classNames = model.model.names
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
 
            result = model.predict(im)
            bbox_xywh = result[0].boxes.xywh
            bbox_xywh = np.array(bbox_xywh)
            cls_ids = result[0].boxes.cls
            cls_ids = np.array(cls_ids, dtype=int)
            cls_conf = np.array(result[0].boxes.conf)
            mask = [x == 2 or x == 3 or x == 5 or x == 7 for x in cls_ids]
            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]
            names = [self.classNames[int(x)] for x in cls_ids]
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            if len(outputs) > 0:                
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities)                
                for box_xxwh, cid, identity in zip(bbox_xywh, cls_ids, identities):
                    x, y, w, h = box_xxwh
                    i = identity
                    idx = self.idx_object.index(cid)
                    ori_im = self.count_vehicle([x, y, w, h, i, idx], ori_im)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            # cv2.putText(ori_im, "bd", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            # cv2.putText(ori_im, "db", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            # cv2.putText(ori_im, "Car:        "+str(self.bd_list[0])+"     "+ str(self.db_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            # cv2.putText(ori_im, "Motorbike:  "+str(self.bd_list[1])+"     "+ str(self.db_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            # cv2.putText(ori_im, "Bus:        "+str(self.bd_list[2])+"     "+ str(self.db_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            # cv2.putText(ori_im, "Truck:      "+str(self.bd_list[3])+"     "+ str(self.db_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)

            directions = [self.ab_list.copy(), 
                        self.ba_list.copy(), 
                        self.ac_list.copy(), 
                        self.ca_list.copy(), 
                        self.ad_list.copy(), 
                        self.da_list.copy(), 
                        self.bc_list.copy(), 
                        self.cb_list.copy(), 
                        self.bd_list.copy(), 
                        self.db_list.copy(), 
                        self.cd_list.copy(), 
                        self.dc_list.copy()]
            lasttime = time.time()
            if lasttime - starttime > self.save_per_time * 60:
                starttime = lasttime
                save_data4 = Save_data()
                save_data4.save_data("data4.xlsx", directions)


            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False
    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
