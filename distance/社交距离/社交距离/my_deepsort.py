import argparse
import warnings

from deep_sort import build_tracker
from deep_utils.draw_test import draw_boxes
from deep_utils.parser import get_config
from detect import *


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # if args.display:
        #     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        #     #cv2.resizeWindow("test", args.display_width, args.display_height*2)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
            self.vdo.open(self.args.VIDEO_PATH)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height*2))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        # Load yolov3_tiny_se detect
        weights = 'yolov5s.pt'
        device = torch_utils.select_device(device='0')
        half = device.type != 'cpu'

        model = torch.load(weights, map_location=device)['model'].float()
        model.to(device).eval()
        if half:
            model.half()

        fps_list = []
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im_blank = ori_im.copy()
            im_blank.fill(255)
            print(im.shape)
            start = time.time()
            # do detection
            bbox_xywh, cls_conf, cls_ids, bbox_xyxy1 = detector(im, model, device, half)

            if bbox_xywh is not None:
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]	
                    ori_im, im_blank = draw_boxes(ori_im, im_blank, bbox_xyxy, identities)
                    ori_im = np.vstack((ori_im, im_blank))

            end = time.time()
            fps = 1/(end-start+0.001)
            fps_list.append(fps)
            print("total-time: {:.03f}s, fps: {:.03f}".format(end-start, fps))

            if self.args.display:
                cv2.imshow("t1", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
        avg_fps = np.mean(fps_list)
        print("avg_fps: {:.03f}".format(avg_fps))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH",default='1.mp4', type=str)
    parser.add_argument("--config_deepsort", type=str, default="deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
