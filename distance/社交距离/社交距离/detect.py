from utils.utils import *
from utils.datasets import *


def detector(frame, model, device, half):
    img_size = 640
    img0 = frame
    img = letterbox(img0, new_shape=img_size)[0]

    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.1)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                bbox_xywh1 = []
                bbox_xyxy1 = []
                cls_conf1 = []
                cls_id1 = []
                for *xyxy, conf, cls in det:
                    cls_confirm = int(cls.cpu().numpy())
                    xyxy = [x.cpu() for x in xyxy]
                    if cls_confirm == 0:
                        bbox_xyxy1.append(xyxy)
                        bbox_xywh1.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())
                        cls_conf1.append(conf.item())
                        cls_id1.append(int(cls))


                bbox_xyxy1 = np.array(bbox_xyxy1)
                bbox_xywh1 = np.array(bbox_xywh1)
                cls_conf1 = np.array(cls_conf1)
                cls_id1 = np.array(cls_id1)

                return bbox_xywh1, cls_conf1, cls_id1, bbox_xyxy1
            else:
                return None, 0, 0, None
