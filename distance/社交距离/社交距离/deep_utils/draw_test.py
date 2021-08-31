import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
pp_name = {}


def draw_dis(img, point_list):
    for i in range(len(point_list)):
        origin = point_list.copy()
        left_point = origin.pop(i)
        dist = {}
        for j in range(len(origin)):
            dist0 = np.sqrt(np.square(left_point[0]-origin[j][0])+np.square(left_point[1]-origin[j][1]))
            dist[j] = dist0
        if dist:
            m = min(dist, key=dist.get)
            cv2.putText(img, str(round(dist[m], 2)), left_point, 0, 0.5, (0, 0, 255), 2)
            cv2.line(img, left_point, origin[m], (255, 0, 0), 2)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class Person():
    def __init__(self, id):
        self.id = id
        self.tracklets = []

    def append_center(self, center):
        self.tracklets.append(center)
        if len(self.tracklets) > 50:
            self.tracklets = self.tracklets[-50:-1]


def draw_boxes(img, im0, bbox, identities=None):
    v = int(0)
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        center = (int((x1+x2)/2), int((y1+y2)/2))
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        ppname = 'person_' + str(id)
        if ppname not in pp_name.keys():
            pp_name[ppname] = Person(id)
        pp_name[ppname].append_center(center)
        count = len(pp_name)
        v += 1
        color = compute_color_for_labels(id)
        # 轨迹
        pts = pp_name[ppname].tracklets

        for j in range(1, len(pts)):
            if pts[j - 1] is None or pts[j] is None:
                continue
            cv2.line(img, (pts[j - 1]), (pts[j]), (color), 3)

        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(im0,(x1, y1),(x2,y2),color,3)
    # 车之间的距离
    list_nodo = []
    for z in range(len(identities)):
        id0 = identities[z]
        pp = 'person_' + str(id0)
        list_nodo.append(pp_name[pp].tracklets[-1])
    draw_dis(im0, list_nodo)

    cv2.putText(img, "Total Object Counter: " + str(count), (int(10), int(110)), 0, 1, (0, 255, 0), 2)
    cv2.putText(img, "Current Object Counter: " + str(v), (int(10), int(70)), 0, 1, (0, 255, 0), 2)
    return img, im0


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
