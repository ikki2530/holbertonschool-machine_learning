#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('2-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    # print("boxes", type(boxes))
    # print("box_confidences", type(box_confidences))
    # print(len(box_confidences))
    # print("box_class_probs", type(box_class_probs))
    # print(len(box_class_probs))
    boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
    print("type boxes", type(boxes[0]))
    print("boxes dims", boxes.shape)
    print("box_classes dims", box_classes.shape)
    print("box_scores dims", box_scores.shape)
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)
