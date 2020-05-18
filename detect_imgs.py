"""
This code is used to batch detect images in a folder.
"""
import argparse
import sys
from pathlib import Path

import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="/home/ppotrykus/Datasets/image/face_thermo/from_video/thermalVideo_1589106539987", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--output', default="./detect_results_gray", type=str,
                    help='specify target output path')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


def is_image(path:Path):
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png'}


imgs_path = Path(args.path)
result_path = Path(args.output)
result_path.mkdir(exist_ok=True)
label_path = "./models/voc-model-labels.txt"
test_device = args.test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
elif args.net_type == 'RFB':
    # model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    # model_path = 'models/pretrained/RFB-640-gray-resume.pth'
    # model_path = "models/RFB-ir_tufts-Adam-1e-4/RFB-Epoch-40-Loss-0.32918739318847656.pth"
    # model_path = "models/RFB-ir_tufts-Adam-1e-5/RFB-Epoch-999-Loss-0.47017228603363037.pth"
    model_path = "models/RFB-ir_tufts_full-Adam-1e-5/RFB-Epoch-400-Loss-1.3762706858771188.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test= True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)
print(f'loaded model: {model_path}')

sum = 0
for file_path in imgs_path.glob('**/*'):
    if not is_image(file_path): continue
    orig_image = cv2.imread(file_path.as_posix())
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        rect = tuple(box.int().tolist())
        # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.rectangle(orig_image, rect[0:2], rect[2:4], (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{probs[i]:.2f}"
        # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if boxes.numel():
        cv2.namedWindow(file_path.name)
        cv2.imshow(file_path.name, orig_image)
        cv2.waitKey()
        cv2.destroyWindow(file_path.name)
    cv2.imwrite((result_path/file_path.name).as_posix(), orig_image)
    print(f"Found {len(probs)} faces. The output image is {result_path/file_path.name}")
print(sum)
