import argparse
import time
from pathlib import Path

import cv2
# from numpy.core.fromnumeric import shape
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

"""
整个网络检测流程：
1. 创建保存输出文件的文件夹
2. 加载模型文件，并做一些检查确保图片尺寸符合网络模型需求
3. 根据输入源进行不同的数据加载方式
4. 迭代数据集进行模型前向推断
   - 非极大值抑制
   - 为输出路径添加图片名称
   - 调整预测框(resize + padding --> 原图片坐标)
   - 原图上画框和标签置信度、以及保存坐标框位置
   - 保存图片、以及坐标框文件

"""

def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://')) # 是否使用摄像头或网址视频等

    # Directories 设置图片保存路径
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # make dir 创建文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging() # 初始化日志文件
    device = select_device(opt.device) # 选择训练的设备 cpu, cuda
    half = device.type != 'cpu'  # half precision only supported on CUDA 如设备为cpu, 则不能使用半精度

    # Load model 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size 确保图片能整除 32 （如果不能则调整为能整除并输出）
    if half:
        model.half()  # to FP16

    # Second-stage classifier 设置第二层分类器，默认不使用
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize 初始化模型
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    # 用来判断是否是另一个新视频
    vid_path, vid_writer = None, None
    # Set Dataloader 根据不同的输入源来设置不同的数据加载方式
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz) # 加载摄像头视频流
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz) # 加载图像或视频

    # Get names and colors
    # 获得数据集标签名称 以及为每一类设置一种颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once 试运行一次，判断模型是否正常
    # 开始迭代
    t0 = time.time()
    """
    path：图片路径
    img：进行 resize 和 pad 之后的图片 (c, h, w) 
    img0s：原图片
    vid_cap：判断是否是视频流
    """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # 如果没有 batch_size or batch_size=1，在最前面添加一个轴
            img = img.unsqueeze(0)

        """
        Inference
        框数量：(h/32 * w/32 + h/16 * w/16 + h/8 * w/8) * 3
        pred 为模型输出结果（预测框）
        pred[:,0:4]为预测框坐标
        pred[:,4] 为 objectiveness 置信度
        pred[:,5:-1] 为分类概率结果
        """
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        """
        pred
        conf_thres
        iou_thres
        classes 是否保留特定类别
        agnostic_nms 进行 nms 是否去除不同类别的框
        max_det 最大的预测框数量
        经过 nms 后 xywh -> xyxy
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized() # 使用时间同步记录当前时间(可能多个 GPU)

        # Apply Classifier 做二级分类
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image (实际上每次只有一张图片)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p 是原图片路径
            p = Path(p)  # to Path
            # 设置保存图片或视频的路径
            save_path = str(save_dir / p.name)  # img.jpg
            # 设置保存预测框坐标 .txt的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # 设置打印信息
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 用于四个坐标 xyxy 的归一化
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于 resize + pad 图片的坐标 --> 基于原 size 图片的坐标
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results 统计并记录检测到的每类的数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                # 保存预测结果，依次保存每一个输出框，以及画出每一个输出框。
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 先 xyxy --> xywh，再归一化、转化为 list 再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # 直接在原图画框
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS) # 输出一张图片推断时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results 设置展示
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results (image with detections) 保存图片
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # 判断是不是另一个新的视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap: # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else: # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[2]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    # 打印总时间
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp8/weights/best.pt', help='model.pt path(s)') # 模型权重文件
    parser.add_argument('--source', type=str, default='D:/hero.mp4', help='source')  # file/folder, 0 for webcam # 数据源路径，0 则是使用摄像头
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)') # 输入模型的图片尺寸格式
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') # 置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') # IoU 阈值
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # 使用设备 
    parser.add_argument('--view-img', action='store_true', help='display results') # 检测时是否展示图片
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # 保存预测框数据
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # 保留置信度
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') # 只检测哪几类
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') # 进行 nms 是否去除不同类别的框
    parser.add_argument('--augment', action='store_true', help='augmented inference') # 前向推断使用图像增广
    parser.add_argument('--update', action='store_true', help='update all models') # 
    parser.add_argument('--project', default='runs/detect', help='save results to project/name') # 预测结果输出文件目录
    parser.add_argument('--name', default='exp', help='save results to project/name') # 文件夹名称
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') # 直接在已经存在的文件夹添加，不新建文件夹
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
