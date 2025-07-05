from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/detect/train/weights/best.pt")
    model.val(data=r"mydata.yaml",
              batch=32,
              workers=0,
              imgsz=640)