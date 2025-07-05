from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/detect/train/weights/best.pt")
    # results = model.predict(source=r'datasets/images/val', stream=True, show=False, imgsz=640, save=True)
    results = model.predict(source=r'image_test', stream=True, show=False, imgsz=640, save=True)

    for result in results:
        print(result.boxes.xyxy)
