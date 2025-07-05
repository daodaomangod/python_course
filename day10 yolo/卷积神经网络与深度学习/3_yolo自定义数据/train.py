from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'model/yolo11n.pt')
    model.train(data=r'mydata.yaml',
                epochs=50,
                device=[0], #CPU训练将device=[0]改为device='cpu'
                batch=32,
                workers=0,
                cache=True,
                imgsz=640,
                )