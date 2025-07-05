from ultralytics import YOLO

# 加载模型
model = YOLO('model/yolo11s.pt')
# print(model)
# 预测结果
results = model.predict(source='image',
                        show=False,     # 是否显示图像
                        save=True       # 是否保存检测结果
                        )
# 遍历结果
for result in results:
    print(result.boxes.xyxy)



# # 执行预测
# results = model.predict(
# source='path/to/image.jpg', # 输入源，可以是图像路径、视频路径、摄像头索引等
# conf=0.5, # 置信度阈值
# iou=0.45, # NMS（非极大值抑制）的交并比阈值，用于消除重叠的检测框
# device='cuda', # 使用GPU
# imgsz=(640, 640), # 输入图像的尺寸
# half=False, # 不使用半精度浮点数
# save_txt=True, # 保存预测结果为文本文件
# save_conf=True, # 在保存的文本文件中包含置信度
# save_crop=False, # 不保存裁剪后的检测结果
# save_img=True, # 保存带有检测框的图像
# classes=[0, 1, 2], # 仅检测指定的类  过滤类别
# agnostic_nms=False, # 不使用类别无关的NMS
# augment=False, # 不使用数据增强
# visualize=False, # 不可视化特征图
# project='runs/detect', # 保存预测结果的项目目录
# name='exp', # 保存预测结果的子目录名称
# exist_ok=False, # 不允许项目目录存在
# line_thickness=3, # 绘制检测框的线条粗细
# hide_labels=False, # 显示检测框的标签
# hide_conf=False # 显示检测框的置信度
# )