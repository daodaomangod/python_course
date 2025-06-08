import joblib

# 加载保存的模型
load_model = joblib.load('model.joblib')

# 使用加载的模型进行预测
x_val = [[0.4], [0.9]]  # 测试数据的特征值
y_val = load_model.predict(x_val)  # 测试数据的预测值

print(f'输出结果：{y_val}')