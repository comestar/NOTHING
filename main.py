import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 初始参数
w, b = 0.0, 0.0
lr = 0.1  # 学习率
x = 2
y = 4

# 保存参数变化过程
w_list = [w]
b_list = [b]
loss_list = []

# SGD 收敛过程迭代
for epoch in range(200):
    # 预测值
    y_hat = w * x + b
    # 计算损失
    loss = 0.5 * (y_hat - y) 
    # 计算梯度
    grad_w =   x
    grad_b =   1
    # 参数更新
    w -= lr * grad_w
    b -= lr * grad_b

    # 记录
    w_list.append(w)
    b_list.append(b)
    loss_list.append(loss)

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# w 和 b 的变化过程
ax[0].plot(w_list, label='w')
ax[0].plot(b_list, label='b')
ax[0].set_title('参数收敛过程')
ax[0].set_xlabel('迭代次数')
ax[0].set_ylabel('值')
ax[0].legend()

# 损失变化
ax[1].plot(loss_list, marker='o')
ax[1].set_title('损失函数下降过程')
ax[1].set_xlabel('迭代次数')
ax[1].set_ylabel('Loss')

plt.tight_layout()
plt.show()
