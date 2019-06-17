``` python

```
#加载库函数 
import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
#构建model,分两步
#1.调用Sequential函数，设置模型输入、节点数
#2.设置loss与optimizer       Tip:compile(编译)
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
#输入数据（xs为输入，ys为输出（label））       Tip:xs=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],这种写法也对，但np.array效率更高
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
#调用model.fit，epochs为迭代次数
model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]))
#Tip:一般只有字符串才加''，表示数字不加''或""
#https://www.coursera.org/learn/introduction-tensorflow

