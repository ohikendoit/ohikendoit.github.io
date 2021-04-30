---
layout: single
title: "Exercise 2. Handwriting Recognition"
date: 2020-04-28
comments: true
categories: 
    [TensorFlow]
tags:
    [deeplearning.ai, coursera, MNIST, 텐서플로우 개발자 자격증]
toc: false
publish: true
classes: wide
comments: true
---

텐서플로우 개발자 자격증 (TensorFlow Developer Certification) 시험 준비를 위해 코세라 텐서플로우 심화과정 (TF in Practice Specialization) 을 복습하고 있습니다.

**Task:** In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?

(Reference: https://www.coursera.org/learn/introduction-tensorflow/notebook/FExZ4/exercise-2-handwriting-recognition)

```python
# YOUR CODE SHOULD START HERE
class callback(tf.keras.callbacks.Callback): #using callbacks to control training
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.99): #99%의 정확도를 달성하면 학습을 중단합니다
      print("\n Reached 99% accuracy so canceling training!")
      self.model.stop_training=True
# YOUR CODE SHOULD END HERE


import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# YOUR CODE SHOULD START HERE
x_train, x_test = x_train/255, x_test/255  #0-1값으로 정규화를합니다
# YOUR CODE SHOULD END HERE


model = tf.keras.models.Sequential([
# YOUR CODE SHOULD START HERE
tf.keras.layers.Flatten(input_shape=(28,28)), #60000만개의 28X28 이미지 인풋
tf.keras.layers.Dense(784, activation=tf.nn.relu),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)    
# YOUR CODE SHOULD END HERE
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# YOUR CODE SHOULD START HERE
result = model.fit(x_train, y_train, epochs=10, callbacks=[callback()])
# YOUR CODE SHOULD END HERE
```

```  
Epoch 1/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.1905 - accuracy: 0.9435
Epoch 2/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0758 - accuracy: 0.9765
Epoch 3/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0486 - accuracy: 0.9847
Epoch 4/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0342 - accuracy: 0.9891
Epoch 5/10
1872/1875 [============================>.] - ETA: 0s - loss: 0.0259 - accuracy: 0.9918
 Reached 99% accuracy so canceling training!
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0259 - accuracy: 0.9918
```