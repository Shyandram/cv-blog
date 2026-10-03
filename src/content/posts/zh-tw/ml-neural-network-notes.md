---
title: "機器學習及類神經網路筆記"
description: "快速掌握類神經網路的基本觀念：感知器、MLP、激活函數、輸出層設計與訓練方法。"
tags: [machine-learning, artificial-neural-network]
published: 2024-02-02
original: https://medium.com/@shyangen104/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%8F%8A%E9%A1%9E%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%E7%AD%86%E8%A8%98-778c0a527bfe
---

## 機器學習及類神經網路學習筆記

快速上手基本的觀念  
Feb 1, 2024

## Artificial Neural Network 類神經網路

### Perceptron

又稱為logistic regression，發想自神經元的運作。由線性回歸和activation function組成

![Figure 1](./images/ml-neural-network-notes/01.png)

### MultiLayer Perceptron (MLP)

疊加複數個logistic regression形成

![Figure 2](./images/ml-neural-network-notes/02.png)

### Deep Neural Network (DNN)

MLP持續疊加，即為深度學習的神經網路，通常定義為三層以上稱為Deep Neural Network

![Figure 3](./images/ml-neural-network-notes/03.png)

### Activation Function

注意：若在perceptron or MLP各個節點沒有使用nonlinear activation function，函式仍為線性

1. Softmax
2. Sigmoid
3. ReLU
4. Leaky ReLU

### Normalization

![Figure 4](./images/ml-neural-network-notes/04.png)

### Design of Output Layer

根據目的採取不同的輸出層模式

- **普通的output**

接上特定的activation Function，把數值鎖在固定區間，根據目的變化。

E.g. Sigmoid為\[0,1\], 適合用於回歸問題上

- **Softmax layer**

![Figure 5](./images/ml-neural-network-notes/05.png)

### Deep Learning

越深(depth)越寬(width)學習能力越強，1990年代發展以深度(width)為主（如SVM），2000年後以深度(depth)為主並有突破性的發展。

![Figure 6](./images/ml-neural-network-notes/06.png)

## Training

### Backpropagation (Gradient Decent)

Chain Rule,…

[Introduction to Machine Learning](/en/blog/introduction-to-machine-learning/)

### Optimization

皆為基於SGD的優化方法

SGD, SGD with momentum, RMSprop, Adam, …

### Mini-Batch (或簡稱Batch)

如何選擇minibatch的大小(batch size)攸關收斂性，理論上越大越好

太小

- 訓練時間拉長、mini-batch差異大，導致梯度震盪大、不容易收斂

太大

mini-batch差異極小，容易停在local minimum

### Iteration and Epoch

迭代次數

### Loss Function 損失函數

Classification: Cross entropy loss

Regression: MSE (L2 Loss), MAE(L1 Loss)

### Evaluation of Loss Curve

![Figure 7](./images/ml-neural-network-notes/07.png)

四種狀況，觀察其變化趨勢

- A: loss都不變

1. 若loss數值高，代表幾乎不具收斂性。數據集or模型需要替換
2. 若loss數值低，代表已經收斂，可能問題過於簡單或iteration間隔過大(一個batch的資料太多)，無法顯示其趨勢

- B: Overfitting

Early stopping，在minimum loss時結束訓練

- C: fitting或 underfitting。

1. 若已經達到optimal的狀態可結束訓練，通常是訓練集和驗證集高度相似才會發生
2. 若是underfitting，可增加iteration使其接近收斂情形。

- D: 驗證數值高於訓練數值，穩定下降。

常見情況，停止在lowest validation loss

![From textbook Deep Learning with Pytorch](./images/ml-neural-network-notes/08.png)

*From textbook Deep Learning with Pytorch*

### Capacity, Underfitting, Overfitting

**Capacity**

模型的能力誤差曲線也相似於訓練曲線

![Figure 9](./images/ml-neural-network-notes/09.png)

**Underfitting**

1. Check your code (bug or input/target)
2. Make model bigger (if high testing error, for the guarantee of convergence) or smaller (if low testing error, for faster convergence)
3. select better optimization way(Optimizer, Learning Rate)

**Overfitting (Low Training Error but High Testing Error)**

1. Early Stopping
2. Check your dataset (Divergence between the Training set and validation set (or testing set))

### Vanishing and Exploding gradient problem

- Gradient Vanishing

深度學習網路常見的問題，由於深度過深，不斷地反向傳播，梯度隨著深度增加越變越小（指數級別變小，幾乎消失）。

- Gradient Exploding

深度學習網路、初始權重數值過大，發現權重數值變極大，輸出數值變成NaN or Inf。

### **補救方法**

- Activation Function (ReLU),
- Normalization (Gradient Normalization, Batch Normalization),
- Residual Connection
- Weight Initialization
- Fine Tuning
- Reducing the model size

### 如何加強模型效能

- data augmentation
- more training data
- Stronger Model (Deeper, better structure for better convergence)
- better training strategy (optimizer, …)

## References

- [Deep Learning](https://www.deeplearningbook.org/)
- [Deep Learning with PyTorch - Eli Stevens, Luca Antiga, and Thomas Viehmann](https://www.manning.com/books/deep-learning-with-pytorch)
- [ML 2021 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)
- [浅析深度学习中Batch Size大小对训练过程的影响](https://zhuanlan.zhihu.com/p/83626029)
