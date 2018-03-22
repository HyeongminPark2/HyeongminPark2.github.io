---
layout: post
title: cs231n lecture 2 - Loss Function
feature-img: "assets/img/posts/cs231n-lecture2-loss-function/post-head.png"
thumbnail: "assets/img/posts/cs231n-lecture2-loss-function/post-head.png"
tags: [cnn, cs231n]
---

# Loss function

## SVM - Multiclass Support Vector Machine

$$i$$번째 example의 SVM loss:

$$L_i = \sum_{j \not = y_i} max(0,s_j-s_{y_i}+\Delta)$$

$$\sum_{j \not = y_i} $$ : 모든 class

$$s_j$$ : $$j$$번째 class에 대한 score

$$s_{y_i}$$ : $$y_i$$ (실제 class)에 대한 score

$$\Delta$$ : margin of SVM

### $$\Delta$$ (margin)의 의미

> $$S_j$$(선택한 class의 score)가 $$S_{y_i}$$(실제 class에서 score)보다 $$\Delta$$이상의 격차로
충분히 작아야 Loss에 더하지 않겠다는 뜻. 예를들어 $$S_j=11, S_{y_i}=13$$ 이라면 비록 정확한 class의
score가 더 높았지만($$2$$만큼) margin $$\Delta = 10$$ 을 넘지 않았으므로 나머지 8을 Loss로 치는 것

만약 Score를 구하기 위해 Linear classification 하고 있다면 $$f(x_i,W) = W_{x_i}$$로 classification
score vector를 구할 수 있으므로

$$L_i = \sum_{j=y_i} max(0,w_j^T x_i - w_{y_i}^T x_i + \Delta)$$

- unsquared hinge loss: $$max(0, \cdots )$$
- squared hinge loss: $$max(0, \cdots )^2$$ - 큰 오차에 더 큰 loss를 부과한다

## Regularization

$$W$$가 단순히 커져 overfit 하는것을 막기 위해 regularization penalty $$R(W)$$ 를 Loss에 포함한다.

- 여러가지 방법이 있지만 여기선 $$L2-norm$$을 이용한다

$$R(W) = \underbrace{ {\frac 1 N}\sum_{i}L_i}_{\text{data loss}} + \underbrace{\lambda R(W)}_{\text{regularization loss}}$$

extended:

$$L={\frac 1 N} \sum_{i}\sum_{j \not = y_i}[max(0,f(x_i;W)_j)-f(x_i;W)_{y_i} + \Delta] + \lambda\sum_{k}\sum_{l}W_{k,l}^2$$

## $$\Delta, \lambda$$ 값은 어떻게 정하는가?

어차피 $$f(x,W)$$ 값은 $$W$$ 크기에 따라 클 수도 작을 수도 있기에 $$\Delta$$값을 미리 정하는건 의미가 없다.
(can safely set to $$\Delta = 1.0$$) 따라서 $$\lambda$$값을 통해 $$\Delta$$크기를 제한하는것만 유의하다.

## Softmax classifier

softmax는 SVM처럼 class의 score를 내는 대신 각 class의 normalized된 확률을 제공하기때문에 좀 더 직관적이다

$$f(x_i;W)=W_{x_i}$$는 그대로지만, 이 score를 unnormalized log probability로 간주하고 hinge loss를 cross-entropy loss로 대체한다.

### Cross-entropy loss function

- 전체 Loss는 평균이다

$$\boxed{L_i = -\log(\frac {e^{F_{y_i}}} {\sum_{j} e^{f_j}})}$$ or equivalently $$\boxed{L_i = -f_{y_i} + \log \sum_{j} e^{f_j}}$$

여기서 $$ \frac {e^{z_j}} {\sum_{k} e^{z_k}} = f_j(z)$$ 는 **Softmax function** 이다. 이는 임의의 값들로 이루어진 벡터에서 0과 1
사이로 정규화된 값을 돌려준다.

확률론적 표현: $$ P(y_i \vert x_i;W) = \frac {e^{f_{y_i}}} {\sum_j e^{f_j}}$$

실제로 컴퓨터를 이용해 계산할 경우 발생 가능한 문제:

$$e^{f_{y_i}}$$와 $$\sum_j e^{f_j}$$가 아주 커질 경우 $$\frac {e^{f_{y_i}}} {\sum_j e^{f_j}}$$ 계산에 부동소수점 문제가 발생할 수 있다.

따라서

$$ \frac {e^{f_{y_i}}} {\sum_j e^{f_j}} = \frac {C e^{f_{y_i}}} {C \sum_j e^{f_j}} = \frac {e^{f_{y_i} + \log C}} {\sum_j e^{f_j + \log C}}$$

으로 계산하고 여기서 $$\log C = -\max_j f_j$$ 인데, 각 class에 대한 score중 가장 높은 값을 $$-\log C$$ 로 둬야한다

> Softmax는 확률을 제공하며 never fully happy with score. SVM에서 margin $$\Delta$$ 만족하면 Loss에서 제외하는것과는 다름. 차와 트럭을 구분하는것을 신경써야하는 상황에 개구리를 신경쓸 필요는 없다.

## Summary

- image pixels → class score 매핑하는 score function $$\bold{Wb} \rightarrow \text{biases}$$
- kNN과 달리 모델을 생성하는 parametric approach의 장점은 학습 후 training data가 필요 없으며 $$\bold{W}$$만 곱하면 되므로 미번 모든 학습데이터와 비교하는 kNN보다 빠르다.
- bias trick: $$\bold{W}$$ 에 $$\bold{b}$$를 한 column 추가하는식으로 합치고 $$\bold{x}$$ push 1.
- loss function : 실제 class와 다르게 분류하였을때 패널티 부여 - SVM, Softmax