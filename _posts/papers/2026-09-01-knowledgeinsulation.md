---
layout: single
title: "Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better 리뷰"
categories: paper
tag: [Robotics]
author_profile: false
sidebar:
    nav: "counts"
toc: true
toc_sticky: true
toc_label: Table of Contents
use_math: true
---

# Background

LLM과 VLM을 physical action에 특화시키기 위해서는 어떻게 해야 할까? 일반적인 Language task보다 action task가 어려운 이유는 action task는 연속적이면서 자세한 action을 요구하면서 실시간으로 높은 frequency을 요구하기 때문이다.

이런 문제를 해결하기 위해서 pretrained VLM 뒤에 Diffusion, Flow matching 학습 기반의 transformer을 붙여서 action을 chunk 단위로 추론하는 구조가 주로 사용된다. 

본 논문은 이 구조로 학습을 할 때 생기는 문제점을 지적한다. Pretrained 된 VLM에 random initialize된 transformer을 붙이고 바로 joint training을 하게 되면 초기 학습 단계에서 transformer가 VLM에 random에 가까운 gradient을 주기 때문에 VLM의 generalized knowledge / VLA 전체의 성능이 떨어질 수 있다고 주장한다.

# Standard VLA training recipes

## Action Representation

Robot의 action은 보통 robot의 joint 값이나 end effector 좌표값을 사용한다. 

Robot의 실시간 action을 예측하는 AI를 만들기 위해서 보통 시간 H 동안의 action을 하나의 action chunk으로 설정한다. 이때 이 action chunk을 표현하는데에 여러 방법이 존재한다.

**Naiive Discretization**

Action chunk의 한 action을 하나의 token으로 변환해서 action chunk을 여러 개의 discrete한 action token으로 표현하는 방법이다.

Cross entropy loss을 사용해서 Next token prediction처럼 학습이 되는데 language model의 본래 training objective와 동일한 objective을 사용하기 때문에 objective 간의 gap이 발생하지 않는다.

하지만 action sequence가 매우 길어질 수 있어서 학습 및 추론 속도가 매우 오래 걸린다.

**Temporal action abstractions**

FAST tokenizer처럼 여러 timestep의 action 정보를 시간에 대해서 압축시켜서 문제를 완화하는 방법이다. 즉, 동일한 autoregressive objective을 사용하되, VLM이 예측해야 하는 token sequence 길이를 줄이는 것이다.

**Diffusion and flow matching**

Autoregressive 방법에서 벗어나서 직접 continuous한 action을 직접 생성하는 것이다. 



# Problems with standard VLA recipes

**Autoregressive VLA는 느리다**

Action을 discrete한 token으로 여기고 next token prediction으로 학습하는 VLA는 무조건 추론을 할 때도 autoregressive 하게 해야 하기 때문에 inference 속도가 매우 느리다.이로 인해서 real time이 중요한 robot action에 대해서는 성능이 좋지 않다.

**Robotic specific architectures and modality adapters don’t benefit as much from VLM pretraining**

PI 연구나 GROOT 계열의 VLA에서 Action expert라고 불리는 모듈을 붙여서 학습을 하는데 원하는 control frequency을 맞추기 위해서 보통 VLM보다 크기가 훨씬 적은 transformer을 사용한다. 

이때 이 모듈들은 모두 random initialize 된 상태로 학습에 들어가기 때문에 바로 같이 joint하게 학습되었을 때 모델이 language을 따르는 능력이 떨어지는 경향을 보인다.

![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure2.png)


**VLM pretraining does not have sufficient representations for robotics**

또, VLM pretraining은 보통 QA dataset을 사용하기 때문에 VLM의 representation은 모두 language-image space에 제한되어있다. 따라서 VLM의 knowledge을 보존하기 위해서 freeze을 하더라도 이 representation들은 action에 특화되어있지 않기 때문에 robotics에 적용될 때 좋은 성능을 내지 못한다.


#  Improving VLAs with co-training, joint-training & knowledge insulation

##  Co-training & representation learning with joint discrete/continous action prediction

이런 문제점들을 해결하기 위해서 VLA을 학습할 때 autoregressive 와 flow matching 둘 다 사용하는 co-training 방법을 제안한다. VLM 자체는 autoregressive objective으로 representation learning을 하고 action expert는 flow matching objective으로 학습하도록 하였다.

본 논문은 두 종류의 학습 방법을 한 모델에 적용했을 때 VLM data로 학습을 하면서도 (autoregressive으로 VLM을 학습하므로), 효과적인 knowledge transfer가 가능해지고, 빠른 학습도 가능 (flow matching 학습은 denoising step으로 인해서 autoregressive보다 느리다)하다고 주장한다.

이 방법을 통해서 VLA으로부터 action chunks $a_{1:H}$ 과 text $\hat l$ (language token 또는 discrete action token) 을 둘 다 얻을 수 있게 된다. $(a, \hat{\ell}) \sim \pi\left(\cdot, \cdot \mid I^{1:V}, q, \ell\right)$ 

> 이때 Discrete action token을 위해서 FAST tokenizer을 사용한다

$$\mathcal{L}_{\text{CO-VLA}}(\theta) = \mathbb{E}_{\mathcal{D}, \tau, \omega} \left[ - \sum_{j=1}^{n-1} M_j^\ell \log p_\theta\left(\hat{\ell}_{j+1} \mid x_{1:j}\right) + \alpha M_{\text{act}} \left\| \omega - a_{1:H} - f_\theta^a\left(a_{1:H}^{\tau, \omega}\right) \right\|^2 \right]$$

여기서 $M^l$ 은 language loss mask이고 $M^{act}$ 는 action loss mask이다. 이 loss function을 통해 다양한 modality의 data를 co-train 할 수 있게 된다. 

학습을 할 때 VLM data (text - image), action only data(image와 text으로 action 예측), 그리고 combined data(action only data에 로봇이 다음에 무엇을 해야 할까라는 language description 추가)를 섞어서 학습을 한다.

본 논문은 이 학습 방법을 사용하면 autoregressive (FAST token) 학습을 통해 fast convergence가 가능하면서 action 특화된 representation 학습이 가능해지며 flow matching 학습을 통해 빠른 inference가 가능해진다고 한다.

## Knowledge Insulation & Gradient Flow

이 co-training을 할 때 random initialize된 action expert의 gradient가 VLM까지 back propagate 되면 image encoder와 language model backbone 성능이 저하된다고 한다. 따라서 action expert의 gradient가 backbone에 흘러가지 않도록 stop gradient을 적용한다. 

이 stop gradient가 적용되기 위해서 VLM의 representaton와 Action expert이 상호작용하는 flow matching 과정에서의 attention operation을 수정한다.

일반적인 Attention weight을 구할 때 X를 attention layer의 input이라고 할 때 다음 식을 사용한다.

$$P = \text{softmax}\left(Q(X)K(X)^T + A\right) = \begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix}$$

> 이때 A는 attnetion mask이다.

$P_{bb}$ 는 VLM token이 다른 VLM token을 보는 attention weight, $P_{ab}$ 는 Action expert token이 VLM token을 보는 attention weight, $P_{aa}$ 는 Action expert token끼리 보는 attention weight 이다. 이때 Attention에서 action expert의 token이 query로 작동하기 때문에 VLM token이 action expert token을 보는 weight는 존재하지 않다.

여기서 X를 $X_a$ : Action expert token과 $X_b$ : VLM backbone token 으로 정의하면 stop gradient을 다음처럼 attention 과정에서 표현할 수 있다. 

$$\begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix} = \text{softmax} \left( \begin{pmatrix} Q_b(X_b)K_b(X_b)^T & 0 \\ Q_a(X_a)\text{sg}(K_b(X_b))^T & Q_a(X_a)K_a(X_a)^T \end{pmatrix} + A \right)$$

이때 sg는 stop gradient operator으로 action expert가 VLM token을 사용할 때 이 부분에서 미분 과정에서 생기는 gradient가 VLM embedding에 흘러가지 않도록 해주는 역할을 한다. 

따라서 최종 value embedding은 다음처럼 계산된다.

$$E = \begin{pmatrix} E_b \\ E_a \end{pmatrix} = \begin{pmatrix} P_{bb} V_b(X_b) \\ P_{ab} \text{sg}\left(V_b(X_b)\right) + P_{aa} V_a(X_a) \end{pmatrix}$$

즉, flow matching 과정의 attention을 통해서 생기는 것은 VLM representation의 self attention, Action expert token 간의 self attention, 그리고 stop gradient가 적용된 VLM representation과 action expert token 간의 cross attention이다.

# Results

Knowledge insulation 방법론은 우선 dextrous task에서 정밀하고 빠른 성능을 보여주었다.

Language instruction에 따라 계속 물건을 치워야 하는 table bussing task에 대해서도 높은 성공률을 보여주었다.

Stop gradient만 적용하지 않고 joint training만 했을 때는 language following 능력이 떨어졌지만 (70%) Stop gradient을 적용했을 때는 84% 성공률을 보여주었다.

In distribution task에서와 out of distribution task (unseen object)에서 모두 80%~90%의 성능을 보이면서 stop gradient을 통해 VLM의 generalization 능력을 잃지 않았다고 주장한다.


![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure4.png)

![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure5.png)

![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure6.png)

![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure7.png)

![joowan1108]({{site.url}}/images/papers/knowledgeinsulation/figure9.png)




