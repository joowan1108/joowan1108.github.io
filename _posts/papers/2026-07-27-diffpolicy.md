---
layout: single
title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion 리뷰"
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

Demonstration을 통한 policy 학습은 observation으로부터 action의 mapping을 학습하는 supervised regression task이라고 볼 수 있다. 하지만 로봇 action에서는 multimodal, 순차적 관련성, 그리고 높은 정밀도가 요구되기 때문에 일반적인 supervised learning task보다는 훨씬 복잡하다. 

본 논문은 robot visuomotor policy을 conditional denoising diffusion process on robot action space라고 정의한다. 

# Diffusion Policy Formulation

Visuomotor robot policy를 denoising probabilistic model (DDPM) 문제로 정의하였다. 

Diffusion policy는 복잡한 multimodal-action 분포를 표현할 수 있고 조정해야 하는 hyperparameter가 적기 때문에 안정된 학습을 할 수 있다.

## Denoising Diffusion Probabilistic Models

DDPM은 generative model으로 generate 과정이 denoising process으로 표현되어있다. 이 과정을 Stochastic Langevin Dynamics이라고 한다.

**추론 과정**

Gaussian noise $x^k$에서 DDPM은 $k$번의 denoising iteration을 통해서 한 series의 intermediate actions $x^k, x^{k-1}, \ldots, x^0$을 생성한다. 각 iteration마다 noise를 줄이는 방향으로 action이 생성되면서 noise가 없는 output $x^0$이 될 때까지 반복된다.

한 iteration 과정을 식으로 표현하면

$$
x^{k-1} = \alpha \left(x^k - \gamma \epsilon_{\theta}(x^k, k) + \mathcal{N}(0, \sigma^2 I)\right)
$$

이때 $\epsilon_{\theta}$가 noise를 예측하는 network이고 $\mathcal{N}(0, \sigma^2 I)$가 매 iteration마다 추가되는 Gaussian noise이다.

각 iteration에서의 noise을 제거하는 과정을 다음처럼 볼 수 있다.

$$
x' = x - \gamma \nabla E(x) 
$$

$\nabla E(X)$는 $\epsilon$ 이 예측하는 gradient field이다. 즉, $\epsilon$ 이 예측하는 noise의 방향으로 조금씩 이동하면서 noise을 제거하여 더 뚜렷한 action $x'$ 으로 만드는 것이다.

$\alpha, \gamma, \sigma$ 는 noise schedule으로 gradient descent 과정에서 learning rate scheduling이라고 생각하면 된다.

**학습 과정**

DDPM 학습을 시작할 때는 정답 action $x^0$ 인 sample이 있고 random noise $\epsilon^k$ 가 추가되었을 때 noise prediction network $\epsilon_{\theta}$ 는 이 추가된 noise을 예측하도록 학습이 된다.

$$
\mathcal{L} = \operatorname{MSE}\left(\epsilon^k, \epsilon_{\theta}(x^0 + \epsilon^k, k)\right)
$$

이 MSE objective을 최소화하는 과정은 정답 행동 분포 $p(x^0)$ 과 DDPM이 생성하는 행동 분포 $q(x^0)$ 가 비슷하게 하는 KL-divergence 최소화 효과와 동일하다.

> Variational Lower Bound(ELBO) 최대화하는 과정이 결국 MSE처럼 된다는 연구 결과가 존재한다.

> 즉, noise을 예측하도록 하는 것이 결국은 정답 action 생성 분포를 올바르게 학습하는 것과 동일한 objective하는 것이다.

## Diffusion for Visuomotor Policy Learning

DDPM은 보통 image 생성에 사용되지만 로봇 행동 예측에서 생성하기 위해서는 추가적인 modification이 요구된다.

1. Output $x$를 robot action이 되도록 하기

$\rightarrow$ **Closed loop Action sequence prediction**

좋은 action을 생성한다는 것은 long horizon task에서 일관적이고 예상하지 못한 상황에서도 잘 반응을 하는 action을 생성하는 것이다. 

그러기 위해서 diffusion model으로 예측한 action sequence을 일정 시간동안 실행하면서 replanning을 하는 방법을 사용한다.

Time step t에서 policy가 최근 $T_O$ steps의 Observation $O_t$을 input으로 갖게 하고 $T_p$ steps의 action을 예측한다. 이때, $T_a$ 개의 steps은 replanning을 하기 전에 실행되는 steps 수이다.
> $T_O$는 observation horizon, $T_p$는 action prediction horizon, $T_a$는 action execution horizon이라고 부른다.

이렇게 해야 action들이 일관성을 가지면서 유연성(responsive)을 갖게 된다.


2. Denoising 과정을 input observation $O_t$을 기반으로 만들기

$\rightarrow$ **Visual Observation Conditioning**

이전 논문은 joint distribution $p(A_t, O_t)$ 으로 근사하도록 했지만 이렇게 하면 미래 observation과 미래 actions을 모두 생성해야 하기에 inference cost가 높고 정확성도 떨어진다.

따라서 DDPM을 joint distribution $p(A_t, O_t)$ 로 하지 않고 conditional distribution $p(A_t \mid O_t)$ 을 근사하도록 설계하였다.

Diffusion model에서 denoising step을 다음과 같이 정의했었는데 

$$
x^{k-1} = \alpha \left(x^k - \gamma \epsilon_{\theta}(x^k, k) + \mathcal{N}(0, \sigma^2 I)\right)
$$

conditional distribution $p(A_t \mid O_t)$ 이 capture되도록 다음과 같이 바꾼다.

$$
A_t^{k-1} = \alpha \left(A_t^k - \gamma \epsilon_{\theta}(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\right)
$$

Training loss도 다음처럼 바뀐다.

$$
\mathcal{L} = \operatorname{MSE}\left(\epsilon^k, \epsilon_{\theta}(O_t, A_t^0 + \epsilon^k, k)\right)
$$

> Output을 생성할 때 observation features $O_t$을 없애는 것이 denoising 과정을 더 빠르게 만들고 실시간 반응도 더 쉽게 해주며 end to end training도 가능하게 만들어준다.

## Network Architecture Options

Policy $\epsilon_{\theta}$ 을 CNN으로 정의하는 것이 좋은지 Transformer로 정의하는 것이 좋은지 알아내기 위해 둘 다 사용해서 성능을 비교한다.

우선 action을 생성할 때는 observation 값들로만 condition한다. 그리고 목표 상태 또한 condition에 넣지 않고 receding prediction horizon 개념을 사용하여 observation 기반으로 action을 생성하는 것을 반복한다.

실험 결과 CNN의 성능이 Transformer보다 좋았다. 하지만 CNN의 temporal convolution layer는 smooth한 sequence을 선호하는 경향이 존재하기 때문에 정답 action sequence가 빠르게 변하는 actions일 때는 성능이 저하되었다.

> Temporal Convolution은 주변 시점의 정보를 평균내면서 (local averaging) 특징을 추출하기 때문에 급격하게 변하는 신호보다는 부드럽게 변하는 (low frequency) 신호를 더 선호하는 bias을 가진다.

**Time Series Diffusion Transformer**

CNN의 과도한 smoothing을 줄이기 위해서 transformer 구조를 채용하였다.

Noise $A_t^k$ 가 든 actions는 token으로 변환되어 transformer decoder의 입력으로 들어간다. 그리고 각 action token의 순서 정보를 알려주기 위해 sinusoidal token들이 prepend 되어 들어간다.

Observation $O_t$ 또한 MLP을 통해 embedding되어 decoder에 들어간다. 

이렇게 attention을 통해서 observation embedding과 action token들이 서로 attend하고 새로운 action sequence (noise gradient)을 출력할 수 있게 되는 것이다.

## Visual Encoder

Vision Encoder는 이미지를 latent observation embedding $O_t$ 으로 바꾸는 역할을 하고 diffusion policy와 같이 학습이 된다. 다른 시점의 camera들은 서로 다른 encoder에 들어가고 embedding들이 concat 되어 $O_t$ 를 생성하게 되는 것이다.

본 논문은 visual encoder가 공간 정보를 더 잘 이해하게 하도록 global average pooling을 spatial softmax pooling으로 바꾸었다. 

## Noise Schedule

Noise $\sigma$, $\alpha$, $\gamma$ 을 각 timestep에 추가해서 denoising을 하는 방향으로 실제 action을 예측하는 것이 diffusion network라고 하였다. 이때 얼만큼의, 어떤 종류의 noise을 넣느냐에 따라 실제 action sequence의 frequency 정보의 뚜렷함 (세부적인 변화 정보)이 달라진다.  

따라서 noise을 작은 값부터 넣어서 천천히 늘리면 high frequency action도 잘 학습할 수 있게 되지만 처음부터 큰 값을 넣으면 frequency 정보가 가려져서 high frequency action의 학습에 어려움을 겪을 수 있다.

본 논문은 그래서 square cosine schedule을 사용한다.

## Acceleration Inference for Real-time Control

Closed-loop real-time control에서 fast inference speed는 매우 중요하다. 따라서 inference 때 iteration 횟수를 학습할 때보다 훨씬 작게 한다. $100 \rightarrow 10$


## Model Multi-Modal Action Distributions

Behavior cloning에서 사람의 demonstration으로부터 multi-modal action distribution을 학습하는 것이 어렵다.

Multi-modal action distribution이란 정답이 하나가 아닐 때를 의미한다고 보면 된다. 

Behavior cloning은 보통 human demonstration의 평균을 학습하는 경향이 존재한다. 따라서 왼쪽으로 가는 demonstration과 오른쪽으로 가는 demonstration이 존재한다면 평균을 내서 정면으로 가는 행동을 학습하게 되는 것이다. 

이 그림은 다양한 모델에서 action trajectory을 시각화 한 것이다.

![joowan1108]({{site.url}}/images/papers/diffpolicy/figure3.PNG)

이 그림을 보면 diffusion policy는 왼쪽, 오른쪽 두 경로를 even하게 사용하는 반면 다른 모델들은 한쪽으로 편향이 되어있거나 중간을 가로지르는 행동을 보이곤 한다.

Diffusion policy는 두 이유로 인해서 이런 문제점이 생기지 않는다.

1. Stochastic sampling procedure

학습을 할 때 매 denoising 과정에서 추가되는 gaussian noise가 human demonstration을 조금씩 다르게 해서 다양한 trajectory들을 학습할 수 있게 한다.

2. Stochastic Initialization

또 같은 observation이더라도 학습을 시작할 때 사용하는 noise 값이 매번 다르기 때문에 여러 개의 서로 다른 성공적인 행동들을 배울 수 있는 것이다.


## Synergy with Position Control

Action의 종류는 Position control 기반 action과 velocity control 기반 action으로 나뉜다. Position 기반으로 action을 표현하면 좌표 A에서 좌표 B로 가는 action은 다양하다. 반대로 velocity 기반으로 action을 표현하면 비교적 제한적인 다양성을 가진다. 

Position 기반은 너무 다양하기 때문에 behavior cloning을 적용했을 때 불안정한 문제를 가져서 velocity control 기반 action을 선호했었다.

하지만 diffusion policy는 이미 noise을 통해서 다양한 trajectory들을 잘 다루기 때문에 (action multimodality를 더 잘 표현하기 때문에) 오히려 diffusion policy에서는 position control이 더 유리하다.

또, Position control은 trajectory가 다양하더라도 compounding error effect (오차 누적)에 강하다. Velocity을 기반으로 하는 action은 적분을 통해서 다음 위치를 계산한다. 그래서 이처럼 action sequence을 예측해야 하는 task에서 초기에 오차가 생겨버린다면 그 이후 시점들의 action에도 오차가 누적이 되는 문제를 가지고 있다. 하지만 position control은 좌표값을 사용하기 때문에 오차 누적 정도가 훨씬 적다. 

추가로, action sequence를 한 번에 다 출력해야 하는 robotics에서는 position control이 더 유리할 수밖에 없다. Position control 기반 action sequence가 $[a_1, a_2, a_3, \ldots, a_t]$라면 각 action $a_i$는 각각 자기만의 position 목표를 지니고 있다. 하지만 velocity 기반 action sequence에서 $a_i$는 이전 위치에서 얼마나 더 움직여야 하는지에 대한 값이고 $a_i$가 $a_{i-1}$의 적분값을 통해서 만들어지기 때문에 초반에 예측이 실패하면 sequence 내에서 다른 action 값에도 영향을 준다.

즉, diffusion policy는 position control의 단점에 영향을 받지 않으면서 장점은 다 활용할 수 있기 때문에 유리하다.

![joowan1108]({{site.url}}/images/papers/diffpolicy/figure4.PNG)


## Benefits of Action-Sequence Prediction

Action sequence을 한번에 다 출력하는 것은 high dimension space에서 sampling을 하는 것이기 때문에 기존 방법들이 잘 하지 못했다. 

> action sequence도 high-dimensional space를 가진다. 하나의 action 크기가 $\lvert A \rvert$이고 sequence 길이가 $L$이라면 총 dimension은 $\lvert A \rvert \times L$이다.

하지만 DDPM은 이미 이미지 생성에서 좋은 성능을 보여주었기 때문에 본 논문은 diffusion policy 또한 high-dimensional space에서 scale을 잘할 것이라고 주장한다. 즉, diffusion policy에서는 다른 방법들과 달리 action sequence 전체를 예측하는 것이 유리하다.


또, action sequence 전체를 한번에 예측하지 않으면 생기는 문제들도 존재한다.

1. Temporal Action Consistency

    각 action이 매번 독립적으로 예측이 된다면 왔다갔다 하는 jittery actions가 생길 가능성이 높다. 하지만 전체 흐름 (sequence)을 예측하도록 하면 trajectory가 일관성을 갖게 되어서 더 좋은 성능을 보일 것이다. 

2. Robustness to idle actions

    Idle action이란 물을 부을 때와 같은 demonstration에서 멈춰있어야 하는 행동을 의미한다. 멈춰있기 때문에 어쩔 수 없이 action 값이 일정 시간동안 똑같이 유지된다. Action을 하나씩만 예측하고 학습하게 하면 idle action에서는 동일한 값이 나오기 때문에 (데이터셋에 동일한 값이 여러번 나오기 때문에) 예측 분포가 overfitting 되는 문제가 생길 수 있다. 하지만 action sequence 전체를 예측하고 학습하게 하면 이런 문제가 생기는 경우가 줄어든다.

## Training Stability

다음은 IBC (Implicit Behavior Cloning)의 학습 과정이다.

![joowan1108]({{site.url}}/images/papers/diffpolicy/figure6.PNG)

IBC의 training error가 왔다갔다 한다는 점과 evaluation 결과도 들쭉날쭉해서 hyperparameter tuning이 어렵다. 

IBC의 학습이 불안정한 이유는 Energy Based Model으로 action 분포 $p_{\theta} (a \mid o)$ 를 표현하기 때문이다.

$$
p_{\theta}(a \mid o) = \frac{e^{-E_{\theta}(o,a)}}{Z(o,\theta)}
$$

이때 $Z(o,\theta)$ 는 normalization constant으로 모든 action에 대한 $e^{-E_{\theta} (o,a)}$ 을 적분해야 구할 수 있다. 모든 action을 알지 못하기 때문에 IBC는 InfoNCE으로 학습 objective을 대체한다. ( Z 값을 $e^{-E_{\theta} (o,a)} + \sum_{j=1}^{N_{\mathrm{neg}}} e^{-E_{\theta}(o,\tilde{a}_j)}$ 으로 대체)


$$
L_{\mathrm{InfoNCE}}
=
-\log\left(
\frac{
e^{-E_{\theta}(o,a)}
}{
e^{-E_{\theta}(o,a)}
+
\sum_{j=1}^{N_{\mathrm{neg}}}
e^{-E_{\theta}(o,\tilde{a}_j)}
}
\right)
$$

이전 연구 결과에 따르면 EBM의 학습 불안정성이 $Z$를 대체하기 위한 negative sampling의 불정확성에서 나온다고 한다. 즉, $Z$를 대체하려고 하기 때문에 불안정성이 생기는 것이다.

하지만 Diffusion policy와 DDPM은 Z을 무시할 수 있다. 다음 식에

$$
p_{\theta}(a \mid o) = \frac{e^{-E_{\theta}(o,a)}}{Z(o,\theta)}
$$

log을 씌우면

$$
\log p_{\theta}(a \mid o) = -E_{\theta}(o,a) - \log Z(o,\theta)
$$

이 식을 action $a$에 대해서 미분하면 $Z$는 $o$와 $\theta$에만 대한 식이기 때문에 0이 된다.

$$
\nabla_a \log p(a \mid o) = -\nabla_a E_{\theta}(a,o) - \nabla_a \log Z(o,\theta) = -\nabla_a E_{\theta}(a,o)
$$

이때 
$$
\nabla_a \log p(a \mid o) = -\epsilon_{\theta}(a,o)
$$

이기 때문에 diffusion policy와 DDPM에서는 $Z$ 값을 알지 못해서 생기는 문제가 생기지 않아 안정적인 학습이 가능하다.
