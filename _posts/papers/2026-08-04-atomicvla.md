---
layout: single
title: "AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots 리뷰"
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

VLA는 언어와 이미지 처리, 그리고 action 생성까지 할 수 있지만 long horizon task에 적용하기 어렵고 새로운 skill들을 연속적으로 내재화하는 능력이 부족하다.

이것이 가능하기 위해서는 하나의 model 안에서 high level reasoning과 low level action을 생성해야 하며 scalable continual learning이 가능해야 한다. 

그래서 기존의 연구들은 두 단계로 구성된 구조를 사용하여 pretrained VLM이 high level planner로 작동해서 long horizon task가 주어졌을 때 subtask instructions을 생성하고 별도의 action policy가 이 instructions을 action으로 바꿔주도록 하였다. 

하지만 이런 decoupling은 planner가 생각하는 것과 action policy가 생각하는 것과 다르게 만들어서 오류가 생길 가능성이 높아진다고 한다. 

또, VLA 모델들은 보통 하나의 action decoding module을 사용하기 때문에 추가 skill을 학습하기 위해서는 추가적인 finetuning을 해야 하기 때문에 더 많은 dataset이 필요해진다.또, finetuning을 해도 기존 skill들의 성능이 떨어지는 등의 **catastrophic forgetting**이 발생하기 때문에 skill 종류의 scalability에 문제가 존재한다.

# AtomicVLA

따라서 본 논문은 VLM과 action policy를 decoupling 하지 않고 하나의 framework 안에서 task planning과 action 생성을 하는 방법론을 제시한다.

현재 상태를 바탕으로 thinking module을 실행할 지 acting module을 실행할 지 결정한다.

처음 task을 시작할 때나 하나의 skill을 끝내고 다른 skill로 넘어가기 전에는 thinking module을 실행해서 atomic skill들로 이루어진 chain을 생성한다.

Acting module이 실행될 때는 dynamic하게 skill에 적합한 expert을 선택하여 action을 생성한다. 

또, catastrophic forgetting이 발생하지 않고 skill scalability을 얻기 위해서 continual learning을 적용한다. 

**본 논문의 MoE 구조 요약**

본 논문의 방법론에서 scalability가 가능하도록 **skill library**라는 것을 만든다. Library 안에는 공유되는 generalized expert와 특정 skill에 특화된 experts가 존재하는데 이 experts는 skill encoder을 통해 저장되고 routing encoder을 통해 선택되고 실행된다.

새로운 skill이 생기더라도 동일한 skill encoder를 통해 library에 그대로 저장되면 되고 이 skill에만 특화된 expert와 routing parameter만 추가로 학습하면 되기 때문에 scalability가 보장된다.

> 본 논문에서 각 expert는 $\pi_0$ 을 기반으로 구성된다.

> 기존의 MoE를 사용하는 방법들은 각 expert을 그저 고정된 부품으로 생각을 하고 각 부품들이 정확히 무엇을 잘하는지 이해하는 것이 불가능했다. 하지만 현재 논문은 각 expert을 semantic하게 이해할 수 있는 action primitive으로 정의하였다.

**본 논문의 Continual Learning 방법 요약**

VLA 분야에서는 개별 skill을 학습하는 것이 아니라 generalize 능력을 높이는 방법에 중점을 두고 있다. 그러기 위해서 large scale pretraining에 의존하고 있고 continual learining 관련해서는 연구가 부족하다. 

또, VLA는 여러 action decoding 방법들을 연구하지만 (diffusion, flow matching, etc) 모두 하나의 decoder을 사용한다. 즉, 요즘의 action policy는 특정 task의 정확도에 집중을 하고 여러 task을 잘하도록 하는 scalability에는 연구가 부족하다. 

본 논문은 robotic behavior을 가진 atomic unit들로 확장 가능한 skill expert library을 구축하고 routing module을 통해 scalability을 얻었다.

## Method

![joowan1108]({{site.url}}/images/papers/atomicvla/figure2.png)

### Unified Task Planning and Action Execution

본 논문은 robot policy가 task planning과 action execution을 동시에 할 수 있게 하고 어떤 modality / 무엇을 output 할지 자동으로 결정하게 하는 방법에 대해서 연구한다. 

예를 들어 thinking mode에 들어갔을 때 policy는 여러 camera input $O^{\text{1:n}}_t$ 와 language instruction $l$ 을 받으면 high level task plan [$C_{0-k}, C_t, \sigma$] 을 text으로 출력한다. 반면 acting mode에 들어가면 policy는 가장 최근의 planning output $\sigma$ 와 proprioceptive state $S_t$ 을 바탕으로 actions을 생성한다.

하나의 policy 내부에서 이런 switching이 가능하도록 하기 위해 두 개의 special output tokens "think"와 "act"을 사용한다. $O^{\text{1:n}}_t$ 와 $l$ 이 주어졌을 때 model은 우선 현재가 "think"을 해야 하는 단계인지 "act"을 해야 하는 단계인지 파악한 뒤, "think" token을 출력하면 thinking mode에 들어가서 task chain $C_{0-k}$, 현재까지의 progress $C_t$, 그리고 실행해야 하는 atomic skill abstraction $\sigma$ 을 츨력한다.

> Thinking mode에 들어가는 빈도는 자주가 아니라 task을 시작할 때나 sub skill 간의 전환점에서만 발동된다. 

반면 acting mode에서는 가장 최근의 think step에서 출력한 $\sigma$ 을 바탕으로 low level action chunk $A_t$ 을 생성한다.

### Atomic Skill Abstract Embedding

Skill library의 atomic skill embedding을 생성하기 위해서 diffusion denoising model에서 noise scheduling에서 영감을 얻은 방법을 사용했다고 한다.

각 atomic skill을 $\sigma \in (0,100)$ 에 mapping한 뒤에 high dimensional vector $Z_{\sigma}$ 에 embedding한다. Embedding을 통해서 semantic한 separation을 가능하게 하고 skill specific experts에게 routing이 가능하도록 한다. 

$$
Z_{\sigma} = E(norm(\log(\sigma)))
$$

## Skill Guided Dynamic Routing

본 논문은 $\pi_0$ VLA을 atomic action abstraction guided MoE 구조로 연장한다. 

![joowan1108]({{site.url}}/images/papers/atomicvla/figure2b.png)

이 그림을 보면 skill library 안에는 skill router, pretrained $\pi_0$ 을 보존하는 shared expert, 그리고 multi atomic skill experts가 존재한다. 

학습을 할 때 기존 Expert들의 specialized skills을 유지시키기 위해서 thinking pipeline에서 high level task instruction과 observation값을 통해서 atomic action abstraction을 제공한다. 예를 들어 make coffee라는 high level instruction이 들어왔을 때 이를 바탕으로 바로 expert들을 선택하게 될 때 혼란을 겪을 수 있다. 따라서 thinking pipeline 쪽에서 먼저 atomic action abstraction: {grasp cup, move cup, place cup, press button}으로 분해하고 각 abstraction에 맞는 expert을 선택하게 설계를 하였다. 

더 자세하게는 이 abstraction은 fixed high dimensional embedding $Z_{\sigma} \in \mathbb{R}^d$ 으로 mapping되고 이 embedding을 통해서 skill router는 적절한 expert을 고르는 것이다. K가 expert의 수라고 할 때, skill router가 각 expert에 주는 점수 (probabilirt distribution)은 다음과 같다.

$$
w_k = \text{Router}(Z_{\sigma}), \qquad k \in \{1,2,\ldots,K\}
$$

이 다음 제일 점수가 높은 expert만 선택되고 최종 action chunk $A_t$는 weighted combination으로 예측된다. 

$$
F_{\text{out}} = (1-w_k) \cdot F_{\text{share}}(x_t) + w_k \cdot F_k(x_t)
$$

> 이때 $x_t$는 현재 multimodal input [$O^{\text{1:n}}_t, l, s_t$] 이다.

이런 weighted combination을 통해 $\pi_0$ 의 generalization 능력과 expert의 specific 능력을 합친 것이다. 

## Continual Learning with Skill Expansion

새로운 skill을 배워야 할 때 이 skill을 새로 추가하려고 하면 보통 기존의 능력을 잃는 catastrophic forgetting 방법이 발생할 수 있다. 

따라서 본 논문은 modular skill expert 원리를 사용한다. 각 atomic skill이 embedding vector로 변한다고 했는데 새로운 skill이 생겼을 때 이 skill의 vector까지 router가 인식할 수 있도록 확장하고 skill에 해당되는 expert module만 기존 구조에 붙이면 된다는 것이다. 

더 완만하게 적용시키기 위해서 router가 확장할 때 기존 skill들에 대한 것은 그대로 두고 새로운 skill으로 routing하는 parameter에는 작은 random weight을 부여하였다. 이 방법을 적용하면 최소한의 fine-tuning을 통해 skill set을 넓힐 수 있고 기존 skill 능력도 유지시킬 수 있다. 

코드를 보면 이 skill expert는 attention layer 다음에 존재하는 SwiGLU FFN이다. 각 attention layer 사이에는 SwiGLU FFN이 존재하며 각 skill마다 자기의 SwiGLU FFN이 지정되어 있는 것이다. Attention을 통해서 contextualized 정보가 SwiGLU FFN에 들어가게 되면 SwiGLU FFN은 이 정보를 자기의 skill에 적합한 특징으로 가공한다. 

x을 hiddens tate 벡터, $W_1$ 은 gate을 계산하기 위한 projection, $W_3$ 은 value을 계산하기 위한 projection, $W_2$ 는 차원을 맞춰주는 projection이다. 

$$
\operatorname{SwiGLU}(x) = W_2(\operatorname{SiLU}(W_1x) \odot W_3x)
$$

여기서 gate을 계산한다는 것은 특정 feature을 얼마나 강하게 사용할 지 그리고 value을 계산한다는 것은 어떤 feature를 사용할 것인지를 결정하는 것이라고 생각하면 된다. 

그리고 skill을 추가한다는 것은 SwiGLU FFN을 추가하는 것이다. 학습할 때는 attention은 그대로고 SwiGLU FFN만 attention output을 자기 skill에 맞게 처리하는 방법을 배우는 것이라고 생각하면 된다.

## Task Planning Embodied Data Generation

Atomic action들을 정확하게 label하기 위해서 Principal Axis Analysis 방법을 사용한 trajectory based atomic decomposition 방법을 제안한다. 

Pick up the mug and place it on the table. 와 같은 demonstration이 있다고 할 때 이 demonstration을 atomic skill들로 나누기 위해서는 어떤 구간이 pick인지 어떤 구간이 place인지 알아야 한다. 기존의 방법들은 VLM의 video understanding 능력에 의존하였지만 본 논문은 end-effector의 이동 경로에서 key kinematic dimension을 분석해서 atomic action들을 분리하였다. 

> Key Kinematic이란 좌표 변화, 회전 변화, gripper 상태 등을 말한다

이렇게 key kinematic으로 segmentation을 한다면 각 segment의 semantic meaning을 얻을 수 있다. 즉, 각 segment의 주된 motion 정보를 얻을 수 있다. 예를 들어 z 좌표가 집중적으로 감소하면서 gripper가 닫힌다면, pick action 중이라는 것을 알 수 있다. 

이렇게 physics 기반 decomposition을 하면 긴 trajectory를 semantic하면서 정확한 atomic actions으로 나눌 수 있게 된다.
