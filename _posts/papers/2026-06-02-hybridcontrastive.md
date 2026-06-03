---
layout: single
title: "A Hybrid Contrastive Ordinal Regression Method for Advancing Disease Severity Assessment in Imbalanced Medical Datasets 리뷰"
categories: paper
tag: [Medical, Vision]
author_profile: false
sidebar:
    nav: "counts"
toc: true
toc_sticky: true
toc_label: Table of Contents
use_math: true
---

# Background

Disease grading이란 질병의 severity을 맞추는 task을 의미한다. Disease grading은 조기 진단과 치료 방법 설정에 큰 도움이 되어 큰 주목을 받고있다. 

하지만 생각해보면 특정 질병을 가진 환자들 중에서 severity가 심한 환자들보다 덜한 환자들의 수가 압도적으로 많다. 이런 이유로 실제 disease grading 데이터셋의 class에는 불균형이 존재한다. 또, 질병 진단에서 정확한 severity 판단은 애매하고 근접한 severity 단계끼리 아주 미세한 차이가 존재하는 경우가 많아서 기존 classification model들은 질병의 severity 정보를 고려하면서 학습을 하는 것이 어렵다. 왜냐하면 기존의 classification model들의 loss function은 각 severity 단계를 분리하는데에만 집중하고 이들간의 관계 정보는 고려하지 않기 때문이다.

# Method

이런 문제를 해결하기 위해 본 논문은 disease grading 문제를 ordinal regression 문제로 치환하고 supervised contrastive learning 방법을 도입하여 hybrid supervised contrastive ordinal learning framework을 제안한다. 

> 여기서 ordinal이라는 것은 class 사이에 순서가 존재한다는 것이다. DR dataset의 경우 0(정상)~4(매우 심각함)으로 분류되는데 0 -> 4로 질병이 흘러가기 때문에 순서가 존재한다. 따라서 ordinal classification에서 0인 데이터를 4로 판단하는 것과 0인 데이터를 1로 판단하는 것은 모두 똑같이 틀린 답이지만 전자를 더 심각한 오류로 판단한다. 

Class 불균형이 존재하는 데이터셋에서 ordinal 정보까지 고려할 수 있는 Prototype-based Contrastive Ordinal Loss (PCOL)과 weighted Supervised Contrastive Ordinal Loss ($\text{SCOL}_w$)에 대해 동시에 최적화시키는 framework이다.  

## Preliminaries

이미지 X를 이해하기 위해서는 이미지의 특성 정보를 모델이 이해할 수 있는 embedding / vector으로 바꾸어야 한다. 바꿔주는 함수를 feature mapping function $f$ 라고 할 때 embedding vector $\psi$는 다음과 같이 표현할 수 있다. $\psi = f_{\theta_b}(X), \quad \psi \in \mathbb{R}^{h_0 \times w_0 \times c}.$ 이때 $h_0, w_0, c$는 각각 embedding vector의 크기 정보인 높이, 너비, channel 수이다.

Supervised contrastive ordinal learning이란 질병 severity grading을 위해서 최적의 feature map $\psi$을 찾는 것이다. 