---
layout: single
title:
categories: SKKU_SwEng
tag:
author_profile: false
sidebar:
  nav: counts
toc: true
toc_sticky: true
toc_label: Table of Contents
use_math: true
---
# Software Process

- **Specification**
	- Define functionality and constraints
- **Developement**
	- Produce software
- **Validation**
	- Checking the requirements 
	- Unit testing (False positive 문제 있음), System testing, Customer testing
- **Evolution**
	- Evolving

Software Process을 정의하기 위해서는 세 가지를 정의해야 한다.

- **Product & Deliverables** : outcomes of activity
- **Roles**: Responsiblities
- **Pre and Post conditions**: Conditions that must hold before and after

Pre condition으로 input specification이 정의된 것이 architectural design으로 들어가고 post condition으로 정의된 expected output이 나와야 한다.

Software process으로는 두 종류가 있다.

- **In plan-driven processes**
	- 모든게 먼저 계획되어 있어야 함
- **In agiles processes**
	- 계획이 incremental, continual하게 된다.

Large system은 이 두 process의 중간점을 가져야 한다.

# Software Process Model

이런 software process 방법을 표현하기 위해 software process model을 사용한다.

- Waterfall model
	- Scratch
	- Process activities as separate process phases
	- for plan driven
- Incremental Development
	- Scratch
	- Interleaving process activities
	- for agile, series of versions
- Integration and Configuration
	- Reusable software components

최근 large system들은 모든 모델의 elements을 조금씩 사용함

## Waterfall Model

Planning all of process activies before starting

Direct하게 development activities reflect함

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure1.png)

Critical system / Embedded system에 적합

빠르게 변하는 상황에서는 별로임

## Incremental Development

Interleaved

Initial implementation을 한 다음, feedback을 받고, evolving하는 과정을 반복한다

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure2.png)

**장점**
- Cost of implementing changes is reduced
	- Waterfall model은 documentation까지도 바꿔야 하기 때문
- Customer feedback을 받기 쉽다
- Deploy가 빠르다

**단점**
- Process is not visible
	- 개발이 빠르게 진행되기 때문에 매 version마다 documents을 만드는게 어렵다. 따라서 progress을 알기 어렵다
- Degrade 발생 가능
	- Messy code, 새로운 feature을 더 추가하는 것은 어려움
	- 따라서 refactor을 자주 해야함

## Integration and Configuration

이미 존재하는 reusable components을 재사용해서 integrating

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure3.png)

이 model에 중요한 key process들은 다음과 같다.

- Requirements Specification
	- 간단한 description을 포함
- Software Discovery and Evaluation
	- 존재하는 components 찾고 검증하는 과정
- Requirements Refinement
	- 찾은 components을 바탕으로 requirements refining
- Application System Configuration
	- COTS을 변경
	- COTS 가능할 때 (Commercial Off The Shelf application 있을 때)
- Component Adaptation and Integration
	- Components을 변경
	- COTS 없을 때

**장점**
- Software develop에 드는 cost 줄이기
- Delivery 빠르게 하기

**단점**
- 기존 소프트웨어에 맞는 요구사항을 타협해야 하기에 user의 실제 요구를 완벽하게 만족 못 할 수 있음
- Control over system evolution is lost : 사용한 reusable components을 조정할 수 없기에



# Software Process Activities

Specification, development, validation, evolution

이 activities는 Software process model마다 다르다.

## 1. Software Specification

== Requirements Engineering

Service의 requirements을 이해하고 정의하는 과정
Contraints을 정의하는 과정

- **Requirements elicitation and analysis**
	- Observation과 discussions을 통해 requirements을 알아내기
- **Requirements specification**
	- 얻은 정보를 document으로 정리하기
- **Requirements validation**
	- Requirements가 realism (실현 가능), consistency (모순적이지 않은지), completeness (빠짐없이 정의되어 있는지)을 만족하는지 확인

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure4.png)

## 2. Software Development

- **Architectural Design**
	- 전체 구조, 주요 components, relationships 정의
- **Database Design**
	- Datastructure 정의
- **Interface Design**
	- System component 간 interface 정의
- **Compoenent Selection and Design**
	- Resuable component 찾지 및 새로운 components 정의

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure5.png)

> Testing은 bug을 찾아내는거고 debugging은 bug가 어디있는지 찾고 수정하는 것

## 3. Software Validation

System이 specification과 requirements을 만족하는지 확인하는 과정

모든 단계에서 수행된다

Requirements 확인보다 프로그램 testing에 더 큰 노력을 한다.

Testing에는 여러 state가 존재한다.

- **Component Testing**
	- Testing individual components
- **System Testing**
	- Entire system
- **Customer Testing**
	- Test data 말고 실제 customer가 test

**V-model**은 plan driven software process을 위한 testing model이다. 각 단계마다 미리 어떻게 test을 할 것인지 미리 정의한다. Link between testing and development activities


## 4. Software Evolution

Software에서 development와 maintenance을 분리하는 것은 점점 모호해진다.

Software에서 change는 cost을 유발하는데 이 cost을 최소화하는 것은 두 방법이 있다.

- Change Anticipation
	- 변경사항을 예측하고 구현
- Change Tolerance
	- 쉽게 수정될 수 있도록 설계

Change에 유연해지도록 하는 두 방법

- **Prototyping**
	- Early version of software to demonstrate concepts
	- 장점
		- Potential users가 사용할 수 있게 해서 system이 자기 requirements을 만족하는지 보기 가능
		- Errors와 omissions을 파악하기 가능
		- 변한 것을 바로 반영 가능
		- User interface을 개발하기 위한 유일한 방법
	- 단점
		- Final system과 prototype은 다름
		- Prototype tester는 실제 user와 다름

	- Prototype을 개발할 때는 중요한 점
		- 필요한 것만 남기기
		- Non functional requirements는 고려하지 않기
		- Error handling은 고려하지 않기



- **Incremental Delivery**
	- 중간에 완성된 기능들을 보고하기
	- 이때 각 increment가 실제 가치를 지녀야함
	- 장점
		- 중간에 추가되는 기능들을 통해 prototype을 또 구성할 수 있고 더 testing 가능
		- User가 기다리지 않아도 된다
		- 가장 중요한 system이 가장 많은 testing을 받는다
	- 단점
		- 새로운 system으로 바꿀 때 user는 이전 system의 기능까지도 필요로 한다.
			- 기존 system을 사용하던 user는 모든 기능이 가능하길 원하지만 incremental delivery는 그게 불가능함
		- Requirements가 자세히 defined 되지 않았기 때문에 increment가 공통적으로 필요로 하는 설계가 나중에 완성된다.


> Process Improvement: 기존 process을 바꿔서 효율을 높이는 것
> 방법 1: Process maturity approach : Process와 project management 향상
> 방법 2: Agile Approach : 반복적인 development와 개발 overhead 줄이기