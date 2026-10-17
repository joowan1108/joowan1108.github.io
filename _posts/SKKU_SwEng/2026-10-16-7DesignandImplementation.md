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
설계를 실제 implementation으로 만드는 방법


# SW Design and Implementation

Executable SW system이 develop되는 단계이다

- SW design: Identy SW components and their relationships based on customer's requirements
- Implementation: Realizing the design as a program

**중요한 implementation decision**

Implementation에서 중요한 것은 직접 개발할 것인가 아니면 application software을 구매할 것인가이다.

구매를 하게 된다면 이 software을 어떻게 우리 요구사항에 맞게 바꿀 것인가가 중요하다


# System 설계 전체 절차

**System context and interactions**

Sw design의 첫 단계이다
- SW와 external environment 간의 상호작용을 배우는 것

System Context Model
- Other system in environments을 보여줌

Interaction model
- System이 environment와 상호작용하는 법을 보여줌

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure35.png)

Weather station이라는 system을 생각했을 때 외부 environment가 이 system을 어떻게 사용하는지를 보여주는게 use case이다. 따라서 외부 environment가 primary actor가 되고 weather station이 secondary actor가 된다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure36.png)


**Architectural design**

Major components을 안아내고 이들 간의 interaction을 찾아내는 것

Architectural pattern을 통해 system organization design하기

Variant of **Client and Server** design으로 Weather station을 구성할 수 있다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure37.png)

**Object class identification**

System interaction을 encapsulate하는 high level system objects을 파악

이들을 파악하는 방법으로는 세 가지가 존재한다

- **Grammatical analysis**
	- Grammatical analysis of natural language description 사용

- **Tangible Entities**
	- 구체적 실체를 사용
	- domain -> aircraft
	- role -> manager
	- location -> office

- **Scenario based analysis**

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure38.png)

여기서 찾은 object들을 더 refine 할 수 있다. (instrument을 하나로 묶음)

Weather Station Object Class
- Weather station의 interface을 정의
- usecase으로부터 함수들이 정의되었음

Weather Data object class
- Report weather command을 처리

Ground thermometer, Anemometer, and Barometer object classes
- tangible hardware entities

**Design methods**

System requirement와 implementation의 bridge 역할을 한다. 

Object / Object class을 보여주고 이들간의 association / relationship을 보여준다

이때 무슨 design model을 사용할 것인지 잘 선택해야 하며 design process에 따라 detail이 달라진다.

> Agile에서는 whiteboard면 충분

그리고 model의 수를 최소화해서 cost reducing을 해야 한다.

Design model에는 두 종류가 있다.

- **Structural Models**
	- Static structure을 보여줌
	- ex) class diagram
- **Dynamic Models**
	- Dynamic structure와 expected runtime interactions을 보여준다
	- sequence diagram, state diagram

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure39.png)

- 이때 점선 화살표는 asynchronous : sender가 기다리지 않음
- 중간 직사각형의 크기가 클수록 active하다

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure40.png)



**Interface specification**

- Interface의 detail을 명시
- Service의 semantics을 정의
- 하지만 data representation의 detail은 정의하지 않는다
- Access / update data operations까지만 정의

Interface는 class diagram과 동일한 UML으로 정의될 수 있다
- 이때 attribute section은 신경쓰지 않고 연산만 신경쓴다
- stereotype가 name part에 있어야 한다 (interface)
- 1 대 1 대응 아니여도 괜찮다


# Design Pattern

Design pattern은 이미 accumulated 된 지식이고 well tried solution이며 reusing knowledge가 장점이다.

이때 patterns는 object oriented design에 적합하다.

Design pattern의 elements는 다음과 같다.

- Name
	-  reference
- Problem Description
	- 어디에 적용되는지
- Solution Description
	-  Design 설명. relationships과 responsibilities
- Statement of Consequences
	-  어떤 문제 / 단점이 있는지

**Observer Pattern**

Single object에 대해 다양한 presentations 필요할 때 사용한다

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure42.png)
![joowan1108]({{site.url}}/images/SKKU_SwEng/figure43.png)
![joowan1108]({{site.url}}/images/SKKU_SwEng/figure44.png)

Need to recognize that your design problem has an associated pattern that can be applied.
- Tell several objects that the state of some other object has changed (Observer pattern).
- Tidy up the interfaces to a number of related objects that have often been developed incrementally (Façade pattern).
- Allow for the possibility of extending the functionality of an existing class at
runtime (Decorator pattern).

# Implementation Issues

Important aspects는 다음과 같다

- Reuse
- Configuration Management
	- 다양한 SW versions들을 관리
- Host target development
	- Host system: 하나의 computer에서 SW 구현
	- Target system: 여러 computer에서 실행
	- 즉, 실행 환경을 고려해야 함

**Reuse**

Reuse의 단위가 다를 수 있다

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure41.png)

- Abstraction Level
	- Knowledge of successful abstractions 사용
	- SW을 직접 reuse하는 것 아님
- Object Level
	- Library 잘 찾기
- Component Level
	- Components을 직접 재사용
- System Level
	- 전체 application system을 재사용

Reuse을 함으로써 얻는 cost는 얼마인가

- Reuse할 거 찾는 cost
- Reusable SW 구매하는 cost
- Reusable SW을 요구사항에 맞게 configuring cost
- Reusable SW을 integrating 하는 cost


**Configuration Management**

Managing a changing software system

**Goal**
- Supporting the system integration process so that all developers can access the project code in a controlled way. (e.g., conflict in git)
- Ensuring that everyone can access the most up-to-date versions of SW.
- Finding out what changes have been made.

**4 fundamental configuration management activities**

- Version management
	- 다양한 version 관리
	- 다양한 programmer들의 development을 coordinate
	- Overwriting code 멈추기

- System Integration
	- Component의 각 버전들이 어떻게 사용되었는지 알려주기

- Problem Tracking
	- Report bugs and problems
	- 누가 고치고 있고 언제 고쳐지는지 알기

- Release management
	- 새 release의 functionality 계획하고 software 배포를 위해 정리


**Host target development**


# Open-source development

Source code가 공개되고 참여자들이 이 개발에 참여하는 것

cost가 싸고 widely used open source들은 매우 reliable하다

하지만 실제 기업에서 Opensource을 사용할 때의 문제점은 다음과 같다.

- 우리 회사의 제품을 개발할 때 기존 오픈소스 컴포넌트를 사용할 것인가?
- 우리 회사가 개발하는 소프트웨어 자체를 오픈소스 방식으로 개발할 것인가?

## Open source Licensing

Open source는 free하게 사용 가능하지만 code을 맘대로 다룰 수 있다는 것은 아니다. 

Open source을 올릴 때 license에 따라 restrictions가 다르다

- The GNU General Public License (GPL)
	- reciprocal
	- 이걸 사용하면 사용한 것도 open source 해야 한다
- The GNU Lesser General Public License (LGPL)
	- Variant of GPL
	- 사용할 때 사용한 것은 open source 하지 않아도 되지만 license된 component을 수정했다면 open source 해야함
- The Berkley Standard Distribution (BSD) License
	- 아무것도 안 해도 된다