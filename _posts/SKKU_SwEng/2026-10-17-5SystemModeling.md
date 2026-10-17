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

# System Modeling

System의 abstract model을 개발하는 방법

각 model을 각자 다른 perspective을 보여주고 diagram type을 UML을 통해 보여준다.

이미 존재하는 system과 새로운 system을 modeling할 때는 다른 정보를 담아야 한다

- Models of existing system
	- Existing system이 하는 것
	- 장점과 단점
- Models of new system
	- Proposed requirements 설명하기
	- Design proposal 설명하기
	- 구현하기 위한 documentation

Graphical model을 사용하는 이유
- Existing, proposed system에 집중할 때
	- Model은 incomplete + informal notation 가능
- Documenting existing system
	- Model은 incomplete 가능 하지만 correct 해야함
- Generating system implementation
	- Model은 complete 해야 하고 correct 해야함


System modeling을 하기 위해서 사용되는 언어가 UML: Unified Modeling Language이다. 이 langauge는 object oriented modeling을 위한 standard langauge이다.

종류로는 5가지가 존재한다.

- **Use Case Diagram**
	- Iteractions between system and environment
- **Sequence Diagram**
	- Interactions between actors and system + between system components
- **Class Diagram**
	- Object classes in the system and associations
- **Activity Diagram**
	- Activities involved in process
- **State Diagram**
	- How system reacts to internal / external events



System modeling의 perspective은 4가지가 존재한다.

- **External perspective**
	- Context와 environment을 modeling
- **Interaction perspective**
	- System과 environment의 interaction modeling
- **Structural perspective**
	- System의 organization과 data의 structure
- **Behavioral perspective**
	- Dynamic behavior modeling


## External Perspective

External perspective을 보여주는 system model은 context model이다. System이 다른 system들이 존재하는 environment에서 어떤 위치에 있는지 알려준다.

이 중 Simple Context model이 있다.

**Simple Context Model**

Relationship type을 보여주지는 않는다. 따라서 다음 정보는 없다

- System이 생성하고 소비하는 data
- Physical location


![joowan1108]({{site.url}}/images/SKKU_SwEng/figure10.png)

하지만 relation 정보는 system의 requirements와 design에 영향을 주기 때문에 단독으로 사용되지는 않는다

예를 들어 activity model과 같이 사용되면 다음처럼 된다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure11.png)

## Interaction Perspective

Interaction modeling을 하기 위해서 보통 use case 또는 sequence diagram을 사용한다.

Use case는 system과 external agents(human users or other systems) 간의 interaction

Sequence diagram은 system components 간의 interaction. 이때 external agents도 포함될 수 있다.

**Use Case Diagram**

User가 interaction을 통해 무엇을 expect하는지 설명하는 것이라고 보면 된다.

Use case diagram은 actor, use case, system, relationship으로 구성된다.

- **Actor** 
	- **Primary actors**: 자기의 목적을 달성하기 위해 system을 사용하는 사람 (고객)
	- **Secondary actors**: Primary actor의 목표를 달성할 수 있도록 시스템에 서비스를 제공하는 actor (결제 시스템)

- **Relationship**
	- **Association**: Actor가 use case에 참여 / 상호 작용하는 관계
	- **Include**: 가리키는 대상이 실행되어야 base use case도 실행된다 (mandatory)
	- **Extend**: 조건에 따라 추가되는 use case
	- **Generalization**: Actor / Use case는 behavior / relationship을 inherit

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure12.png)

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure13.png)


**Sequence Diagram**

Actor와 object 간의 interaction을 설명한다. (주고 받는 message)

Interaction의 sequence을 특정 use case에 대해서 설명한다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure14.png)

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure15.png)

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure16.png)

## Structural Perspective

Structural model을 사용한다. System의 organization을 통해 system과 relationship을 보여준다.

이때 class diagram을 사용한다. SW system에서의 object classes의 static structure을 modeling한다. message의 순서는 표시되지 않는다.


![joowan1108]({{site.url}}/images/SKKU_SwEng/figure17.png)

Class 간의 relationship은 5가지가 존재한다.

- **Inheritance**
	- 하위 클래스가 상위 클래스를 가리킴
- **Association**
	- 두 객체들 간의 연결 /연관
- **Aggregation**
	- 전체 (하얀색 사각형) - 부분 으로 표시되고 부분은 life cycle이 전체와 independent함. Independent하게 exist가 가능함
- **Composition**
	- 전체 (검은색 사각형) - 부분 으로 표시되고 부분은 life cycle이 전체와 dependent함.
- **Multiplicity**
	- 관계에 참여할 수 있는 객체 개수
	- 4..* 적혀있으면 4개 이상이 참여한다고 보면 된다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure18.png)

## Behavioral models

Dynamic behavior을 modeling함

System이 stimulus에 respond할 때 어떻게 되는지를 보여줌 (Stimulus = data / event)

Stimulus에도 다양한 종류가 존재한다.

Business system들은 data driven stimulus이다. 예를 들어서 phone billing system에서 call이 생기면 cost을 계산하고 bill generation 은 stimulus가 data이다.

한면 real time system들은 event driven stimulus이다. 

**Data driven model**

Input/output을 처리하는 sequence of actions을 보여준다

이때 사용하는게 activity diagram이다. Activity는 rounded rectangles이고 data는 rectangles으로 보여준다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure19.png)

**Event driven model**

System이 external과 internal event에 어떻게 반응하는지 보여줌

이때 states의 수가 제한적이고 event는 state transition을 일으킨다는 전제가 존재함

State diagram에서 rounded reactiangles는 system state을 보여주고 labeled arrows는 transition을 보여준다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure20.png)

**Superstate**

System이 복잡해질수록 states가 늘어난다.

이때 모든 state들을 개별적으로 표현하면 모델이 복잡해지므로 세부사항을 묶어서 숨긴다.

그리고 state들을 하나의 상위 state으로 묶어서 표현한다. 

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure22.png)
# Model Driven Architecture

System modeling의 결과 model을 통해서 실제 software design / implementation에도 사용하는 방법이 MDA이다.

즉, model을 중심으로 소프트웨어를 설계하고 구현하는 접근법이다. 지금까지 배운 system modeling은 시스템을 이해하고 다른 사람에게 설명하기 위해서인데 MDA는 이것을 구현 과정으로 옮기는 과정을 다룬다고 보면 된다.

이 MDA는 하나의 가설을 바탕으로 작동한다. Programs that execute on a SW platform are generated automatically from the models == 프로그램을 모델로부터 자동으로 생성할 수 있을 것이라는 기대 또는 가설

MDA에는 세 가지 abstract system model을 사용한다.

- **CIM Computation Independent Model**
	- Model important domain abstractions
- **PIM Platform Independent Model**
	- Enable mapping to one or more platforms by abstracting technical details (플랫폼 특징을 고려하지 않는다)
- **PSM Platform Specific Model**
	- Combine with PIM with details required for using in particular type of platform

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure21.png)

각 model들은 자기만의 translator을 통해서 다른 model에게 전달한다. 이때 필요한 정보가 각각 다르다.

**하지만 MDA가 주요 approach가 되지 않은 이유는 다음과 같다.**

Abstractions that are usefulf ro discussion are not right abstractions for implementation

For complex system, implementation is not the major problem
- requirements engineering, security, dependability, testing이 더 중요한 문제임

Agile method가 너무 좋음



