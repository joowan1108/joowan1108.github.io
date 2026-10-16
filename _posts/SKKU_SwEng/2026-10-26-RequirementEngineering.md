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

# Requirement Engineering

- Requirement
	- System이 제공해야 하는 service의 descriptions과 constraints
	- User requirements는 abstract한 service / constraint
	- System requirements는 detailed한 formal definition

- Requirement Engineering
	- Service와 constraints을 finding out, analyzing, documenting, checking 하는 과정
	- Software Engineering process의 첫 단계

User와 System은 알아야 하는 정보, 이해하는 정보가 다르기 때문에 user쪽은 abstract한 것만 알면 되고 system은 detailed한 것까지 알아야 함

Stakeholder란 system에 어떻게든 영향을 받는 사람인데 이들은 system에 영향을 주면서 요구는 모두 다르다

System requirements에는 두 종류가 있다.

- Functional requirements
	- System이 제공해야 하는 statements (input / output)
	- System이 하면 안 되는 일
- Non-functional requirements
	- Service의 constraint (responsiveness 등)
	- 전체 system의 constraints

하지만 이 둘의 구분은 정의만 봤을 때는 어렵다.

> A statement limiting access to authorized users
> - functional 관점: 승인되지 않은 사용자에게 기록을 보여주지 않는다
> - Non functional 관점: 시스템은 보안 정책을 충족해야 한다

## Functional Requirements

functional requirements는 general하거나 specific 하다. 

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure7.png)

이 서비스에서 functional requirements의 예시는 다음과 같다.
1. User is able to search the appointments lists for all clinics
2. 각 clinic과 날짜마다 예약된 환자 명단을 생성해야 한다
3. 각 staff는 employee number로 identified 되어야 한다.

functional requirements의 **단점**은 imprecision, 불명확하다.

- 이 불명확함은 developer와 customer의 이해도에 영향을 줄 수 있다
- System delivery가 늦어질 수 있고 cost도 높아질 수 있다.

ex) User is able to search the appointments lists for all clinics

- 모든 clinic에서의 모든 예약에서 이름으로 검색
- User가 참여한 clinic에서 이름을 검색

$\rightarrow$ 불명확함

따라서 이상적인 functional requirements는 두 특징을 가져야 한다.

- Completeness: User가 요구하는 모든 service가 정의되어 있어야 한다
- Constistency: 요구사항들이 모순되면 안된다

하지만 large system에서는 stake holders가 많고 실수가 생기기 때문에 이 두 특징을 갖기 어렵다


## Non Functional Requirements

- 시스템 전체의 특징에 constrain을 건다 
	- ex) response time, memory use, reliability

- Non functional requirements가 individual function requirements보다 더 큰 영향을 준다
	- Non functional requirements가 안 맞으면 전체 시스템의 사용 가능성에 더 큰 영향을 준다

- Non functional requirements는 user needs으로부터 나온다

**Types of non-functional requirements**

- Product requirements
	- Runtime behavior of software
	- execution speed, memory usage, 등등
- Organizational requirements
	- Organization에서부터 오는 정책으로 인한 requirements
	- 프로그래밍 언어, process standard 등
- External requirements
	- System 외부 factor
	- 법을 생각하면 된다

ex) Mentcare System의 non functional requirements

Product requirement
- 정상 근무 시간에는 작동을 해야 하며 작동하지 않는 시간은 5초를 초과하면 안된다

Organizational requirement
- Identity card을 통해 접근이 가능해야 한다. (Customer's organization policy)

External requirement
- HStan-03-2006-priv에 명시된 환자 개인정보 보호 규정을 구현

**Non functional requirements의 문제**

Stakeholder들은 requirements을 general하게 표현한다. ex) 사용하기 쉬운, 반응이 빠른..

예를 들어 The system should be easy to use by medical staff and should be organized in such a way that user errors are minimized.

--> User errors are minimized는 너무 general하고 구체적인 수치가 제공되지 않는다

이를 해결하기 위해서 이런 요구 사항들을 **testable** 으로 바꿔야 한다.
ex) 평균 error 발생이 시간당 두개를 넘으면 안된다

즉, non functionasl requirements을 수치적으로 testable하게 작성해야 한다. 그리고 customer의 goal을 측정 가능한 requirements으로 바꿔야 한다. 이때 측정 property는 speed, size, ease of use, reliability, portability 등이 있다.


## Requirements Engineering Key Activities

어떻게 functional / non functional requirements을 알아내고 문서화하고 검증할 것인가

- Elicitation and analysis
	- Stakeholder와 상호작용해서 requirements 얻기
- Specification
	- Requirements을 documentation 하기
- Validation
	- Requirements가 제대로 되어있는지 확인하기

Requirements Engineering은 iterative하고 interleaved 되어있다. (Spiral)

### Elicitation

Stakeholder가 하는 일을 이해하고 이 일을 어떻게 도울 것인가를 이해하기

이때 문제는 다음과 같다

- Stakeholder들이 SW system으로부터 무엇을 정말 원하는지 모른다
- Stakeholder의 domain 단어들로 설명한다
- Stakeholder마다 요구사항이 다르다
- Political factors가 관여한다
- Requirements가 dynamic하게 바뀐다.

**Requirements discovery and understanding**

- Stakeholder와 상호작용

**Requirements classification and organization**

- 관련된 requirements을 묶고 clustering하기

**Requirements prioritization and negotation**

- Requirement 들에 우선순위를 매기고 conflict 해결하기

**Requiremnts documentation**

-  Documenting

Elicitation을 하기 위해서 보통 두 가지 방법을 사용한다

- Interviewing
	- Stakeholder와 대화
	- Closed interview
		- 미리 알려준 질문 대답
	- Open interview
		- 알려지지 않은 질문 대답
- Observation or Ethnography
	- 직접 일하는 것을 관찰하기

하지만 interviewing을 통해 정보를 알기가 어렵다

- domain specific term들을 사용하고 

- 자기의 knowledge을 설명하기도 어렵고

- fundamental knowledge을 빼먹을 수도 있다.

Interviewing을 보완하기 위한 방법이 Stories and Scenarios이다.

**Stories and Scenarios**

특정 task을 위해서 system이 어떻게 활용될 수 있는지에 대한 설명

- **Stories**: narrative text for **High level** description of system use 
- **Scenarios**: **specific** information collected such as inputs and outputs

Discussion을 쉽게 도와주고 starting point가 된다.

Story는 어떻게 보면 사용 맥락이고 scenario는 user story의 구조화된 버전이다.

Scenario의 특징

- Scenario가 시작하는 지점 (initialization state)에 대한 설명
- normal flow의 description
- what can go wrong의 description
- 동시에 실행되는 다른 activity의 정보
- Scenario가 끝나는 지점에 대한 설명


### Requirement Specification

User와 system requirements을 문서화하는 단계

User requirements는 natural language으로 적혀있고  end-user가 이해하기 쉽다

System requirements는 user requirements의 확장 버전으로 implementation에 사용되는 contract이다.

이때 system requirements을 작성하기 위해서는 다양한 방법을 사용할 수 있다.

- Natural language
	- Numbered sentence 사용
	- 각 sentence마다 하나의 requirement만 담는다
	- 가장 많이 사용되지만 ambiguous하다
	- highlighting을 사용해야 한다.
	- 필수
- Structured natural language
	- Template에 적혀있다
	- 각 field는 requirement의 한 aspect을 담는다 
- Graphical notations
	- Functional requirements을 담는다
	- UML use case와 sequence diagram이 사용된다
- Mathematical specification
	- Finite state machines가 사용된다.
	- Unambiguous는 없어지지만 customers가 이해하기는 어렵다



