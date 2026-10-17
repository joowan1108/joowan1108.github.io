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
# Architectural Design

소프트웨어 전체 구조를 어떻게 설계할 것인지 설명한다. 그러기 위해서는 네 가지 관점을 얻어야 한다. 

1. Designing the overall strucutre of SW system
2. Identifying the main structural components and relationships
3. Satisfying functional and non-functional requirements
4. Depending on system type, architect experience and specific requirements

Architecture을 표현한다고 할 때 block diagram을 사용할 수 있다.

Box는 component을 표현하고 arrow는 flow of data / signal을 표현한다.

Architectural design을 하면 얻은 세 장점이 있다.

- **Stakeholder communication**
- **System analysis**
	- System이 critical requirements을 만족할 수 있는지에 design이 영향을 준다
- **Large scale reuse**
	- Large scale SW reuse에 도움이 된다.

**이때 architectural design 결정에 영향을 주는 것은 non functional requirements이다.**

- **Performance**
	- 적은 수의 components에 critical operations 압축
		- Localize critical operations within a small number of components
	- Large components을 통해 Component communications을 최소화 
- **Security**
	- Layered structure을 사용하여 critical assets을 innermost에 숨김
- **Safety**
	- Reduce the costs and problems of safety validation
	- Safety related operations을 하나의 component을 압축
- **Availability**
	- Components을 중복 사용해서 하나가 무너져도 작동되도록
- **Maintainability**
	- Fine grained, self-contained compoenents을 사용해서 유지 보수 쉽게

이때 performance와 maintainability가 충돌한다. Performance는 large components을 사용하고자 하고 maintainability는 small components을 유도한다.

## Architectural Views

어떤 views / perspective가 designing에 중요할까?

다양한 사람들은 자기에 맞는 정보만 보고 싶어 하기 때문에 하나의 그림으로 모든 정보를 표현하기 어렵다.

따라서 4+1 view model을 사용한다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure23.png)

- **Logical View**
	- Key abstractions in the system as objects
	- Functional requirements
- **Process View**
	- interacting process가 run time에 어떻게 구성되는지
	- Non functional requirements 중에 performance
- **Development View**
	- Software가 어떻게 decompose 되었는지
	- SW managers / developers을 위한 것
- **Physical View**
	- HW / SW가 어떻게 분산되어있는지
	- System engineers
- **Scenarios**
	- Object와 process 간의 sequence of interactions

## Architectural Patterns

Abstract description of good practice

이미 다양한 environments와 system에 tested / tried 된 것

이때 특정 pattern이 언제 안 좋은지 / 좋은지에 대한 정보도 들어있음

 ### Model View Controller Pattern

Separate presentation and interaction of system data

**Model**: system data와 data에 가해지는 operations

**View**: Data가 user에게 보여지는 부분

**Controller**: User interaction과 interactions들을 model과 view에 보내는 것

**언제 사용하는가**
- Data을 view하고 interact하는 여러 방법이 있을 때 사용

**장점**
- Data가 independent하게 변함
- 다양한 data에 대한 Presentation을 다양하게 하기

**단점**
- Data model과 interaction이 간단해도 code complexity가 높다

### Layered Architecture

System을 여러 layer으로 쌓고 각 layer가 위 layer에 service을 제공한다

**User interface**

$\uparrow$

**User interface management** 

**Authentication and authorization**

$\uparrow$

**Core business logic / Application functionality** 

$\uparrow$

**System support (OS, Database)**

**특징**
- Separation과 independence가 가능하고
- Incremental development of system을 지원
	- Layer interface가 달라지면 Adjacent layer만 영향을 받는다

**언제 사용하는가**
- 이미 존재하는 system에 추가를 할 때
- 여러 team이 관여하고 각 team은 하나의 layer을 담당할 때
- Multi-level security 가능

**장점**
- 전체 layer의 교체가 가능하다. Interface만 유지되면 내부 수정은 상관 없다
- 각 layer에서 중복 facilities가 제공될 수 있어서 system의 신뢰성 (dependability)가 높다

**단점**
- Layer간의 clear separation 어려움
- 붙어있지 않은 layer 간의 직접적인 소통이 불가능함
- 여러 계층을 거쳐야 되기 때문에 performance 감소

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure24.png)


### Repository Architecture

모든 data가 하나의 repository에 관리된다.

Component끼리는 직접 소통할 수 없고 repository을 통해서만 가능하다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure25.png)

**언제 사용하는가**
- Large volume of information이 오랫동안 저장되어야 할 때

**장점**
- Component independent 가능
- 서로 존재 알 필요 없음
- 모든 data가 managed consistenly (한 곳에만 존재하므로)

**단점**
- Repository가 고장나면 전체가 고장남
- 모든 communication을 하나의 repository으로 관리하는 것은 어려움
- 이 repository을 여러 computer에 분배하는 것도 어려움

Toy story 2에서 pixar가 rm -rf을 root에 실행해서 toy story 파일이 다 없어졌을 뻔했다.


### Client Server Architecture

System이 여러 service으로 구현되어 있고 각 service는 자기만의 server를 가진다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure26.png)

**주요 components**
- Set of servers
	- service 제공자
- Set of clients
	- Server 이용자
- Network
	- Client가 server에 접근하도록 도와줌

**언제 사용하는가**
- 다양한 곳에서 shared DB에 있는 정보에 접근해야 할 때

**장점**
- Server가 network에서 분배가 가능
- General한 functionality가 모든 client에게 제공될 수 있다.


**단점**
- Denial of Service에 영향을 받을 수 있다
- Performance가 network에 영향을 받음
- Server를 직접 구현하지 않으면 management problem 존재


### Pipe and Filer Architecture

Data을 여러 처리 단계로 전달하면서 변환

각 processing component (filter)는 discrete하고 one type of data transformation을 수행한다.

Data flow는 pipe으로 filter들을 지나가도록 한다.

**언제 사용하는가**
- Data processing app
- Input이 분리된 stage들을 통해 처리될 때
- ex) cat test.c | grep 'cha' 

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure27.png)

**장점**
- 이해하기 쉽고 transformation reuse 가능
- business process와 일치
- tranformation 추가 및 evolution이 직관적임

**단점**
- Filter가 받고 주는 data 형식이 미리 정해져있어야 함
- Data 구조 간의 호환이 되어야 한다.

# Application Architecture

Business가 사용하는 Application system은 공통적인 특성을 가진다.

Application architecture는 특정 system의 principal characteristics을 encapsulate 한다.

Application architecture model은 사용 용도가 4가지이다.
- Architectiral design process의 starting point
	- 기존에 알려진 구조를 통해 필요한 component들을 결정 가능
- Design checklist
	- 설계 과정에 필요한 구성요소나 기능을 누락하지 않았는지 확인 가능
- Organizing the work of development team
- Application type을 위한 의사소통을 위한 vocabulary
	- 공통적인 개념을 사용하여 의사소통 편리화

Application Architecture의 종류에는 2가지가 있다

- **Transaction Processing Applications**
	- Database 중심
	- User가 data을 사용하고 update하는 작업
	- Interactive business system에서 가장 흔한 구조
	- 이때 transaction 이란 특정 goal을 만족하는 sequence of operations
	- Pipe and filter architecture으로 표현될 수 있다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure28.png)

> Information System
> 모든 system이 shared database와 작용을 한다 ex) web based
> 이때 information system은 layered approach을 적용한다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure29.png)

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure30.png)


- **Language Processing Systems**
	- User의 의도가 formal langauge으로 표현되고 그 의미를 이해하는 것
	- Compiler라고 보면 된다 (high level program -> machine code)
	- Translator라고 봐도 된다.

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure31.png)

**Lexical analyzer**

- character stream
pos = init + rate * 10

- Lexemes sequence
“pos”, “=”, “init”, “+”, “rate”, “$*$”, “10”

- token sequence
(ID, pos), ASSIGN, (ID, init), PLUS, (ID, rate), MULT, (NUM,10)

**Syntax Analyzer**

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure32.png)

**Semantic Analyzer**

Type errors 검토

Semantic erros 검토
- Array out of bounds
- null dereference
- divide by zero

**IR Translator**

Intermediate Representation (IR)

- Lower level than source langauge
- higher level than target language

Three address code으로 바꿈

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure33.png)

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure34.png)

