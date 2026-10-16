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
Software system을 개발할 때의 문제는 빠른 배포를 할 것인가 소프트웨어의 품질을 좋게 할 것인가이다.

Software process 중 Plan Driven method는 소프트웨어의 품질을 좋게 하는 방법론인 반면 빠른 처리가 중요한 환경에서는 Agile method가 유리하다. 

# Agile Development

Program specification, design, implementation이 모두 interleave 되어있다. System specification을 간단화하고 documentation을 최소화하였다.

개발은 series of versions / increments으로 되어있고 stakeholder는 later version을 위한 new requirements을 제안한다.

빠른 일처리를 위해서 자동화된 test tool처럼 Extensive tool support도 필요하다.

Plan driven approach에서는 iteration은 within activities에서 진행되고 activities끼리는 진행되지 않는다. Agile approach에서는 activity끼리도 iteration이 진행된다는 차이가 있다.

따라서 agile approach는 software에만 집중을 하고 evolving도 빠르게 한다.

## Principles of Agile Development

- Customer Involvement
	- Customer들이 development에 많은 관여를 한다
- Incremental Delivery
	- Increments으로 개발
- Embrace Change
	- Expect to change
	- System design 자체가 변화에 유연함
- Maintain Simplicity
	- 간단함을 유지
- People, not process
	- 자기에게 맞는 방법으로 스스로 개발


## Agile 개발 방법론: Extreme Programming

Iterative development to extreme levels

Several versions of system are developed, integrated, and tested in a day

Extreme Programming에서 수행하는 practices에는 10가지가 있다. 이때 company들은 선택적으로 사용한다.

**Extreme Programming Practices**

- Collective ownership
	- work on all areas
	- take responsibility for all of the code
- Continuous integration
	- Integrated to whole system
	- 그 다음에는 unit tests가 pass 해야 함
- Incremental planning
	- Requirements는 그때그때 남은 시간마다 달라짐 + relative priority
- On-site customer
	- customer가 development team에 참여해서 requirements 알려줌
- Pair programming
	- Work in pairs
- Refactoring
	- Code simpler
- Simple Design
	- meet current requirements only
- Small releases
	- 최소한의 유용한 기능 먼저
- Sustainable pace
	- 야근 x
- Test first development
	- Implementation보단 test을 먼저 작성함


**User Story**

이때 requirements을 짧게, 이해하기 쉽게 정의하기 위해서 user story을 사용한다.

User가 system을 사용하면서 경험할 법한 story card을 통해 customer needs을 전달한다.

이 story card는 여러 task으로 분해

각 task마다 필요한 effort와 resources을 계산

Story들을 business support usefulness을 바탕으로 우선순위 정하기

$\rightarrow$ 이해하기는 쉽지만 불완전하다

**Test first development**

Test가 implementation보다 먼저여야 하기에 test development도 incremental하다

Test development에 user도 involve한다. 그리고 automated test framework (tool)도 사용한다. (Agile을 위해서)


ex) 일회분 분량 check task에 대한 test 작성한다고 하자

Single dose of drug는 몇 mg으로 구성되는지 그리고 이 dose을 하루에 몇번 받는지가 input이라고 하자.

Test case:  총 4개임 If else statement가 4개이므로 (조건 2개 : 가능성 $2^2$)

1. Dose의 양은 맞더라도 frequency가 너무 빠르거나 느린지 확인
2. Frequency는 맞더라도 dose 양이 너무 높거나 낮은지
3. Dose와 frequency가 위험한 범위인지
4. Dose와 frequency가 모두 정당한 범위인지

 Test first development의 문제점

- Test을 제대로 작성하지 않음 (short cut)\
- Test는 increment하게 작성하기 어려움
- Test가 complete coverage을 하는지 판단하기 어려움 (기준은 100% branch < 100% path)

**Pair Programming**

Pairs는 dynamic하게 구성되며 모든 team member가 다른 사람과 같이 일을 함

장점
- Collective ownership and responsibility
- Informal review
- Encourage refactoring 

실험 결과
- 시간 오래 걸리지만 test pass 높음
- 만족도 높음
- Code 수 적음 (refactor)


## Agile Project Management: Scrum

Agile의 informal planning과 project control은 너무 빠르게 변하기에 visibility가 없다는 단점이 있다. 즉, 진행 상황을 파악하기 어려움

이를 알아내기 위해서 scrum이라는 방법론이 사용된다. Scrum은 새로운 terminology을 사용해서 구별해서 사용한다.

**Scrume Terminology**

- Development team
	- 7명 이하로 이뤄진 self organizing group
- Sprint
	- development iteration (2~4 weeks)
- Scrum
	- Daily meeting: progress와 우선순위 작업 설정
- Scrum Master
	- Project manager와 비슷
	- 다른 사람들과 interface
- Product backlog
	- To do list
- Product owner
	- 요구사항 알아내기
	- Backlog을 보고 요구사항과 맞는지 확인
	- Customer / Product manager 등이 될 수 있음
- Velocity
	- 작업 처리 속도
	- backlog을 하나의 sprint에서 얼만큼 처리할 수 있는지

![joowan1108]({{site.url}}/images/SKKU_SwEng/figure6.png)

Sprints는 절대 연장되지 않고 끝나지 않은 일은 다시 backlog으로 돌아간다

Scrum board는 team간의 daily interactions을 보여준다

# Scaling Agile Methods

**장점**
- Product가 이해가 가능한 수준으로 분해되기에 stakeholder들도 감당 가능하다
- 요구사항 불완전해도 progress가 멈추지 않음
- 전체 team이 visibility을 가진다 (Agile의 visibility의 문제를 해결한다)
- Customer는 시간에 따른 increments을 갖고 feedback을 줄 수 있다
- Trust

**Agile method 자체의 단점**
- Agile은 우선 small / medium sized system에만 사용된다는 전제를 갖고 있다.
- Informality 때문에 contractual issue 발생 가능
- Agile은 new development에만 유용함
- Development가 worldwide 하게 distributed 되었을 때는 agile이 더 어려워짐

**Maintenance에서 Agile method 자체의 단점**
- Product documentation 부족
- Agile team이 분해되면 documentation으로 기록을 안 해두기 때문에 implicit knowledge가 아예 없어짐 
- Customer을 계속 참여시키는게 어려움


**Agile methods for large systems**

Agile이 large system에 사용되기 위해서는 개선이 필요하다.

**그럼 large scale system을 복잡하게 만드는 factor는 무엇인가**

- System of systems
	- 개별적인 team이 서로 다른 환경에서 개발을 하기 때문에 전체 system의 view는 파악이 어려움
- Brownfield development
	- 이미 존재하는 system을 사용함
- System configuration
	- 실제 개발보다는 system configuration에 맞춰져 있음
- Regulatory constraints
	- 규칙으로 개발 방식이 제한됨
- Prolonged procurement
	- 팀 유지 어려움
- Diverse stakeholders
	- 다양한 stakeholder들을 참여시키는 것은 어려움

**Agile method을 large system에 적용할 때 고려해야 하는 것**

- 완전한 incremental approach는 불가능함
- 한 명의 product owner는 불가능함
- 코드에만 집중하는 것은 어려움: documentation도 해야 함
- Cross team communication을 통해 team 간 visibility 얻어야 함
- 변화가 있을 때마다 전체 system에 반영시키는 것은 어려움

**Agile for Large System : Multi-team Scrum**

- Role replication
	- 각 team에 product owner와 scrum master 설정
- Product architects
	- 각 team은 product architect을 선정하고 이 architect들끼리 협업하여 overall system architecture을 구축함
- Release alignment
	- Product release 날짜들을 일치
- Scrum of Scrums
	- 각 team의 대표들끼리 scrum을 진행해서 진행 사항을 공유함