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
# Introduction

- **Program**: collection of instructions
- **Software**: Programs + documentation (testing results, user manual, etc.)

좋은 Software의 Essential Attributes
- Maintainability
	- Evolve to meet changing needs
	- annotation 중요
- Dependability and Security
	- Should not cause damage
	- Malicious users cannot access
- Efficiency
	- responsiveness, processing time, memory utilisation 중요
- Acceptability
	- Understandable, usable, compatible

**ISCE** (International Conference on Software Engineering)에서 정의하는 필요한 documents
- README: what the software does
- REQUIREMENTS: environment information
- LICENSE: distribution rights
- INSTALL: Usage example / installation test

Software 개발에 드는 cost는 develop보다 maintain하는데 오래 걸린다. Debugging만 해도 develope의 절반의 cost가 필요하다고 한다.

**Software project들이 실패하는 이유** 

- System complexity increase exponentially on software size
	- Execution path increase exponentially with conditional statement

- Software engineering method 사용 x


# Software Engineering

Software engineering이란 early stage부터 all aspects of software production + after it has gone to use까지 고려하는 engineering discipline. Technical process만 고려하는게 아님

**Documenting** $\rightarrow$ **Programming** $\rightarrow$ **Testing** $\rightarrow$ **Debugging** $\rightarrow$ **Evolving**

이때 engineering discipline이란 constraints을 고려한 theories와 methods을 사용해서 문제를 해결하는 것

Software Engineering에 사용되는 topics

1. AI and software engineering
2. Testing and analysis
3. Software Evolution
4. Social Aspects of Software Engineering
5. Requirements, Modeling, Design

Software Engineering이 중요한 이유
1. Dependency of Software: Software System에 더 의존하게 된다. 
2. Cost saving: 길게 보면 cost saving. 대부분의 cost는 사용하기 시작한 다음에 software가 바뀔 때 생긴다

Software Engineering의 key challenge: Diversity가 증가하더라도 demnads for reduced deliveryt imes and developing trustworthy software

# Software Process Activities

- **Software Specification**
	- Define the software and constraints
- **Software Development**
	- Designed and programmed
- **Software Validation**
	- Check requirements are implemented
- **Software Evolution**
	- Modify

**Software에 영향을 주는 문제들**
- **Heterogeneity**
	- 여러 environments에 작동해야 한다.
- **Business and Social Change**
	- Business와 society가 빨리 변한다
- **Security and Trust**

# Software Products

- Generic Products
	- Stand alone system: Marketed and sold to any customer
	- Specification은 software developer가 가짐
	- Decision도 developer가 함
- Customized Products
	- Commisioned by specific customer
	- Specification은 customer가 가짐
	- Decision도 customer가 함

하지만 이 product type 간의 distinction이 점점 옅어지고 있다.

# Software Engineering Ethics

Wide Responsibility, Honest, more than upholding the law

**Issues of professional responsiblity**

- Confidentiality
- Competence
	- Should not misrepresent level of competence
- Intellectual property rights
	- local laws, intellectual property
- Computer misuse


> ACM: Association for Computing Machinery
   IEEE: Institute of Electrical and Electronics Engineers

Ethical Principles에서 Colleagues, self가 중오

- Colleagues: Fair and Supportive
- Self: Lifelong learning