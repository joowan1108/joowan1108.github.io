---
layout: single
title: "8: Address Translation"
categories: SKKU_OS
tag: [SKKU]
author_profile: false
sidebar:
    nav: "counts"
toc: true
toc_sticky: true
toc_label: Table of Contents
use_math: true
---


# Memory Virtualization에 필요한 mechanism: Address Translation

CPU virtualization에서 LDE, 즉 system call과 timer interrupt와 같은 hardware의 보조를 받았었다. Memory에서도 LDE처럼 hardware의 도움을 통해 memory virtualization을 가능하게 한다.

Memory virtualization에서 활용하는 hardware support는 address translation이다. 각 process가 자기의 virtual address space에 접근하면 hardware가 virtual addresss space 접근을 physical address 접근으로 바꾸어준다는 것이다. 

> 이때 이 translation을 해주는 hardware가 Memory Management Unit (MMU)이다. 

하지만 memory virutalization이 hardware으로만 되는 것이 아니다. OS가 memory을 관리를 해야 하며 어디가 free되고 allocate 되었는지 확인해주어야 한다.


## Assumptions

Memory virtualization을 설명하기 위해서는 우선 virtual address space가 physical memory에 연속적으로 놓여지고, address space의 크기가 physical memory보다 작고, 모두 동일한 크기를 가진다고 가정한다.

## Address Translation

예를 들어 process가 다음 code을 실행한다고 하자.

``` c
void func() {
int x = 3000; 
x = x + 3; // line of code we are interested in
```

Memory에 존재하는 x을 load하고 3을 더한 뒤에 다시 memory에 저장하는 코드이다. 이 코드는 실행되기 위해 assembly로 변해서 address space의 128번 주소에 존재하게 된다.

``` c
128: movl 0x0(%ebx), %eax ;load 0+ebx into eax
132: addl $0x03, %eax ;add 3 to eax register
135: movl %eax, 0x0(%ebx) ;store eax back to mem
```

Process 입장에서 이 코드를 실행할 때 다음 순서로 행동을 한다.

- Fetch instruction at address 128
- Execute this instruction (load from address 15 KB)
- Fetch instruction at address 132
- Execute this instruction (no memory reference)
- Fetch the instruction at address 135
- Execute this instruction (store to address 15 KB)

메모리에 접근하는 행동들만 따로 보면 process 입장에서는 자기 address space의 주소 128, 132, 135에 접근을 해서 코드를 불러와 실행을 한다. 

![joowan1108]({{site.url}}/images/SKKU_OS/pg169.png)

하지만 이 주소는 실제 physical memory address와 동일한 것이 아니다. 다음 그림처럼 사실 각 process의 address space는 physical address space의 한 부분에 저장되어서 실행되는 것이다.

![joowan1108]({{site.url}}/images/SKKU_OS/pg170.png)

이때 physical memory의 어떤 위치에 어떻게 virtual address space을 저장시킬 지 결정하는 방법은 여러 가지가 존재한다. 그 중 가장 간단한 방법은 Dynamic Relocation이다.

### Dynamic (Hardware based) Relocation (Base and Bounds method)

첫 방법은 dynamic relocation 또는 base and bounds라고 한다. Program이 실행될 때 OS가 program의 virtual address space의 시작 지점이 physical memory 내에서 어디에 위치할지 정한다. 이 시작 지점을 base라고 한다. 즉, 위 그림에서 virtual address space는 32KB부터 시작되므로 base는 32KB인 것이다.

이후, process에서 자기의 virtual address space에 접근한다고 해보자. 이때, address translation이 사용되는 것이다. 접근하려는 virtual address space의 주소를 hardware을 이용해 physical address space의 주소로 바꿔주는 것이다.

> physical address = 접근하려는 virtual address space 주소 + base

*이때 virtual address space는 bounds (virtual address space의 크기)보다 크거나 음수이면 안 된다. 접근하려는 virtual address space가 허용된 범위를 벗어나면 CPU에 의해 exception이 발생한다. *

실제로 한 번 계산해보자. 위 예시에서 x = x+3 코드를 실행하기 위해서는 virtual space의 128번 주소에서 instructions을 가져와야 한다고 하였다. Process가 128번 주소에 접근하려고 할 때 hardware는 이 정보를 통해 대응되는 physical memory 주소을 가져와주는 역할을 하는 것이다.

1KB는 $2^10$ 이기 때문에 physical memory의 32*1024 + 128 = 32768번 주소값을 가져와주는 것이다. 이 address translation 과정이 바로 memory virtualization을 가능하게 해주는 hardware의 도움이다. 이 hardware의 도움으로 process는 physical memory을 접근할 수 있기 때문에 physical memory을 다 소유하고 있다는 착각이 드는 것이다.

> 이 address relocation이 runtime에서 발생하고 process가 시작했을 때도 base 값만 바꾸면 address space을 옮길 수 있다는 점에서 dynamic relocation이라고 부른다.

#### OS support needed for Dynamic Relocation

Dynamic Relocation에서 OS도 관여를 해야 한다.

우선 새 process가 생길 때 OS는 이 process의 virtual address space을 저장할 빈 공간을 찾아야 한다. 하지만 virtual address space는 physical memory보다 작으며 모두 동일한 크기를 가진다고 가정하였기 때문에 그냥 빈 slot을 찾는 것처럼 간단하다. (physical memory가 virutal address space의 단위로 나뉘어져 있으므로)

또, process가 끝났다면 그 process에 할당된 slot을 비워야 한다.

그리고 dynamic relocation에서 각 CPU는 한 쌍의 base register와 bounds register가 존재하기 때문에 context switch가 일어날 때 각 process의 base register와 bounds register 값을 PCB에 저장하고 불러와야 한다.
