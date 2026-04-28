---
layout: single
title: "13: Paging Faster Translations (TLB)"
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

# Paging: Smaller Tables

Paging의 단점 중 하나는 paging을 하기 위해 저장해야 하는 page table이 너무 크다는 것이다. 이 문제를 해결하기 위한 방법을 알아보자

## Bigger Pages

Page table의 크기를 줄이기 위해서 할 수 있는 가장 간단한 방법은 page의 크기를 키워서 page 수를 줄이는 것이다.하지만 너무 큰 page을 할당하게 되면 internal fragmentation이 발생할 가능성이 높아진다. 

## Hybrid Approach

다른 방법으로는 paging과 segmentation을 둘 다 사용하는 것이다. 이를 설명하기 위해서 virtual address space가 16KB이고 page의 크기가 1KB라고 하자.

![joowan1108]({{site.url}}/images/SKKU_OS/pg244.png)

![joowan1108]({{site.url}}/images/SKKU_OS/pg244_1.png)

Table을 보면 대부분의 page가 사용되고 있지 않기에 invalid entries가 많아진다. 즉, 사용되는 정도에 비해 쓸데없이 너무 큰 page table을 사용하는 것이다.

이런 낭비를 줄이기 위해서 hybrid는 각 logical segment마다 page table을 따로 두면 각 segment가 필요한 만큼만 page table의 크기를 정할 수 있게 해준다. MMU에는 각 segment의 page table의 physical address을 base register에 저장하고 사용하는 것이다. 

Hybrid을 적용한 예시를 살펴보자. 32 bit virtual address space와 page 크기가 4KB이라고 하자. 그리고 code segment는 01, heap은 10, stack은 11이라고 하자. 그러면 virtual address는 다음처럼 생기게 된다.

![joowan1108]({{site.url}}/images/SKKU_OS/pg245.png)

각 segment마다 base / bound 쌍이 생기므로 context switch가 될 때마다 이 register 값만 갈아끼우면 되는 것이다. Hardware managed TLB을 사용한다고 했을 때TLB miss가 발생했을 때는, Segment bit (Seg) 값을 톻해서 hardware는 page table의 정확한 위치를 알 수 있게 되는 것이다. 

``` c

SN = (VirtualAddress & SEG_MASK) >> SN_SHIFT
VPN = (VirtualAddress & VPN_MASK) >> VPN_SHIFT
AddressOfPTE = Base[SN] + (VPN * sizeof(PTE))

```

Bounds에는 각 segment의 page table의 크기가 저장된다. 따라서, 각 segment마다 bounds의 값이 달라진다. 이로 인해 생기는 문제는 당연히 external fragmentation과 flexible하지 않다는 것, 그리고 크기가 다르기 때문에 physical memory에 저장하는 것이 어려워진다는 것이다.


## Multi level Page Tables

그래서 다른 접근 방법을 사용한다. 지금 문제가 page table에 invalid한 page 정보까지 들어가야 한다는 것이다. 따라서 invalid한 page는 저장하지 않는 방법을 고안하였다. 

Multi level page table은 이 아이디어를 사용한 방법이다. Linear page table을 tree로 바꾸는 방법이라고 생각하면 된다.

우선 page table을 page 크기로 나누어서 그 chunk 안에 모든 page entry가 invalid 하다면 그 chunk 안에 page 정보는 page table 안에 저장하지 않는것이다. 그리고 chunk 안에 valid한 page entry가 하나라도 존재한다면 그 chunk의 주소가 **page directory**에 저장된다. Multi level page table의 예시를 보면 다음과 같다.


![joowan1108]({{site.url}}/images/SKKU_OS/pg247.png)

왼쪽의 일반적인 linear table의 경우, invalid한 page entry을 모두 저장하는 반면, 오른쪽의 multi level page table은 page directory을 통해 valid page들이 들어있는 page의 주소를 알려주고 각 주소에는 valid page entry가 든 page 크기의 memory가 있어서 저장되는 invalid page entry의 수를 줄일 수 있게 된다. 

Page directory에 들어가는 최소 정보는 PFN과 valid bit이다. 하지만 이때의 valid bit는 최소 한 page entry가 valid 한지를 알려주는 bit이다. Multi level page는 page 단위로 정보를 저장하면서 필요한 정보만 저장하기에 memory 관리가 수월하고 효율적이다. 

하지만 단점은 TLB miss가 났다고 했을 때 TLB update을 위해 page table로부터 올바른 정보를 가져오는 과정에서 memory access을 두 번 해야한다. (page directory로 한번, page table에서 PTE에 접근하는데 한 번)

또, complex하다는 단점이 존재한다.
