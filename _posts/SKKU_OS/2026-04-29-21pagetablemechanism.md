---
layout: single
title: "14: Beyond Physical Memory: Mechanisms"
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

# Beyond Physical Memory: Mechanisms

지금까지 모든 process의 address space가 memory 안에 들어간다는 가정을 하였다. 하지만 이 가정을 없애고 concurrent 하게 여러 개의 process의 큰 address space을 지원하고 싶다고 해보자. Process에게 크고 독립적인 address space을 주게 된다면 process는 얼만큼의 공간이 남았는지 신경쓰지 않고 원하는 만큼 memory을 사용할 수 있기 때문에 편리하다.

하지만 이렇게 큰 address space을 제공하게 된다면 모든 address space가 physical memory에 안 들어갈 가능성이 높다. 따라서 OS는 memory hierarchy을 활용한다. OS는 사용되지 않는 page들은 더 큰 memory (hard disk drive)에 저장하고 필요할 때 다시 불러오는 방법으로 large address space을 제공할 수 있다.이렇게 사용되지 않는 address space을 저장하는 공간을 **swap space**라고 한다.

## Swap space

Disk와 memory 간에 page 정보를 swap 하면서 필요한 건 불러오고 필요없는 건 다시 넣기 때문에 swap space라고 한다. Swap space의 크기는 이 과정의 난이도를 결정하기 때문에 매우 중요하다.

우선 physical memory에 4개의 page가 들어가고 swap space에는 8개의 page가 들어간다고 해보자.

![joowan1108]({{site.url}}/images/SKKU_OS/pg259.png)

각 process는 physical memory을 공유하면서 일부 page들은 swap space에 저장해두고 있음을 알 수 있다. 

*Swap space가 swapping을 하기 위한 유일한 저장소는 아니다. 예를 들어 code page들은 원래 disk에 존재해서 program이 실행될 때만 load되고 더 필요없다면 이를 다시 disk에 넣을 수 있다. 이때 code page가 저장되는 곳은 꼭 swap space가 아니다.*

### Present bit

Swap space을 사용함으로써 찾고 있는 page가 memory에 존재할 수도 있고 disk에 존재할 수도 있게 된다. 따라서, swap space을 사용하기 위해서는 추가 정보가 필요하다. 예를 들어 TLB miss가 나서 hardware managed TLB가 miss난 page를 찾는데 이 정보가 memory에 존재하는지 disk에 존재하는지 알아내야 한다. 이럴 때 present bit가 사용된다. Present bit = 1이면 memory에, 0이면 disk에 존재한다는 것을 의미한다.

### Page Fault

Physical memory에 없는 page에 접근을 하면 page fault가 발생한다. Page fault가 발생할 때는 OS가 실행되고 page fault handler로 처리한다. 

Page fault handler는 찾고 있는 page을 disk에서 찾아 physical memory에 load하고 page table을 update하는 역할을 한다고 보면 된다. 이때, 다음 instruction을 실행하는 것이 아니라 fault가 났던 instruction을 다시 실행한다.


### Memory Full

Disk으로부터 page을 memory에 저장하고 싶은데 자리가 없다고 하자. 이때는 그럼 memory의 일부를 제거해야 한다. 어떤 page을 제거할 지 고르는 방법은 page replacement policy라고 한다. 

잘못된 page을 없애게 된다면 TLB miss가 늘어나 엄청난 성능 저하를 불러일으킬 수 있기 때문에 매우 중요하다.


### Page Fault Control

TLB Miss로 page fault가 일어날 때 하는 control 경우의 수를 소개한다.

1. Present 하면서 valid 한 경우

    Page가 present이면서 valid하다면, memory에 존재한다는 것이기에 PFN을 가져와서 instruction을 다시 시도한다.

2. Present하지 않은 경우

    Page가 present 하지 않다면, physical memory에 존재하지 않는다는 뜻이기에 hard disk에 접근을 해서 가져온다. 이때, page fault handler가 실행된다.

3. Valid하지 않은 경우

    이것은 그냥 valid하지 않은 접근이므로 OS trap handler가 처리하고 process을 terminate 한다.

이 control flow는 다음 코드로 더 자세히 알 수 있다.

``` c

VPN = (VirtualAddress & VPN_MASK) >> SHIFT
(Success, TlbEntry) = TLB_Lookup(VPN)
if (Success == True) // TLB Hit
    if (CanAccess(TlbEntry.ProtectBits) == True)
        Offset = VirtualAddress & OFFSET_MASK
        PhysAddr = (TlbEntry.PFN << SHIFT) | Offset
        Register = AccessMemory(PhysAddr)
    else
        RaiseException(PROTECTION_FAULT)

else // TLB Miss
    PTEAddr = PTBR + (VPN * sizeof(PTE))
    PTE = AccessMemory(PTEAddr)
    if (PTE.Valid == False)
        RaiseException(SEGMENTATION_FAULT)
    else
        if (CanAccess(PTE.ProtectBits) == False)
            RaiseException(PROTECTION_FAULT)
        else if (PTE.Present == True)
            // assuming hardware-managed TLB
            TLB_Insert(VPN, PTE.PFN, PTE.ProtectBits)
            RetryInstruction()
        else if (PTE.Present == False)
            RaiseException(PAGE_FAULT)

```

### Replacement가 실제로 일어난다면?

Replacement가 일어날 때는 physical memory가 꽉 차서 그런것이라고 가정을 했었다. 하지만 실제로는 physical memory가 가득 찰 때까지 기다리지 않는다. 

항상 조금의 여유 공간을 만들기 위해서 high watermark, low watermark 값을 사용한다. 이 값들은 조금의 여유 공간의 정도를 정의해주는 값으로 memory에 page가 lower watermark 값보다 적게 들어갈 수 있다면, page가 high watermark 개수 만큼 들어갈 수 있을 때까지 page들을 처리한다. 이 처리 과정은 background에서 일어나며 thread가 해주는 것이다. 이 thread을 swap daemon 또는 page daemon이라고 부른다.

또, swap을 할 때 한 page씩 한다고 했지만 사실은 cluster 또는 group이라는 것을 사용해서 swap 해야 하는 page들을 모았다가 한번에 처리한다. 







