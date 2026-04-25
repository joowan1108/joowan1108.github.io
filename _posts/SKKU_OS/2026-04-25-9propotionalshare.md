# Propotional Share Scheduler

지금까지 배운 scheduler는 turnaround time과 response time을 최소화하기 위한 scheduler였다. 하지만 Proportional Share Scheduler는 전혀 다른 성격을 지닌다.

Proportional Share Scheduler는 오직 모든 process가 공평한 기회를 가질 수 있도록 하는 것에 집중한다.

## Lottery Scheduling

Lottery scheduling은 Proportional share의 성격을 잘 보여준다. 가장 기본적인 아이디어는 필요한 시간이 긴 process는 더 많은 로또 티켓을 주어 로또에 당첨되는 확률을 높여주는 것이다. 

즉, ticket 수는 자원을 얼마동안 써야하는지에 따라 달라지는 것이다. 만약 100개의 ticket이 있는데 A는 75개, B는 25개 있다면 A가 75%의 시간동안 자원을 받고 B가 나머지 25%의 시간동안 자원을 받게 하는 것이다.

Lottery scheduling이 수행되기 위해서는 Scheduler는 전체 ticket 수를 알고 있어야 하며 매 interrupt마다 당첨 ticket을 뽑아 당첨 번호가 적힌 ticket을 가지고 있는 process가 다음에 실행될 process가 되는 것이다. 

Lottery scheduling은 **ticket currrency**라는 개념도 사용한다. User A와 B 모두 100개의 ticket을 갖고있는데 User A는 2개의 job을 실행시켜야 하기에 자기의 ticket을 1000개로 만들어서 각 job에게 500개씩 주었다고 하자. 이렇게 ticket 수를 늘리더라도 결국 User A는 ticket의 가치를 $\frac {1} {10}$으로 줄인 것이기 때문에 최종 scheduling을 할 때는 A의 각 job은 50의 ticket을 가진 것과 동일하다.

$\rightarrow$ Ticket currency의 유용성은 각 job가 갖게 될 비율을 더 세분화할 수 있다는 점이다.

**Ticket Transfer**: Process끼리 합의 하에 ticket을 주고 받는 행위를 의미한다. 이는 client server paradigm을 생각하면 편하다. Client가 server에 요청을 했을 때, client는 server가 빨리 CPU을 받아서 자기의 요청으 처리해줬으면 좋으므로 자기의 ticket을 빌려줄 수 있게 되는 것이다.

**Ticket inflation**: 특정 process가 현재 많이 중요하다는 가정 하에 그 process가 가진 가치를 증폭시켜서 CPU의 할당을 받을 가능성을 높일 수 있다.

