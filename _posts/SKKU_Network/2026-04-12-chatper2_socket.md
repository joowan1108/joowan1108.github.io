---
layout: single
title: Chatper 2 Socket Interface Programming 정리
categories: SKKU_Network
tag: [SKKU]
author_profile: false
sidebar:
    nav: "counts"
toc: true
toc_sticky: true
toc_label: Table of Contents
use_math: true
---


# Socket Interface Programming

Socket은 5개의 fields을 가진다

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg176_left.png)

- Family: 
    사용하는 Family protocol을 정의한다. ex) PF_INET (IPv4 internet protocol), PF_INET6 (IPv6 internet protocol)

- Type
    4가지 socket 종류 중 하나 정의. ex) SOCK_STREAM (for TCP), SOCK_DGRAM (for UDP), SOCK_SEQPACKET (for SCTP), and IP (for applications that directly use the services of IP)

- Protocol
    Family에서 특정 protocol 값 ex)TCP/IP에서는 0

- Local Socket Address
    Length는 주소의 길이, family는 address family이다 ex) AF_INET 

- Remote Socket Address
    Local socket address와 동일한 구조이다

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg176_right.png)


## UDP

Client와 server는 각각 하나의 socket을 사용하고 datagram을 주고 받는다.

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg177.png)

Server는 먼저 passive open을 시작해서 client가 연결을 요청할 때까지 기다린다.

Client는 필요시에 active open을 시작하여 연결을 시작한다.

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg178.png)

**CODE**

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg179.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg180.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg181.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg182.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg183.png)


## TCP

Connection oriented protocol으로 data을 보내기 전에 client와 server 사이에 연결이 활성화되어야 한다.

TCP 통신은 반복적 (client 하나씩 응답)일 수도 있고 concurrent에도 가능하다 (한번에 여러 client 동시 응답)

TCP server는 UDP와 다르게 두 개의 socket을 사용한다.
1. Listen to socket: Client의 요청을 기다리는 socket. 연결을 establish하는 역할을 한다 
2. Socket: Client와 실제로 data를 주고받는 socket. 연결이 establish 된 다음에 생성된다


![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg185.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg186.png)

**Code**

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg186.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg187.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg188.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg189.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg190.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg191.png)

![joowan1108]({{site.url}}/images/SKKU_Network/chp2pg192.png)


# Practice Problems

1. What is the purpose of Listen Socket in TCP?
- For listening and establishing connection from the client
2. The UDP server usually only needed one socket, whereas the TCP server needed at least two sockets. Why? If the TCP server were to support n simultaneous connections, each from a different client host, how many sockets would the TCP server need?
- With the UDP server, there is no listening (welcoming) socket, and all data from different clients enters the server through this one socket.
- With the TCP server, there is a welcoming socket, and each time a client initiates a connection to the server, a new socket is created. Thus, to support n simultaneous connections, the server would need n+1 sockets


