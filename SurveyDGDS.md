# Document-Grounded Dialogue System

## Abstract
DS는 학계/산업계에서 여러 적용적 측면에서 쓸모가 많음
그러나, 봇을 기능적(functional)하게 구분짓는 것은 어려운 일.
왜냐, 대화하다 다른 토픽으로 질문을 하거나 추천으로 이어질 지도 모르는 일이기 때문.
function classification은 현재 Trend에 맞지 않음!
현재의 DS는 unstructed document에 중점을 두고 있으며, 본 연구에선 DGDS에 대해
연구할 연구자들을 위해 Research Survey를 제공할 것.

## 1. Introduction
Early DS, 대화에서의 사람의 행동을 모의 실험, 다양한 Turing Test를 거침
- Eliza, a computer program for the study of natural language communication between man and machine.
- Parry, Artificial Paranoia
- Alice, 3. The anatomy of programming languages

그러나 위 시스템들은 제한된 환경에서만 동작, Open-Domain까진 최근까지 힘든 일이었음
- [Challenges in Building Intelligent Open-domain Dialog Systems](https://arxiv.org/pdf/1905.05709.pdf)

이후엔 DARPA와 같은 task-oriented DS에 초점을 맞춘 연구가 진행됨

[Gao et al. 2018](https://www.aclweb.org/anthology/P18-5002/) 연구진은 task-oriented DS와 open-domain DS를 기대 보상을 최대화하는 목표롤 가진 optimal decision-making process로 디자인될 수 있다고 언급. 본 논문의 reference 1, 20, 121, 134 등에서 이들을 통합하려는 시도가 계속되어 옴. 특히 Dodge 연구진의 20번째 논문은 QA, Dialogue, Recommendation 등 5가지 task를 Memory Network로 조사함. (https://arxiv.org/pdf/1511.06931.pdf). 기타 다른 내용도 존재하니 시간나면 읽어볼 것.

Apple의 Siri, MS의 Cortana, Amazon의 Alexa 그리고 Google Assistant에 이르기까지 많은 발전이 있었음. 이들은 명시적인 명령없이도 사용자가 원하는 그리고 편리한 시스템을 제공함. 여러 성공담이 있으나, NLU와 NLG의 어려움으로 인간과의 갭은 아직도 큼

현재의 Trend는 앞서 소개한대로 기능의 통합임. 현재 DS는 기능 분류에 초점을 맞추고 있기 때문에 이러한 트렌드를 받아들이기에는 아직 성숙하지 못함. 특히 본 논문은 unstructured document에 기반을 둔 DS인 DGDS에 초점을 맞춰서 서베이할 예정.

DGDS는 외부 추가지식을 얻을 수 있음.

DGDS = chit-chat with external knowledge + MRC with multi-turn

![fig1](https://user-images.githubusercontent.com/37775784/105834704-274b7f80-600e-11eb-97a5-948d007c3977.PNG)

본 paper에서 아래의 multi-turn DGDS에 대해 연구할 것
- Conversational Reading Comprehension (CRC)
- Document-Based Dialogue (DBD)

대부분의 DGDS dataset은 2018~2019년에 release되어 DL쪽의 모델이 대부분임

## 2. Comparison
![table1](https://user-images.githubusercontent.com/37775784/105835213-e011be80-600e-11eb-9f73-0732eb700761.PNG)

### 2.1 DGDS vs DS
Gao의 연구는 다시 말하지만, 현재의 Trend를 잘 반영하고 있음.

"기능의 통합!"

Gao는 논문에서 대화 시스템을 아래 3개의 항목으로 분류함
- QA Agents
- Task-Oriented Dialogue Agents
- Social Bots

DGDS는 free form response에선 Chit-Chat과 유사하고 QA랑은 대답을 잘 해야한다는 점에서 유사.

### 2.2 CRC vs DBD

#### CRC (Conversational Reading Comprehension)
- 유저가 던지는 질문들은 문서를 기반으로 상호 연결된 정보를 가지고 있음
- 때문에 bot은 현재 질문을 이해하기 위해 대화 이력을 고려해야함
- MNT는 너무 핫함!
- Formulation
  - Document $D$
  - Conversation History $C=\{q_1,a_1,\dots,q_{n-1},a_{n-1}\}$
  - Current Qusetion $q_n$
  - Goal: predict the right answer $a_n$
  - By maximizing the conditional probability $P(a_n|D,C,q_n)$
- MRC와 가장 큰 차이점은 과거의 대화 이력을 사용하는지 아닌지!
- 현재 질문을 정확히 이해하기 위해서 agent는 대화 이력의 coreference와 ellipsis 문제를 해결해야 함!

#### DBD (Document-Based Dialogue)
- Conversation History가 QA 페어가 아닌 utterance $U=\{u_1, u_2, \dots, u_n\}$으로 주어짐
- target은 $P(u_{n+1}|D,U)$로 $u_{n+1}$을 예측하는 것
- CRC와 유사한 것은 과거 대화 이력(conv or utterance)가 요구된다는 점!
- Holl-E, CoQA가 그 예시

#### Difference between the CRC and teh DBD
- Dialog Pattern, Evaluation Methods 등에 차이가 있음
- DBD는 과거 대화 이해를 기반으로 답변을 생성하는데 문서 정보를 사용할 수 있다
- CRC는 보통 과거 대화 이력의 도움으로 현재의 질문을 이해하는 것을 기반으로 답변이 어디에 위치하는지 찾는다.
- 답할 수 없는 질문에 마주하면 CRC는 CANNOTANSWER를 준다!
- DBD는 이와 대조적으로 I don't know와 같이 좀 더 free한 답변을 내줄 수 있다.
- CRC의 경우 정확도로 성능을 평가 가능하지만, DBD의 경우 더 유창하고, 정보가 담경;ㅆ고 일관된 답변을 내놓는지를 확인해야한다.

![table2](https://user-images.githubusercontent.com/37775784/105839776-53b6ca00-6015-11eb-9ea8-a11f94548d8c.PNG)

## 3. Architecture
- joint modeling(JM), knowledge selection(KS): NLU problem
- response generation(RG), evaluation(EV): NLG problem
- memory(MM)은 Deep한 주제. 본 논문에선 위 4개만 다룰 것

![fig3](https://user-images.githubusercontent.com/37775784/105840618-a5138900-6016-11eb-8fba-e5d372579b3f.PNG)

### 3.1 Joint Modeling

### 3.2 Knowledge Selection

### 3.3 Response Generation

### 3.4 Evaluation
