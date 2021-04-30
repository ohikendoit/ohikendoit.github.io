---
layout: single
title: "기계 독해를 위한 BERT 언어처리 모델 활용 프로젝트 결과"
date: 2020-04-19
comments: true
categories: 
    [PROJECT]
tags:
    [Natural Language Understanding, Machine Reading Comprehension, BERT, Chatbot, 기계독해 ]
toc: false
publish: true
classes: wide
comments: true
permalink: /nlu_bert_chatbot
---

자연어처리 교육과정을 통해 프로그래밍을  접하게되었고 교육과정을 마무리 지으며 다섯 명의 팀원들과 함께 자연어처리 모델을 활용한 프로젝트를 진행했습니다. 서울시 청년을 위해 관련된 정책 질의응답이 가능한 챗봇을 주제로 서비스까지 구현했습니다.

프로젝트 개요는 아래와 같습니다:

>1. 데이터 인풋을 위한 cdQA 파이프라인 구성과 BERT 모델 처리 과정
>2. 청년 정책 챗봇을 위한 아키텍처 구성 및 문제해결을 위한 접근 방법
>3. 기존 챗봇 빌더와의 차별점 비교 및 프로젝트 결과정리

서론

2019년 서울시 인공지능 챗봇톤 참가를 통해 롯데홈쇼핑 고객서비스 챗봇을 구현한 멘토분과 이야기를 나눌 기회가 있었습니다. 그분 께서 해주신 말씀은 현시점의 자연어 이해(Natural Language Understanding)의 기술력으로는 신뢰도 높은 챗봇 서비스를 구현하기 힘들다는 것 이였습니다. 따라서 상용화된 챗봇의 대부분은 미리 정의한 시나리오를 기반으로 고객과 대화를 나누며, 고객이 입력한 문장을 의도(Intent) 단위로 분석해 답변을 제공하는 룰 매칭을 적용한다고 했습니다.

챗봇은 장기적으로 상황 인지가 가능하고 능동적으로 질의응답을 할 수 있는 방향으로 나아가야 합니다. 따라서 의미추론이 가능한 신경망(Neural Network) 기반의 언어처리모델을 적용해 챗봇을 구현하는 프로젝트를 기획했습니다.

사용자가 입력하는 질문에 대해 답변을 직접 생성(Generative) 모델은 엉뚱한 답을 내는 경우가 많기 때문에 추출(Extractive)모델을 통해 접근했습니다. 자연스러운 대화가 가능한 챗봇을 만들기 위한 핵심기술은 기계 독해(Machine Reading Comprehension)라고 판단했습니다. 기계 독해를 통해 궁극적으로 단어(Lexical) 수준의 정보 색인을 넘어 문법적(Syntactic) 또는 의미적(Semantic) 내용을 바탕으로 정보 습득이 가능하게 되었습니다.

![png](/assets/image/chatbot architecture.jpeg)

*서울시 청년 정책 질의응답 챗봇 아키텍처*

**데이터 인풋을 위한 cdQA 파이프라인 구성과 BERT 모델 처리 과정**

기계 독해 태스크에 최적화된 ETRI BERT 언어처리 모델을 기반으로 챗봇 서비스를 구현했습니다. GPU와 같은 컴퓨팅 자원이 제한적인 환경에서 자체적인 언어 처리 모델 구축에는 어려움이 있습니다. 따라서 ETRI에서 공개한 API를 통해 해당 요소들을 대체함으로써 해결할 수 있었습니다. 2019년 11월 기준, 한국어 자연어 처리 데이터 셋 (KorQuAD)을 활용한 모델 정확도 리더보드에서 ETRI ExoBrain 팀이 개발한 KorBERT 모델이 1위를 유지하고 있습니다. 프로젝트를 위해 한국어 처리에 독보적인 해당 모델을 아키텍처에 적용하였습니다.

사용자가 입력한 문장에 대해 기계 독해 단계에서 진행되는 토큰 임베딩 (Token Embedding)은 한 번의 인풋에 512개 이상의 토큰이 들어간 문장을 처리하지 못하는 메모리 제한이 있습니다. 이에 따라 Doc-to-Sentence 독해를 제공하는 RoBERTa 와 같은 모델이 존재하지만, 좀 더 넓은 범위로부터 데이터를 가져올 수 있는 확장성을 위해 새로운 방식으로 접근했습니다. 사용자의 질문 시 질문과 가장 유사도가 높은 문서와 문단을 호출하는 cdQA (Closed-Domain Question Answering), 특정 도메인에 특화된 데이터 파이프라인을 접목함으로써 자료 입력 크기의 제한이라는 문제를 해결할 수 있었습니다.

![png](/assets/image/cdqa pipeline.jpeg)

*도메인 특정 질의응답을 위한 파이프라인 구성*

**청년 정책 챗봇을 위한 아키텍처 구성 및 문제해결을 위한 접근 방법**

cdQA를 활용한 챗봇은 기존의 서비스와 비교 시 구조상 차이가 있습니다. 서울시 청년 정책에 관련된 사용자의 질문이 들어오면, 형태소 분석기(Morphological Analyzer)를 활용해 해당 질문의 명사와 동사를 세부적으로 추출하게 됩니다. 구문 분석 결과를 바탕으로 구축된 문서 데이터로부터 핵심 단어를 찾는 TF-IDF 혹은 BM25 알고리즘을 활용해 유사도가 가장 높은 원본 문서와 문단을 선택하게 됩니다. 마지막으로, 질문에 대한 답일 가능성이 가장 큰 문단을 메신저 채널을 통해 출력함으로써 사용자의 질의응답에 정확도 높은 답을 제공할 수 있는 구조로 설계되었습니다.

청년 정책 챗봇 아키텍처에 적용된 cdQA 파이프라인은 크게 문서검색을 진행하는 Retriever 와 기계 독해를 진행하는 Reader, 두 가지 부분으로 나누어져 있습니다. TF-IDF와 BM25 알고리즘 중 하나를 선택해서 해당 질문과 가장 유사도가 높은 문서를 선택하게 됩니다. 단어 빈도에 있어서 BM25 알고리즘은 TF-IDF 보다 특정 값으로 수렴하고, 문맥상 큰 의미가 없는 불용어(Stopword)가 검색 점수에 영향을 덜 미친다는 장점이 있습니다. 특히, 문서의 평균 길이(AVGDL)를 계산에 사용함으로써 문서의 길이가 검색 점수에 영향을 덜 미치는 강점이 있기 때문에 최종적으로는 BM25 알고리즘을 통해 문사 유사도를 측정하였습니다.

```python
def Message():
   content = extract_content()
       global best_idx_scores
       if list(list(retriever.predict(ETRI_POS_Tagging(content)).values())[0])[0]>=1. or not best_idx_scores:
           best_idx_scores = retriever.predict(ETRI_POS_Tagging(content))
           if list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(ETRI_POS_Tagging(content)).values())[0])[0]>=1.:
               changed_text = text_tranform(''.join(df.loc[list(best_idx_scores.keys())[0]]['paragraphs']))
               dataSend=changed_text
               best_idx_scores = ''
               return jsonify(make_query(dataSend))
           elif list(best_idx_scores.keys())[0]<8 and list(list(retriever.predict(ETRI_POS_Tagging(content)).values())[0])[0]<1.:
               dataSend=ETRI_wiki(content)
               best_idx_scores = ''
               return jsonify(make_query(dataSend))
           cdqa_pipeline.fit_retriever(df.loc[best_idx_scores.keys()].head(1))
       cdqa_query,validity=ETRI_korBERT(' '.join(list(df.loc[best_idx_scores.keys()].head(1)['paragraphs'])[0]),content)
       if float(validity) <= 0.2 :
           dataSend=ETRI_wiki(content)
           best_idx_scores = ''
       else :
           dataSend_temp = cdqa_pipeline.predict(cdqa_query)
           dataSend=dataSend_temp[2]
   return jsonify(make_query(dataSend))
```
*챗봇에 활용된 메인 함수*

챗봇에 활용된 메인 함수 동작 원리 및 구조는 아래와 같습니다:

>1. 사용자의 질문이 Content 변수로 함수에 입력됩니다
>2. 형태소 분석기를 통해 구문 분석된 질문과 1) 데이터베이스 상 문사 중 유사도가 있는지 2) 함수에 처음 들어온 케이스인지 판단합니다.
>3. 함수를 통해 산출된 best_idx_scores 값에 따라 1) 9 미만의 값일 경우 해당 리스트 (카테고리별 정책목록)을 반환합니다. 2) 값이 9 이상일 경우 유사도가 있는 문서를 선택하고 본 함수로 재입력합니다
>4. 함수에 다시 들어온 경우 데이터베이스 상 문서에 대한 유사도를 계산 (0에서 최대 20까지의 값)을 통해 가장 높은 스코어를 받은 문장이 질문과 함께 BERT 모델로 전달됩니다
>5. 유사도 값이 1 미만일 경우 ETRI Wiki QA API를 통해 위키백과에 기반한 일반응답 결과를 반환합니다

**기존 챗봇 빌더와의 차별점 비교 및 프로젝트 결과정리:**

![png](/assets/image/youth policy chatbot.jpeg)

프로젝트 결과물은 카카오톡 플러스친구에서 ‘청년정책봇' 계정을 추가함으로써 사용해보실 수 있습니다.

“청년 금융 정책 알려줘", “면접을 위한 정장 무료로 대여하고 싶어", “희망 두 배 청년 통장 지원대상 알려줘" 같은 질문에 정확도 높은 응답을 직접 확인해 볼 수 있습니다.

지난 10월에 열린 kt NexR 빅데이터 컨퍼런스에서 LG 전자 챗봇 프로젝트 매니저는 연사를 통해 장기적으로 ‘사람 같은' 챗봇이 각광받을 전망이라고 밝혔습니다. 챗봇의 전반적인 개발 방향은 시나리오에 기반한 스크립트 기술에서(Scripted Chatbots), 의도 인식(Intent Recognizers) 및 가상 도우미(Virtual Agents)를 거쳐 자연스러운 대화가 가능한 방향(Human-like Advisor)으로 나아가고 있습니다. 챗봇이 고객 서비스 및 응대에 대한 역할을 하기 위해서는 실제 대화를 하는 듯한 자연어처리 기술이 필수입니다. 사용자의 발화 의도에 기반해 특정 주제에 답변을 줄 수 있는 cdQA 기반 챗봇 아키텍처는 자연스러운 대화를 위한 챗봇 설계에 적합한 시작점이 될 것입니다.

청년 정책봇 구성에 포함된 cdQA 파이프라인은 현재 상용화된 챗봇 서비스의 대부분에 비교해 크게 두 가지 장점이 있습니다:
>1. 확장 및 지속 가능성 — 기계 독해 기능 면에서 API를 사용함으로써 언어 처리 모델에 대한 추가적인 학습이 필요하지 않아 서비스의 자동화를 달성하기가 쉽습니다. 청년 정책의 경우 서울시 공식 홈페이지의 주기적인 웹 크롤링을 통해 수정된 부분만 데이터베이스에 추가하면 되기에, 소프트웨어 유지 보수에 필요한 자원을 최소화 할 수 있습니다.
>2. 딥러닝 모델 발전에 따른 성능향상 — BERT 언어 처리 모델 성능은 KorQuAD 리더보드에서 볼 수 있듯이 지속해서 발전되고 있습니다. API를 통해 모델의 성능이 향상될 때마다 챗봇의 연산 능력도 같이 발전될 수 있다는 장점이 있습니다. 경량화 모델로 대체할 경우 속도향상의 여지가 남아있습니다.

**마치며**

본 포스트의 주된 목적은 3개월이라는 짧지도 길지도 않은 시간 동안 여섯 명이 함께 동고동락하며 작업했던 내용을 공유하고 기록으로 남기기 위함입니다. 해당 글이 챗봇과 관련된 자연어처리 프로젝트를 구상하거나 진행하고 계시는 분들께 조금이나마 도움이 되었으면 합니다. 궁금하신 점이나 피드백이 있으면 ohikendoit@gmail.com으로 연락해주세요. 끝으로 고생 많이 한 준형이 형, 의형이 형, 원석이, 승연이, 강빈이 한테 고맙다는 말을 전하고 싶습니다.