# DeepResearch 

## Overview
Deep Research는 기술 리서치를 자동화하는 도구입니다. 사용자가 질문을 입력하면 관련 정보를 크롤링하고, 추가적인 질문을 통해 보다 정확한 리포트를 생성합니다. 연구 논문, 기술 동향, 시장 조사 등 다양한 분야의 리서치에 활용할 수 있습니다.

## 주요 기능
- 자동 크롤링: Firecrawl을 활용해 웹에서 질문에 대한 최신 정보를 검색
- LLM 기반 문맥 분석: Google Gemini 모델을 활용해 질문을 이해하고 추가 질문을 생성
- 대화형 리서치: 유저와의 상호작용을 통해 연구 범위를 구체화
- 마크다운 보고서 생성: 최종적으로 구조화된 리포트 출력

## 기술 스텍
- Python 3.12.9
- Firecrawl
- LangChain
- Google Gemini

## 아키텍처
- 사용자가 **질문(string)**을 입력
- Firecrawl을 이용해 관련 정보 크롤링
- LangChain + Gemini가 질문을 분석하고 추가 질문을 생성
- 유저의 추가 응답을 바탕으로 최적의 리서치 방향 설정
- 크롤링 데이터를 LLM이 분석하여 최종 보고서(마크다운 형식) 생성

## 설치 및 실행 방법
### 프로젝트 클론
```bash
git clone https://github.com/taejongK/like-DeepResearch.git
cd deep-research
```
### 환경 설정
- 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```
### 실행 방법
```bash
python main.py
```
- 이후 터미널에서 질문을 입력하고, 이어지는 추가 요청에 답변하면 자동으로 리서치가 진행됩니다.

## 성능 평가 방법
현재 별도의 정량적 평가 방법은 없지만, 향후 다음과 같은 방식으로 평가를 진행할 수 있습니다.

- 정확도 평가: 리서치 결과와 논문/공식 문서 비교

- 사용자 피드백: 리포트 품질에 대한 사용자 평가 (Likert Scale, Open-ended feedback)

## 한계점 및 향후 개선 방향
- 현재 Firecrawl 사용으로 인해 비용이 발생 (추후 직접 구현 계획)

- Gemini 모델을 로컬 모델로 변경 가능 (추후 비용 절감을 위해 Hugging Face 모델 지원 고려)

- 데모 UI 없음 (Streamlit 또는 Gradio 기반의 웹 UI 추가 예정)

