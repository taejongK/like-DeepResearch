import os
import sys
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import system_prompt

from dotenv import load_dotenv

load_dotenv()


class SearchResult(BaseModel):
    url: str
    markdown: str
    description: str
    title: str


def firecrawl_search(query: str, timeout: int = 15000, limit: int = 5) -> List[SearchResult]:
    """
    Firecrawl 검색 API를 호출하여 결과를 반환하는 동기 함수
    TODO: 이 부분만 직접 구현하면 완전한 무료 deep research가 가능
    """
    try:
        app = FirecrawlApp()
        response = app.search(
            query=query,
            params={"timeout": timeout, "limit": limit,
                    "scrapeOptions": {"formats": ['markdown']}}
        )
        return response.get("data", [])
    except Exception as e:
        print(f"Firecrawl 검색 오류: {e}")


class SerpQuery(BaseModel):
    query: str
    research_goal: str


class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery] = Field(
        ..., description="A list of SERP queries to be used for research")


def generate_serp_queries(
    query: str,
    model_name: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """
    사용자의 쿼리와 이전 연구 결과를 바탕으로 SERP 검색 쿼리를 생성합니다.
    json_parser를 사용해 구조화된 JSON을 반환합니다.
    """
    prompt = """
        ### System:
        {system_prompt}
        
        ### Instruction:
        다음 사용자 입력을 기반으로 연구 주제를 조사하기 위한 SERP 검색 쿼리를 생성하세요.
        JSON 객체를 반환하며, 'queries' 배열 필드에 {num_queries}개의 검색 쿼리를 포함해야 합니다 (쿼리가 명확할 경우 더 적을 수도 있음).
        각 쿼리 객체에는 'query'와 'research_goal' 필드가 포함되어야 하며, 각 쿼리는 고유해야 합니다: 
        
        ### Input:
        {query}
    """

    if learnings:
        prompt += f"\n\n다음은 이전 연구에서 얻은 학습 내용입니다. 이를 활용하여 더 구체적인 쿼리를 생성하세요: {' '.join(learnings)}"

    prompt += "\n\n### Answer:\n\n{format_instructions}"

    prompt = PromptTemplate.from_template(prompt)

    json_parser = JsonOutputParser(pydantic_object=SerpQueryResponse)
    sys_prompt = system_prompt()

    prompt = prompt.partial(system_prompt=sys_prompt,
                            format_instructions=json_parser.get_format_instructions())

    llm = ChatGoogleGenerativeAI(model=model_name)

    chain = prompt | llm | json_parser
    response_json = chain.invoke({"query": query, "num_queries": num_queries})
    try:
        result = SerpQueryResponse.model_validate(response_json)
        queries = result.queries if result.queries else []
        print(f"리서치 주제에 대한 SERP 검색 쿼리 {len(queries)}개 생성됨")
        return queries[:num_queries]
    except Exception as e:
        print(f"오류: generate_serp_queries에서 JSON 응답을 처리하는 중 오류 발생: {e}")
        print(f"원시 응답: {response_json}")
        print(f"오류: 쿼리 '{query}'에 대한 JSON 응답 처리 실패")
        return []


class ResearchResult(BaseModel):
    learnings: List[str]
    visited_urls: List[str]


class SerpResultResponse(BaseModel):
    learnings: List[str]
    followUpQuestions: List[str]


def process_serp_result(
    query: str,
    search_result: List[SearchResult],
    model_name: str,
    num_learnings: int = 5,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """
    검색 결과를 처리하여 학습 내용과 후속 질문을 추출합니다.
    json_parser를 사용해 구조화된 JSON을 반환합니다.
    """
    contents = [
        item.get("mardown", "").strip()[:25000]
        for item in search_result if item.get("markdown")
    ]
    json_parser = JsonOutputParser(pydantic_object=SerpResultResponse)

    contents_str = "".join(f"<내용>{content}</내용>" for content in contents)
    prompt = PromptTemplate(
        input_variables=["query", "num_learnings", "num_follow_up_questions"],
        template="""
        ### System:
        {system_prompt}
        
        ### Instruction:
        다음은 쿼리 <쿼리>{query}</쿼리>에 대한 SERP검색 결과입니다.
        이 내용을 바탕으로 학습 내용을 추출하고 후속 질문을 생성하세요.
        JSON 객체로 반환하며, 'learnings' 및 'followUpQuestions' 키를 포함한 배열을 반환하요.
        각 학습 내용은 고유하고 간결하며 정보가 풍부해야 합니다. 최대 {num_learnings}개의 학습 내용과
        {num_follow_up_questions}개의 후속 질문을 포함해야 합니다.\n\n
        <검색결과>{contents_str}</검색결과>
        
        ### Answer:
        {format_instructions}
        """
    ).partial(
        system_prompt=system_prompt(),
        contents_str=contents_str,
        format_instructions=json_parser.get_format_instructions(),
    )

    llm = ChatGoogleGenerativeAI(model=model_name)
    chain = prompt | llm | json_parser
    response_json = chain.invoke({"query": query,
                                  "num_learnings": num_learnings,
                                  "num_follow_up_questions": num_follow_up_questions
                                  })
    try:
        result = SerpResultResponse.model_validate(response_json)
        return {
            "learnings": result.learnings,
            "followUpQuestions": result.followUpQuestions[: num_follow_up_questions],
        }
    except Exception as e:
        print(f"오류: process_serp_result에서 JSON 응답을 처리하는 중 오류 발생: {e}")
        print(f"원시 응답: {response_json}")
        return {"learnings": [], "followUpQuestions": []}


def deep_research(
    query: str,
    breadth: int,
    depth: int,
    model_name: str,
    learnings: Optional[List[str]] = None,
    visited_urls: Optional[List[str]] = None,
) -> ResearchResult:
    """
    주제를 재귀적으로 탐색하여 SERP 쿼리를 생성하고, 검색 결과를 처리하며,
    학습 내용과 방문한 URL을 수집합니다.
    """
    learnings = learnings or []
    visited_urls = visited_urls or []

    print(f"--------Deep Research Start--------")
    print(f"<주제> \n {query} \n </주제>")

    serp_queries = generate_serp_queries(
        query=query,
        model_name=model_name,
        num_queries=breadth,
        learnings=learnings
    )
    print(
        f" ------------ 해당 <주제>에 대해서 생성된 검색 키워드 ({len(serp_queries)}개 생성)------------")
    print(f" {serp_queries} \n")

    for index, serp_query in enumerate(serp_queries, start=1):

        result: List[SearchResult] = firecrawl_search(serp_query.query)
        new_urls = [item.get("url") for item in result if item.get("url")]
        serp_result = process_serp_result(
            query=serp_query.query,
            search_result=result,
            model_name=model_name,
            num_learnings=5,
            num_follow_up_questions=breadth
        )
        print(f"  - 의 {index}번째 검색 키워드 ({serp_query.query})에 대한 조사 완료")
        print(f"  - 조사완료된 URL들:")
        for url in new_urls:
            print(f"    • {url}")
        print()
        print(
            f"  - 조사로 얻은 학습 내용 ({len(serp_result['learnings'])}개 생성) : \n {serp_result['learnings']} \n")

        all_learnings = learnings + serp_result["learnings"]
        all_urls = visited_urls + new_urls
        new_depth = depth - 1
        new_breadth = max(1, breadth // 2)

        if new_depth > 0:
            next_query = (
                f"이전 연구목표: {serp_query.research_goal} \n"
                f"후속 연구방향: {' '.join(serp_result['followUpQuestions'])}"
            )

            # 증가된 시도 획수로 재귀 호출
            sub_result = deep_research(
                query=next_query,
                breadth=new_breadth,
                depth=new_depth,
                model_name=model_name,
                learnings=all_learnings,
                visited_urls=all_urls
            )

            learnings = sub_result["learnings"]
            visited_urls = sub_result["visited_urls"]
        else:
            learnings = all_learnings
            visited_urls = all_urls

        return {
            "learnings": list(set(learnings)),
            "visited_urls": list(set(visited_urls))
        }

if __name__ == "__main__":
    query = "미국 남북 전쟁의 주요 원인은 무엇인가요?"
    model_name = "gemini-1.5-flash-8b"
    breadth = 3
    depth = 2
    result = deep_research(query, breadth, depth, model_name)
    print(f"최종 학습 내용: {result['learnings']}")
    print(f"방문한 URL들: {result['visited_urls']}")
    print(f"--------Deep Research End--------")
    print()