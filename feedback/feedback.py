from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


"""연구 방향을 명확히 하기 위한 후속 질문을 생성합니다."""
class GenerateQuestions(BaseModel):
    questions: List = Field(..., description="The answer to the user's question")

def generate_feedback(query: str, model_name: str, max_feedbacks: int = 3) -> List[str]:
    response_llm = ChatGoogleGenerativeAI(model=model_name)

    response_prompt = PromptTemplate.from_template(
        """
        Given the following query from the user, ask some follow up questions to clarify the research direction. Return a maximum of ${max_feedbacks} questions, but feel free to return less if the original query is clear.
        ask the follow up questions in korean
        <query>${query}</query>`
        # Answer:\n\n
        {format_instructions}
        """
    )
    json_parser = JsonOutputParser(pydantic_object=GenerateQuestions)

    response_prompt = response_prompt.partial(format_instructions=json_parser.get_format_instructions())

    response_chain = response_prompt | response_llm | json_parser

    response = response_chain.invoke({"query": query, "max_feedbacks": max_feedbacks})
    
    try:
        if response is None:
            print("오류: JSON_llm이 None을 반환했습니다.")
            return []
        questions = response['questions']
        print(f"주제 '{query}'에 대한 후속 질문 {len(questions)}개 생성됨")
        print(f"생성된 후속 질문: {questions}")
        return questions
    except Exception as e:
        print(f"오류: JSON 응답 처리 중 문제 발생: {e}")
        print(f"원시 응답: {response}")
        print(f"오류: 쿼리 '{query}'에 대한 JSON 응답 처리 실패")
        return []
    
if __name__ == "__main__":
    query = "What are the main causes of the American Civil War?"
    model_name = "gemini-1.5-flash-8b"
    max_feedbacks = 3
    questions = generate_feedback(query, model_name, max_feedbacks)
    print(questions)