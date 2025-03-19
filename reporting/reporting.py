from typing import List
from utils import system_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def write_final_report(
    prompt,
    learnings: List[str],
    visited_urls: List[str],
    model_name: str,
) -> str:
    """
    모든 연구 결과를 바탕으로 최종 보고서를 생성합니다.
    llm을 사용하여 마크다운 보고서를 얻습니다.
    """
    learning_string = ("\n".join(
        [f"<learning>\n{learning}\n</learning>" for learning in learnings])).strip()[:150000]

    user_prompt = PromptTemplate(
        input_variables=["prompt"],
        template="""
        ### System:
        {system_prompt}
        
        ### Instruction:
        사용자가 제시한 다음 프롬프트에 대해, 리서치 결과를 바탕으로 최종 보고서를 작성하세요.
        마크다운 형식으로 상세한 보고서(6,000자 이상)를 작성하세요.
        리서치에서 얻은 모든 학습 내용을 포함해야 합니다:\n\n
        <prompt>{prompt}</prompt>\n\n
        
        다음은 리서치를 통해 얻은 모든 학습 내용입니다:\n\n<learnings>\n{learnings_string}</learnings>
        """
    ).partial(
        system_prompt=system_prompt(),
        learnings_string=learning_string,
    )
    llm = ChatGoogleGenerativeAI(model=model_name)
    chain = user_prompt | llm | StrOutputParser()
    try:
        report = chain.invoke({"prompt": prompt})
        urls_section = "\n\n## 출처\n\n" + \
            "\n".join(f"- {url}" for url in visited_urls)
        return report + urls_section
    except Exception as e:
        print(f"Error genrating report: {e}")
        return "Error generating report"
