import os
from feedback import generate_feedback
from research import *
from reporting import write_final_report

from dotenv import load_dotenv

load_dotenv()


def main():
    # get query from user
    query = input("어떤 주제에 대해서 리서치하시겠습니까?")

    # 각 항목에 맞는 모델 호출
    feedback_model = "gemini-1.5-flash-8b"
    reasearch_model = "gemini-2.0-flash"

    reporting_model = "gemini-1.5-pro-latest"

    # 추가 적인 질문을 생성하여 연구 방향을 구체화
    print("------------------------------------------------------------1단계: 추가 질문 생성------------------------------------------------------------")
    feedback_questions = generate_feedback(
        query, feedback_model, max_feedbacks=3)
    answers = []
    if feedback_questions:
        print("\n다음 질문에 답변해 주세요")
        for idx, question in enumerate(feedback_questions, start=1):
            answer = input(f"질문 {idx}: {question}\n답변: ")
            answers.append(answer)
    else:
        print("추가 질문이 생성되지 않았습니다.")

    # 초기 질문과 후속 질문 및 답변을 결합
    combined_query = f"초기 질문: {query}\n"
    for i in range(len(feedback_questions)):
        combined_query += f"\n{i+1}. 질문: {feedback_questions[i]}\n"
        combined_query += f"답변: {answers[i]}\n"

    print("최종질문 : \n")
    print(combined_query)

    # 연구 범위 및 깊이를 사용자로부터 입력받음
    try:
        breadth = int(input("연구 범위를 입력하세요 (예: 2): ") or "2")
    except ValueError:
        breadth = 2
    try:
        depth = int(input("연구 깊이를 입력하세요 (예: 2): ") or "2")
    except ValueError:
        depth = 2

    # 심층 연구 수행 (동기적으로 실행)
    print("------------------------------------------------------------2단계: 심층 연구 수행------------------------------------------------------------")
    research_results = deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
        model_name=reasearch_model
    )

    # 연구 결과 출력
    print("\n연구 결과:")
    for learning in research_results["learnings"]:
        print(f"- {learning}")

    # 최종 보고서 생성
    print("------------------------------------------------------------3단계: 최종 보고서 생성------------------------------------------------------------")

    report = write_final_report(
        prompt=combined_query,
        learnings=research_results["learnings"],
        visited_urls=research_results["visited_urls"],
        model_name=reporting_model
    )

    # 최종 보고서 출력 및 파일 저장
    print("\n최종 보고서:\n")
    print(report)

    os.makedirs("reports", exist_ok=True)
    with open("reports/final_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n최종 보고서가 reports/final_report.txt에 저장되었습니다.")


if __name__ == "__main__":
    main()
