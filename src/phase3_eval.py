# src/phase3_eval.py
import json
import os
import time
import sys
from dotenv import load_dotenv

load_dotenv()

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(__file__))

import google.generativeai as genai
from phase2_agent import LangGraphResearchAgent

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel('gemini-flash-lite-latest')

CONFIGURATIONS = {
    "baseline": {"max_retries": 0},
    "full_agent": {"max_retries": 2},
    "no_planner": {"max_retries": 1},
    "no_reflector": {"max_retries": 0},
    "no_verifier": {"max_retries": 1}
}

QUESTIONS_FILE = "./eval/questions.jsonl"
PREDICTIONS_DIR = "./predictions"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def load_questions(filepath):
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def run_evaluation():
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: Could not find {QUESTIONS_FILE}.")
        return

    questions = load_questions(QUESTIONS_FILE)
    print(f"Loaded {len(questions)} evaluation questions.")

    for config_name, params in CONFIGURATIONS.items():
        output_file = os.path.join(PREDICTIONS_DIR, f"{config_name}.jsonl")
        results = []
        start_index = 0

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            start_index = len(results)
            if start_index >= len(questions):
                print(f"\nSkipping '{config_name}' (100% complete).")
                continue
            else:
                print(f"\nResuming '{config_name}' from Question {start_index + 1}...")

        print(f"\n==========================================")
        print(f" STARTING RUN: {config_name.upper()}")
        print(f"==========================================")

        agent = LangGraphResearchAgent(
            model=model,
            max_retries=params.get("max_retries", 2)
        )

        for index in range(start_index, len(questions)):
            q = questions[index]
            question_id = q.get("_id", str(index))
            question_text = q.get("question", "")
            print(f"\n[Question {index + 1}/{len(questions)}] {question_text[:70]}...")

            try:
                answer = agent.run(question_text)
            except Exception as e:
                print(f"Error on question {question_id}: {e}")
                answer = "Error generating answer due to API limits or system crash."
                time.sleep(10)

            results.append({
                "_id": question_id,
                "answer": answer
            })

            with open(output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")

            time.sleep(2)

    print("\nPhase 3 Evaluation complete! Results saved in predictions/.")

if __name__ == "__main__":
    run_evaluation()
