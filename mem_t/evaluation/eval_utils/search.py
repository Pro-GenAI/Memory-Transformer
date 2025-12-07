import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template

# from openai import OpenAI
from tqdm import tqdm

from mem0 import MemoryClient
from openai_python_cache import create_chat_completion

from mem_t.evaluation.prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH

load_dotenv()

if os.getenv("MEM0_BASE_URL"):
    MemoryClient._validate_api_key = lambda *args, **kwargs: "test"

# openai_client = OpenAI()


class MemorySearch:
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
    ):
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            host=os.getenv("MEM0_BASE_URL"),
            org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            project_id=os.getenv("MEM0_PROJECT_ID"),
        )
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        memories = {}
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("Searching with graph")
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()

        # Optionally apply a small lexical rerank after mem0 returns results.
        use_lex_rerank = os.getenv("USE_LEX_RERANK", "1") != "0"
        if not self.is_graph:
            raw_semantic = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": float(memory["score"]),
                }
                for memory in memories
            ]
            if use_lex_rerank and raw_semantic:
                try:
                    q_tokens = [t.lower() for t in query.split() if t]
                    if q_tokens:
                        for m in raw_semantic:
                            mem_tokens = set([t.lower() for t in m["memory"].split() if t])
                            matches = sum(1 for t in q_tokens if t in mem_tokens)
                            lexical_boost = 0.25 * (matches / max(1, len(q_tokens)))
                            m["score"] = m["score"] + lexical_boost
                except Exception:
                    pass
            # sort by adjusted score and round for output
            raw_semantic.sort(key=lambda x: -x["score"])
            semantic_memories = [
                {"memory": m["memory"], "timestamp": m["timestamp"], "score": round(m["score"], 2)}
                for m in raw_semantic
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]  # type: ignore
            ]
            graph_memories = [
                {
                    "source": relation["source"],
                    "relationship": relation["relationship"],
                    "target": relation["target"],
                }
                for relation in memories["relations"]  # type: ignore
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(
        self, speaker_1_user_id, speaker_2_user_id, question, answer, category
    ):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = (
            self.search_memory(speaker_1_user_id, question)
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = (
            self.search_memory(speaker_2_user_id, question)
        )
        if speaker_1_memories or speaker_2_memories:
            search_1_memory = [
                f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories
            ]
            search_2_memory = [
                f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories
            ]

            template = Template(self.ANSWER_PROMPT)
            answer_prompt = template.render(
                speaker_1_user_id=speaker_1_user_id.split("_")[0],
                speaker_2_user_id=speaker_2_user_id.split("_")[0],
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
                speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
                question=question,
            )

            t1 = time.time()
            response = create_chat_completion(
                model=os.getenv("OPENAI_MODEL", ""),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0,
            )
            t2 = time.time()
            response_time = t2 - t1
        else:
            response = "No memory for "
            if not speaker_1_memories:
                response += "Speaker 1"
                if not speaker_2_memories:
                    response += "and Speaker 2"
            elif not speaker_2_memories:
                response += "Speaker 2"
            response_time = 0
        return (
            (
                response.choices[0].message.content  # type: ignore
                if response and hasattr(response, "choices")
                else response or ""
            ),
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(
            speaker_a_user_id, speaker_b_user_id, question, answer, category
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(
            enumerate(data), total=len(data), desc="Processing conversations"
        ):
            if idx > 1:  # Limit to first 2 conversations for testing
                break
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa,
                total=len(qa),
                desc=f"Processing questions for conversation {idx}",
                leave=False,
            ):
                result = self.process_question(
                    question_item, speaker_a_user_id, speaker_b_user_id
                )
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print("Processing complete. Results saved to", self.output_path)
        print("Results:", self.results[:2])  # Print first 2 conversations for verification

    def process_questions_parallel(
        self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1
    ):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_question, qa_list),
                    total=len(qa_list),
                    desc="Answering Questions",
                )
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
