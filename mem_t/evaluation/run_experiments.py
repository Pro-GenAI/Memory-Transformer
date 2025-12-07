import argparse
import os

from mem_t.evaluation.eval_utils.add import MemoryADD
from mem_t.evaluation.eval_utils.search import MemorySearch


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique_type", choices=["mem0"], default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=["add", "search"], default="add", help="Method to use")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")

    args = parser.parse_args()

    # Add your experiment logic here
    print(f"Running experiments with technique: {args.technique_type}")
    data_path = "mem_t/evaluation/locomo10.json"

    if not os.path.exists(data_path):
        import urllib.request
        print("Downloading locomo10.json dataset...")
        urllib.request.urlretrieve(
            "https://github.com/snap-research/locomo/raw/refs/heads/main/data/locomo10.json", data_path
        )

    if args.technique_type == "mem0":
        if args.method == "add":
            memory_manager = MemoryADD(data_path=data_path, is_graph=args.is_graph)
            memory_manager.process_all_conversations()
        elif args.method == "search":
            output_file_path = os.path.join(
                args.output_folder,
                f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}_graph_{args.is_graph}.json",
            )
            memory_searcher = MemorySearch(output_file_path, args.top_k, args.filter_memories, args.is_graph)
            memory_searcher.process_data_file(data_path)
    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()
