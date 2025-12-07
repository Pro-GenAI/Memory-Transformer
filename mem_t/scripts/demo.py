from mem_t.neuro_mem import HierarchicalMemoryModel


def run_basic_demo():
    """Basic memory operations demo."""
    print("=" * 40)
    print("BASIC MEMORY OPERATIONS DEMO")
    print("=" * 40)

    model = HierarchicalMemoryModel()

    # Add various types of memories
    memories = [
        ("I love eating pizza with extra cheese", "food", "short"),
        ("My favorite color is blue", "personal", "short"),
        ("The meeting with John is at 3 PM tomorrow", "schedule", "short"),
        ("John's email is john@company.com", "contact", "long"),
        ("I need to buy groceries: milk, bread, eggs", "todo", "short"),
        ("Alice's phone number is +1-555-0123", "contact", "long"),
        ("I visited Paris last summer and loved the Eiffel Tower", "travel", "long"),
        ("Remember to call mom on Sunday", "personal", "short"),
        ("Python is my favorite programming language", "tech", "long"),
        ("The restaurant 'Bella Vista' has amazing Italian food", "food", "long"),
    ]

    print("Adding memories...")
    for text, category, tier in memories:
        key = model.add_memory(text, tier=tier, meta={"category": category})
        print(f"✓ Added: {text[:50]}... (key: {key})")

    print(
        f"\nTotal memories: {len(model.short_term)} short-term, {len(model.long_term)} long-term"
    )

    # Test queries
    queries = [
        "What food do I like?",
        "John's contact information",
        "programming",
        "phone number",
        "meeting schedule",
    ]

    print("\n" + "=" * 40)
    print("QUERY RESULTS")
    print("=" * 40)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = model.query(query, top_k=3)
        for i, (score, item) in enumerate(results, 1):
            category = item.meta.get("category", "unknown") if item.meta else "unknown"
            print(f"  {i}. [{score:.3f}] {item.text} (tier: {category})")

    print()
    print("=" * 60)
    print("BASIC DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_basic_demo()
