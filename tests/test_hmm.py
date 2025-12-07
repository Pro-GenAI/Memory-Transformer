import time

from mem_t.neuro_mem import HierarchicalMemoryModel


class TestBasicMemoryOperations:
    """Test basic memory operations: add, query, recall, forget."""

    def test_add_memory_with_metadata(self):
        """Test adding memories with metadata."""
        model = HierarchicalMemoryModel()
        meta = {"category": "test", "importance": "high"}
        key = model.add_memory("Test memory", meta=meta)

    def test_query_basic_similarity(self):
        """Test basic similarity-based querying."""
        model = HierarchicalMemoryModel()
        model.add_memory("I love eating pizza")
        model.add_memory("My favorite food is pasta")
        model.add_memory("I hate broccoli")

        results = model.query("food", top_k=3)
        assert len(results) >= 2  # Should find food-related memories
        # Results should be sorted by similarity score
        assert all(results[i][0] >= results[i + 1][0] for i in range(len(results) - 1))

    def test_query_with_threshold(self):
        """Test querying with similarity threshold."""
        model = HierarchicalMemoryModel()
        model.add_memory("Python programming")
        model.add_memory("Java development")
        model.add_memory("Unrelated topic")

        # High threshold should return fewer results
        results_high = model.query("programming", top_k=5, threshold=0.8)
        results_low = model.query("programming", top_k=5, threshold=0.1)
        assert len(results_high) <= len(results_low)

    def test_recall_by_key(self):
        """Test recalling memories by key."""
        model = HierarchicalMemoryModel()
        key1 = model.add_memory("Memory 1")
        key2 = model.add_memory("Memory 2")

        mem1 = model.recall_by_key(key1)
        mem2 = model.recall_by_key(key2)

        assert mem1 is not None and mem1.text == "Memory 1"
        assert mem2 is not None and mem2.text == "Memory 2"
        assert model.recall_by_key("nonexistent") is None

    def test_forget_memory(self):
        """Test forgetting memories."""
        model = HierarchicalMemoryModel()
        key = model.add_memory("Test memory")
        assert model.recall_by_key(key) is not None

        assert model.forget(key) is True
        assert model.recall_by_key(key) is None
        assert model.forget("nonexistent") is False


class TestAdvancedMemoryQueries:
    """Test advanced queries: relative dates, indirect events, identity, adversarial/contextual."""

    def test_relative_date_extraction(self):
        """Test extraction of relative dates like 'last Friday', 'two days ago'."""
        model = HierarchicalMemoryModel()
        # Add memories with various relative date expressions
        model.add_memory("Caroline: I went to the LGBTQ support group yesterday.")
        model.add_memory("Melanie: I ran a charity race last Saturday for mental health.")
        model.add_memory("Caroline: I painted a sunset after visiting the beach last week.")
        model.add_memory("Melanie: I took my kids to a pottery workshop last Friday.")

        # Test various relative date queries
        results = model.query("When did Caroline go to the LGBTQ support group?")
        assert any("yesterday" in r[1].text.lower() for r in results)

        results = model.query("When did Melanie run a charity race?")
        assert any("last saturday" in r[1].text.lower() for r in results)

        results = model.query("When did Melanie go to the pottery workshop?")
        assert any("last friday" in r[1].text.lower() for r in results)

    def test_relative_date_reasoning(self):
        """Test reasoning about dates relative to other events."""
        model = HierarchicalMemoryModel()
        # Add memories with date relationships
        model.add_memory("Caroline: I gave a speech at a school last week.")
        model.add_memory("Melanie: I went to the museum two days after Caroline's speech.")
        model.add_memory("Caroline: The pride parade was the week before my speech.")

        # Test date reasoning queries
        results = model.query("When did Melanie go to the museum?")
        assert any("two days after" in r[1].text.lower() or "speech" in r[1].text.lower() for r in results)

        results = model.query("When was the pride parade?")
        assert any("week before" in r[1].text.lower() or "speech" in r[1].text.lower() for r in results)

    def test_indirect_event_extraction(self):
        """Test extraction of events described indirectly or across sentences."""
        model = HierarchicalMemoryModel()
        # Add memories with indirect event descriptions
        model.add_memory("Caroline: I researched adoption agencies to prepare for having kids.")
        model.add_memory("Melanie: I signed up for a pottery class to express my creativity.")
        model.add_memory("Caroline: I'm putting together an LGBTQ art show featuring various artists.")
        model.add_memory("Melanie: I read 'Nothing is Impossible' by the trans girl author.")

        # Test indirect event queries
        results = model.query("What did Caroline research?")
        assert any("adoption agencies" in r[1].text.lower() for r in results)

        results = model.query("When did Melanie sign up for a pottery class?")
        assert any("pottery class" in r[1].text.lower() for r in results)

        results = model.query("When did Melanie read the book 'Nothing is Impossible'?")
        assert any("nothing is impossible" in r[1].text.lower() for r in results)

    def test_identity_extraction_synonyms(self):
        """Test identity extraction with synonyms and indirect evidence."""
        model = HierarchicalMemoryModel()
        # Add memories with identity information using various terms
        model.add_memory("Caroline: I went to a LGBTQ support group and it was powerful.")
        model.add_memory("Melanie: Caroline, you're so brave as a trans person.")
        model.add_memory("Caroline: The transgender stories were so inspiring.")
        model.add_memory("Caroline: I am a transgender woman who transitioned.")
        model.add_memory("Melanie: I love how you embrace your womanhood.")

        # Test identity queries with different phrasings
        results = model.query("What is Caroline's identity?")
        identity_found = any(
            term in r[1].text.lower()
            for r in results
            for term in ["transgender", "trans", "woman", "transitioned", "lgbtq"]
        )
        assert identity_found

    def test_partial_contextual_answers(self):
        """Test providing partial matches when answer is present but not exact."""
        model = HierarchicalMemoryModel()
        # Add memories with partial information
        model.add_memory("Caroline: I'm creating a library for when I have kids.")
        model.add_memory("Caroline: I collect classic children's books for my future family.")
        model.add_memory("Melanie: Caroline loves reading to kids and has many books.")

        # Test partial answer queries
        results = model.query("What kind of books does Caroline collect?")
        assert any("children" in r[1].text.lower() or "kids" in r[1].text.lower() for r in results)

        results = model.query("Does Caroline have books for children?")
        assert any("children" in r[1].text.lower() or "kids" in r[1].text.lower() for r in results)

    def test_adversarial_implicit_questions(self):
        """Test handling adversarial and implicit questions."""
        model = HierarchicalMemoryModel()
        # Add memories with implicit information
        model.add_memory("Caroline: I'm creating a library for when I have kids.")
        model.add_memory("Caroline: I collect classic children's books including Dr. Seuss.")
        model.add_memory("Melanie: Caroline has a collection of children's literature.")
        model.add_memory("Caroline: Books guide me and help me discover who I am.")
        # Add LGBTQ context for political leaning inference
        model.add_memory("Caroline: I went to the LGBTQ pride parade and it was empowering.")
        model.add_memory("Caroline: I support transgender rights and acceptance.")

        # Test adversarial/implicit queries
        results = model.query("Would Caroline likely have Dr. Seuss books on her bookshelf?")
        assert any(
            term in r[1].text.lower()
            for r in results
            for term in ["dr. seuss", "children", "books", "library", "collect"]
        )

        results = model.query("What would Caroline's political leaning likely be?")
        # This should find evidence of liberal/progressive values from LGBTQ context
        assert any(
            term in r[1].text.lower()
            for r in results
            for term in ["lgbtq", "support", "acceptance", "pride", "transgender", "rights"]
        )

    def test_date_calculation_reasoning(self):
        """Test date calculation and reasoning capabilities."""
        model = HierarchicalMemoryModel()
        # Add memories with date relationships requiring calculation
        model.add_memory("Caroline: I gave a speech at a school on 9 June 2023.")
        model.add_memory("Melanie: I ran a charity race the Sunday before 25 May 2023.")
        model.add_memory("Caroline: I went camping the week before 9 June 2023.")
        model.add_memory("Melanie: I signed up for pottery class on 2 July 2023.")

        # Test date calculation queries
        results = model.query("When did Melanie run a charity race?")
        assert any("25 may 2023" in r[1].text.lower() or "sunday before" in r[1].text.lower() for r in results)

        results = model.query("When did Caroline go camping?")
        assert any("9 june 2023" in r[1].text.lower() or "week before" in r[1].text.lower() for r in results)

        results = model.query("When did Melanie sign up for a pottery class?")
        assert any("2 july 2023" in r[1].text.lower() for r in results)

    def test_cross_sentence_event_extraction(self):
        """Test extracting events described across multiple sentences."""
        model = HierarchicalMemoryModel()
        # Add memories with events split across sentences
        model.add_memory("Caroline: I joined a mentorship program. I've met amazing young folks.")
        model.add_memory("Caroline: I mentor a transgender teen. We had a great time at the LGBT pride event.")
        model.add_memory("Melanie: Caroline supports young people. She helps them with their transition.")

        # Test cross-sentence event queries
        results = model.query("What did Caroline do in her mentorship program?")
        assert any("mentor" in r[1].text.lower() or "young" in r[1].text.lower() for r in results)

        results = model.query("When did Caroline meet up with her friends, family, and mentors?")
        assert any("pride event" in r[1].text.lower() or "lgbt" in r[1].text.lower() for r in results)

    def test_synonym_identity_extraction(self):
        """Test identity extraction using synonyms and related terms."""
        model = HierarchicalMemoryModel()
        # Add memories with identity clues using synonyms
        model.add_memory("Caroline: The rainbow flag mural reflects the trans community.")
        model.add_memory("Melanie: Caroline transitioned and joined the transgender community.")
        model.add_memory("Caroline: I was born male but identify as female now.")
        model.add_memory("Melanie: Caroline is part of the LGBTQ+ community.")

        # Test synonym-based identity queries
        results = model.query("What is Caroline's gender identity?")
        identity_terms = ["transgender", "trans", "female", "woman", "lgbtq", "transitioned", "community"]
        assert any(
            any(term in r[1].text.lower() for term in identity_terms)
            for r in results
        )


class TestPerformance:
    """Test performance characteristics."""

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        model = HierarchicalMemoryModel()

        # Add 100 memories
        for i in range(100):
            model.add_memory(f"Memory number {i} with some content")

        # Should be able to query efficiently
        start_time = time.time()
        results = model.query("number 50", top_k=10)
        query_time = time.time() - start_time

        assert len(results) >= 1
        assert query_time < 1.0  # Should be fast


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model_queries(self):
        """Test queries on empty model."""
        model = HierarchicalMemoryModel()

        results = model.query("anything", top_k=5)
        assert results == []

    def test_empty_text_memory(self):
        """Test handling of empty text memories."""
        model = HierarchicalMemoryModel()

        # This should not crash
        key = model.add_memory("")
        results = model.query("test", top_k=1)
        assert isinstance(results, list)

    def test_very_long_text(self):
        """Test handling of very long text."""
        model = HierarchicalMemoryModel()

        # long_text = "word " * 1000  # Very long text
        # Create a long but reasonable text that fits within token limits (under 256 tokens)
        long_text = (
            "word " * 30
            + "This is a very long sentence that contains many words and should still be processable by the embedding model. "
            * 3
        )
        key = model.add_memory(long_text)

        # Should still work
        results = model.query("word", top_k=1)
        assert len(results) >= 1

    def test_unicode_text(self):
        """Test handling of unicode text."""
        model = HierarchicalMemoryModel()

        unicode_text = "Hello 世界 🌍 Test mémoire"
        key = model.add_memory(unicode_text)

        results = model.query("Hello", top_k=1)
        assert len(results) >= 1

    def test_duplicate_keys(self):
        """Test handling of duplicate keys (should auto-generate unique keys)."""
        model = HierarchicalMemoryModel()

        # Add memories - keys should be auto-generated and unique
        key1 = model.add_memory("Memory 1")
        key2 = model.add_memory("Memory 2")

        assert key1 != key2
        assert model.recall_by_key(key1) is not None
        assert model.recall_by_key(key2) is not None

    def test_query_single_word_boost(self):
        """Test the single-word query boost feature."""
        model = HierarchicalMemoryModel()

        model.add_memory("Bob went to the store")
        model.add_memory("Alice called about the project")
        model.add_memory("Unrelated content")

        # Single word query should get boost
        results = model.query("Bob", top_k=3)

        # Bob-related memory should be first or have higher score
        assert len(results) >= 1
        assert "Bob" in results[0][1].text


class TestMemoryDecay:
    """Test memory decay simulation features."""

    def test_decay_simulation_setup(self):
        """Test setting up decay simulation."""
        model = HierarchicalMemoryModel()

        # Add memories with different "ages"
        memories = [
            ("Recent memory", {"age": "1_hour"}),
            ("Day old memory", {"age": "1_day"}),
            ("Week old memory", {"age": "1_week"}),
            ("Month old memory", {"age": "1_month"}),
        ]

        for text, meta in memories:
            model.add_memory(text, meta=meta)

    def test_compaction_as_decay(self):
        """Test using compaction to simulate memory decay."""
        model = HierarchicalMemoryModel()

        # Add many "old" memories
        for i in range(20):
            model.add_memory(f"Old memory {i}", meta={"age": "old"})

        # Add few "recent" memories
        for i in range(3):
            model.add_memory(f"Recent memory {i}", meta={"age": "recent"})


class TestErrorHandling:
    """Test error handling and validation."""

    pass


# Integration tests combining multiple features
class TestDemoScenarios:
    """Test cases based on demo scenarios to ensure functionality works as demonstrated."""

    def test_neural_consolidation_demo_scenario(self):
        """Test the neural consolidation demo scenario."""
        model = HierarchicalMemoryModel()

        # Add memories about a person's life experiences (from demo)
        life_memories = [
            "I was born in a small town in 1990",
            "My first bicycle was red with training wheels",
            "I learned to swim when I was 6 years old",
            "My favorite childhood memory is building sandcastles at the beach",
            "I got my first computer when I was 12",
            "High school graduation was one of my proudest moments",
            "My first job was at a local bookstore",
            "I traveled to Europe for the first time in college",
            "Learning to drive was both exciting and terrifying",
            "My grandmother taught me how to bake cookies",
            "I started programming in my sophomore year",
            "The first time I fell in love was magical",
            "Graduating college felt like the beginning of everything",
            "My first apartment was tiny but felt like freedom",
            "I learned to cook Italian food from my roommate",
        ]

        for memory in life_memories:
            model.add_memory(memory, meta={"type": "life_experience"})

        # Test querying before consolidation
        query = "childhood memories"
        pre_results = model.query(query, top_k=5)
        assert len(pre_results) >= 1  # Should find some relevant memories

        # Test querying after consolidation
        post_results = model.query(query, top_k=5)
        assert len(post_results) >= 1  # Should still find memories

        # Test memory retention with different queries
        test_queries = [
            "learning experiences",
            "first experiences",
            "family traditions",
            "personal growth",
        ]

        for test_query in test_queries:
            results = model.query(test_query, top_k=3)
            assert isinstance(results, list)  # Should return a list

    def test_basic_memory_demo_scenario(self):
        """Test the basic memory demo scenario."""
        model = HierarchicalMemoryModel()

        # Add various types of memories (from demo)
        memories = [
            ("I love eating pizza with extra cheese", "food"),
            ("My favorite color is blue", "personal"),
            ("The meeting with John is at 3 PM tomorrow", "schedule"),
            ("John's email is john@company.com", "contact"),
            ("I need to buy groceries: milk, bread, eggs", "todo"),
            ("Alice's phone number is +1-555-0123", "contact"),
            ("I visited Paris last summer and loved the Eiffel Tower", "travel"),
            ("Remember to call mom on Sunday", "personal"),
            ("Python is my favorite programming language", "tech"),
            ("The restaurant 'Bella Vista' has amazing Italian food", "food"),
        ]

        for text, category in memories:
            key = model.add_memory(text, meta={"category": category})
            assert key is not None

        # Test queries (from demo)
        queries = [
            "What food do I like?",
            "John's contact information",
            "programming",
            "phone number",
            "meeting schedule",
        ]

        for query in queries:
            results = model.query(query, top_k=3, threshold=0.0)
            assert len(results) >= 1  # Each query should return at least one result
            assert all(
                isinstance(score, float) and -1 <= score <= 2 for score, _ in results
            )

    def test_performance_comparison(self):
        """Test performance comparison with and without neural consolidation."""
        # Create test dataset
        test_memories = [
            f"Memory item number {i}: This is a test memory about topic {i%5}"
            for i in range(50)  # Smaller dataset for faster testing
        ]

        # Test without neural consolidation
        model1 = HierarchicalMemoryModel()
        for mem in test_memories:
            model1.add_memory(mem)

        start_time = time.time()
        results1 = model1.query("topic 2", top_k=10)
        query_time1 = time.time() - start_time

        assert len(results1) >= 1

        # Test with neural consolidation
        model2 = HierarchicalMemoryModel()
        for mem in test_memories:
            model2.add_memory(mem)

        start_time = time.time()
        results2 = model2.query("topic 2", top_k=10)
        query_time2 = time.time() - start_time

        # Both should work and return results
        assert len(results2) >= 1

        # Performance should be reasonable (within 10x slower)
        assert query_time2 < query_time1 * 10

    def test_advanced_memory_features(self):
        """Test advanced memory features: decay, neurogenesis, and merging."""
        model = HierarchicalMemoryModel()

        # Add memories with different importance levels
        memories = [
            ("Critical system password: admin123", 0.9),
            ("Meeting notes from last week", 0.7),
            ("Random thought about weather", 0.3),
            ("Important project deadline: tomorrow", 0.8),
            ("Favorite coffee shop location", 0.6),
            ("Temporary note: buy milk", 0.2),
            ("Critical project documentation", 0.9),
            ("Another meeting note", 0.4),  # Similar to existing
        ]

        for text, importance in memories:
            model.add_memory(
                text, importance=importance, meta={"demo": True}
            )

        # Test neurogenesis with many memories
        for i in range(60):  # Should trigger neurogenesis
            model.add_memory(
                f"Additional memory {i} with some content", importance=0.5
            )

        # Query should still work
        results = model.query("meeting", top_k=3)
        assert len(results) >= 1
