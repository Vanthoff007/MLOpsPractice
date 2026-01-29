from inference_onnx import OnnxInference


def get_test_examples():
    """
    Returns a dictionary of test cases for paraphrase detection
    """

    test_cases = {
        # CLEAR PARAPHRASES (should score high for "paraphrase")
        "paraphrases": [
            {
                "sentence1": "The firm announced increased earnings.",
                "sentence2": "The company reported higher profits.",
                "expected": "paraphrase",
            },
            {
                "sentence1": "The cat sat on the mat.",
                "sentence2": "A feline rested on the rug.",
                "expected": "paraphrase",
            },
            {
                "sentence1": "He is a very intelligent person.",
                "sentence2": "He's an extremely smart individual.",
                "expected": "paraphrase",
            },
            {
                "sentence1": "The weather is quite cold today.",
                "sentence2": "It's pretty chilly outside today.",
                "expected": "paraphrase",
            },
            {
                "sentence1": "She went to the store to buy groceries.",
                "sentence2": "She visited the market to purchase food items.",
                "expected": "paraphrase",
            },
        ],
        # CLEAR NON-PARAPHRASES (should score high for "not_paraphrase")
        "non_paraphrases": [
            {
                "sentence1": "The dog is sleeping.",
                "sentence2": "The cat is playing.",
                "expected": "not_paraphrase",
            },
            {
                "sentence1": "I love summer vacations.",
                "sentence2": "Winter sports are exciting.",
                "expected": "not_paraphrase",
            },
            {
                "sentence1": "The movie starts at 7 PM.",
                "sentence2": "My birthday is in December.",
                "expected": "not_paraphrase",
            },
            {
                "sentence1": "She enjoys reading books.",
                "sentence2": "He dislikes watching television.",
                "expected": "not_paraphrase",
            },
            {
                "sentence1": "The car is red.",
                "sentence2": "The sky is blue.",
                "expected": "not_paraphrase",
            },
        ],
        # EDGE CASES
        "edge_cases": [
            {
                "sentence1": "Hello world!",
                "sentence2": "Hello world!",
                "expected": "paraphrase",
                "note": "Identical sentences",
            },
            {
                "sentence1": "Yes.",
                "sentence2": "No.",
                "expected": "not_paraphrase",
                "note": "Very short sentences - opposite meaning",
            },
            {
                "sentence1": "The meeting is scheduled for tomorrow.",
                "sentence2": "The meeting was held yesterday.",
                "expected": "not_paraphrase",
                "note": "Temporal difference (future vs past)",
            },
            {
                "sentence1": "John gave the book to Mary.",
                "sentence2": "Mary gave the book to John.",
                "expected": "not_paraphrase",
                "note": "Same words, different meaning (reversed roles)",
            },
            {
                "sentence1": "The glass is half full.",
                "sentence2": "The glass is half empty.",
                "expected": "paraphrase",
                "note": "Same meaning, different perspective",
            },
        ],
        # NEGATION CASES
        "negation": [
            {
                "sentence1": "I like ice cream.",
                "sentence2": "I don't like ice cream.",
                "expected": "not_paraphrase",
                "note": "Negation changes meaning",
            },
            {
                "sentence1": "The project was successful.",
                "sentence2": "The project was not successful.",
                "expected": "not_paraphrase",
                "note": "Negation in formal context",
            },
        ],
        # NUMERICAL DIFFERENCES
        "numerical": [
            {
                "sentence1": "The company has 100 employees.",
                "sentence2": "The firm employs 100 people.",
                "expected": "paraphrase",
                "note": "Same number, paraphrased",
            },
            {
                "sentence1": "The price increased by 10%.",
                "sentence2": "The cost went up by 20%.",
                "expected": "not_paraphrase",
                "note": "Different numbers",
            },
        ],
        # CHALLENGING CASES (semantic similarity)
        "challenging": [
            {
                "sentence1": "The economy is growing rapidly.",
                "sentence2": "Economic expansion is accelerating.",
                "expected": "paraphrase",
                "note": "Technical vocabulary",
            },
            {
                "sentence1": "She bought a new house.",
                "sentence2": "She purchased a residence.",
                "expected": "paraphrase",
                "note": "Formal vs casual language",
            },
            {
                "sentence1": "The stock market crashed.",
                "sentence2": "Equity prices plummeted.",
                "expected": "paraphrase",
                "note": "Domain-specific terminology",
            },
            {
                "sentence1": "He started the car.",
                "sentence2": "He stopped the car.",
                "expected": "not_paraphrase",
                "note": "Antonyms - opposite actions",
            },
        ],
    }

    return test_cases


def run_tests(predictor):
    """
    Run all test cases and display results
    """
    test_cases = get_test_examples()

    print("=" * 80)
    print("PARAPHRASE DETECTION TEST SUITE")
    print("=" * 80)

    total_tests = 0
    category_results = {}

    for category, cases in test_cases.items():
        print(f"\n{'=' * 80}")
        print(f"Category: {category.upper()}")
        print(f"{'=' * 80}\n")

        correct = 0
        total = len(cases)

        for i, case in enumerate(cases, 1):
            sentence1 = case["sentence1"]
            sentence2 = case["sentence2"]
            expected = case.get("expected", "unknown")
            note = case.get("note", "")

            print(f"Test {i}/{total}:")
            print(f"  S1: {sentence1}")
            print(f"  S2: {sentence2}")

            try:
                result = predictor.predict(sentence1, sentence2)
                predicted_label = max(result, key=lambda x: x["score"])

                print(f"  Results: {result}")
                print(
                    f"  Predicted: {predicted_label['label']} (confidence: {predicted_label['score']:.4f})"
                )
                print(f"  Expected: {expected}")

                if predicted_label["label"] == expected:
                    print("PASS")
                    correct += 1
                else:
                    print("FAIL")

                if note:
                    print(f"  Note: {note}")

            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")

            print()

        accuracy = (correct / total) * 100 if total > 0 else 0
        category_results[category] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
        total_tests += total

        print(f"Category Results: {correct}/{total} correct ({accuracy:.1f}%)")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_correct = sum(r["correct"] for r in category_results.values())
    overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0

    for category, results in category_results.items():
        print(
            f"{category:20s}: {results['correct']:2d}/{results['total']:2d} ({results['accuracy']:5.1f}%)"
        )

    print(
        f"\n{'TOTAL':20s}: {total_correct:2d}/{total_tests:2d} ({overall_accuracy:5.1f}%)"
    )
    print("=" * 80)


if __name__ == "__main__":
    # Initialize predictor
    predictor = OnnxInference("./models/mrpc_model.onnx")

    # Run all tests
    run_tests(predictor)

    # Or test individual examples
    print("\n" + "=" * 80)
    print("INDIVIDUAL TEST EXAMPLES")
    print("=" * 80 + "\n")

    # Quick test
    s1 = "The firm announced increased earnings."
    s2 = "The company reported higher profits."
    result = predictor.predict(s1, s2)
    print(f"Sentence 1: {s1}")
    print(f"Sentence 2: {s2}")
    print(f"Result: {result}\n")
