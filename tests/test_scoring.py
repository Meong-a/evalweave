import unittest

from group_llm_benchmark.scorers.text import bleu4, normalized_exact_match, rouge_l_f1, token_f1


class ScoringTests(unittest.TestCase):
    def test_normalized_exact_match(self):
        self.assertEqual(normalized_exact_match("Hello, World!", "hello world"), 1.0)

    def test_token_f1_partial(self):
        value = token_f1("alpha beta", "alpha gamma")
        self.assertAlmostEqual(value, 0.5)

    def test_rouge_l(self):
        self.assertGreater(rouge_l_f1("a b c", "a x c"), 0.6)

    def test_bleu4_is_bounded(self):
        value = bleu4("the cat sat", "the cat sat")
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
