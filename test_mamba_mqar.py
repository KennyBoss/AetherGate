import unittest

import numpy as np

from compare_long_context_recall import make_mqar_recall as reference_mqar
from run_mamba_mqar import (
    choose_d_model,
    estimated_parameter_count,
    make_mqar_recall,
)


class MambaMQARTest(unittest.TestCase):
    def test_generator_is_byte_identical_to_registered_mqar(self) -> None:
        arguments = dict(records=7, delay=48, key_count=16, binding_count=4, query_count=2, seed=11)
        expected = reference_mqar(**arguments)
        actual = make_mqar_recall(**arguments)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)

    def test_parameter_search_matches_registered_budget(self) -> None:
        d_model, count = choose_d_model(
            target_params=100_000,
            vocab_size=129,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.assertGreaterEqual(d_model, 8)
        self.assertGreaterEqual(count / 100_000, 0.8)
        self.assertLessEqual(count / 100_000, 1.25)
        self.assertEqual(
            count,
            estimated_parameter_count(
                d_model=d_model,
                vocab_size=129,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
