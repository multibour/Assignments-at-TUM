import pytest
import aa_props
"""
Test amino-acid properties functions. We only provide one example test.
All other tests will look similar.
"""

def test_charge():
    """
    Test the isCharged function. The server test might look similar
    """
    charged = ["R", "D", "K"]
    not_charged = ["A", "Q", "G"]
    results = [aa_props.isCharged(aa) for aa in charged ]
    results += [not aa_props.isCharged(aa) for aa in not_charged]
    assert all(results)