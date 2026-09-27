from scripts.psc_rollout import find_matches


def test_psc_keyword_matching_is_case_insensitive():
    pathways = [
        "Interferon signaling",
        "Extracellular matrix organization",
        "DNA replication",
    ]

    hits = find_matches(
        pathways,
        ["IFN", "extracellular matrix"],
    )

    assert [index for index, _, _ in hits] == [1]
