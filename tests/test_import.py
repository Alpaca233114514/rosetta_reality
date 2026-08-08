"""Package import tests."""


def test_package_import() -> None:
    """The package imports without loading an external model."""

    import rosetta_reality

    assert rosetta_reality.__version__ == "0.1.0"

