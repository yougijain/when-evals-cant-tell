"""What an LLM eval of a given size can and cannot tell you.

The modules here are deliberately small and separable:

    compare       two systems on the same items, done paired and unpaired
    power         how large an eval has to be to see a difference that size
    multiplicity  many variants at once, and the cost of reporting the best
    io            reading a tidy results file

Nothing here calls a model. It takes scored eval results as input, which keeps
every number in this repository reproducible from a clean checkout.
"""
__all__ = ["compare", "power", "multiplicity", "io"]
