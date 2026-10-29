def cache_hit_rate(keys: list[int]) -> tuple[int, int, float]:
    """
    keys: a sequence of request keys in arrival order.

    A request is a hit iff its key has appeared earlier in the sequence
    (the first occurrence of any key is always a miss).

    Return (hits, misses, rate) where rate = hits / len(keys).
    """
    # TODO: track which keys have been seen so far, in order.
    if not keys:
        return 0, 0, 0.0
    cache = set()
    hits, n = 0, 0
    for k in keys:
        if k in cache:
            hits += 1
        else:
            cache.add(k)
        n += 1
    return hits, n - hits, hits / n
