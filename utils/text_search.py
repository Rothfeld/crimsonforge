"""Token-based search matching for the Explorer search bar.

Why this module exists
----------------------
The Explorer search bar previously used a naive ``query in haystack``
substring check. That works for short stems but produces silent false
positives once item names enter the picture: a user typing ``canta``
matched both ``Canta_PlateArmor_Armor`` (the right one) AND
``Eccanta_PlateArmor_Armor`` / ``Cantarts_Leather_Armor`` (unrelated
items whose internal names happen to contain the 5-letter sequence
``canta``). The user's natural-language phrase ``canta plate armor``
also matched ``eccanta plate armor`` for the same reason — substring
matching has no concept of word boundaries.

What it does
------------
Both the query and the per-row search corpus are split into
LOWERCASE TOKENS using one uniform tokenizer that handles whitespace,
underscores, hyphens, slashes, periods, and CamelCase boundaries. A
row is considered a hit when every query token is a prefix of at
least one corpus token (AND semantics).

The CamelCase split is essential — game-data internal names look
like ``Canta_PlateArmor_Armor``. Without splitting between ``Plate``
and ``Armor``, the token would be ``platearmor`` and the user's
``plate armor`` (with a space) wouldn't match.

The prefix rule preserves the natural feel of starts-with search
without leaking into the middle of unrelated tokens:

  * ``canta`` matches the token ``canta`` (exact prefix).
  * ``canta`` matches ``cantarts`` — ``canta`` is a prefix of it.
  * ``canta`` does NOT match ``eccanta`` — the prefix isn't at the
    start of the token.

For most modders the surviving false positive (``cantarts`` matches
``canta``) is acceptable: typing more of the name (``canta_p`` or
``canta plate``) immediately resolves it. The original ``canta``
matched ``Eccanta`` AND ``Cantarts``; this version drops the
``Eccanta`` hit, which is the one users actually complained about.
"""
from __future__ import annotations

import re

# Whitespace and the common name / path separators we always want to
# split on. Backslash is intentionally included for Windows paths.
_SEP_RE = re.compile(r"[\s_./\\\-:|]+")

# CamelCase boundary detector: split between a lowercase / digit and
# the uppercase that follows it (``PlateArmor`` -> ``Plate|Armor``)
# AND between two uppercase letters when the second is followed by a
# lowercase (``URLPath`` -> ``URL|Path``). Standard pattern used by
# Google's open-source style guides and by ``inflection.underscore``.
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into a list of lowercase tokens.

    Tokenization steps:

    1. Split on whitespace + ``_`` + ``.`` + ``/`` + ``\\`` + ``-`` + ``:`` + ``|``.
    2. Split each chunk on CamelCase boundaries.
    3. Lowercase every result; drop empties.

    Examples::

        tokenize("Canta_PlateArmor_Armor") -> ["canta", "plate", "armor", "armor"]
        tokenize("cd_phw_canta_plate_armor_d.dds") ->
            ["cd", "phw", "canta", "plate", "armor", "d", "dds"]
        tokenize("Mace of Ambition") -> ["mace", "of", "ambition"]
    """
    if not text:
        return []
    out: list[str] = []
    for chunk in _SEP_RE.split(text):
        if not chunk:
            continue
        for sub in _CAMEL_RE.split(chunk):
            if sub:
                out.append(sub.lower())
    return out


def tokens_for(*corpora: str) -> set[str]:
    """Tokenize one or more strings into a single deduplicated set.

    Hot-loop callers (the Explorer's 1.4 M-row filter, the Catalog
    Browser's 19 k-row filter) cache this result per row so each
    keystroke only re-tokenizes the query, never the corpus.
    """
    out: set[str] = set()
    for c in corpora:
        if c:
            out.update(tokenize(c))
    return out


def match_prefilter(query_tokens: list[str],
                    corpus_tokens: set[str]) -> bool:
    """Fast inner predicate for already-tokenized inputs.

    Returns True when every token in ``query_tokens`` is a prefix of
    at least one token in ``corpus_tokens``. The caller is responsible
    for deciding what an empty query / empty corpus should mean —
    this function says ``True`` for an empty query (no constraints)
    and ``False`` for a non-empty query against an empty corpus.
    """
    if not query_tokens:
        return True
    if not corpus_tokens:
        return False
    for qt in query_tokens:
        for ct in corpus_tokens:
            if ct.startswith(qt):
                break
        else:
            return False
    return True


def match(query: str, *corpora: str) -> bool:
    """Return True when every token in ``query`` is a prefix of some
    token extracted from one of ``corpora``.

    ``corpora`` accepts multiple strings so the caller doesn't have
    to pre-join the file path with the alias terms. Empty / missing
    strings are simply skipped.

    An empty query matches everything (so an empty filter doesn't
    hide rows). An empty corpus rejects every non-empty query.

    Note: this convenience wrapper re-tokenizes the corpus on every
    call. Hot-loop callers should use :func:`tokens_for` to cache the
    corpus token set on their row objects and call
    :func:`match_prefilter` instead.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return True
    corpus: set[str] = set()
    for c in corpora:
        if c:
            corpus.update(tokenize(c))
    if not corpus:
        return False
    for qt in q_tokens:
        if not any(ct.startswith(qt) for ct in corpus):
            return False
    return True
