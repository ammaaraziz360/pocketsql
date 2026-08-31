"""Prefix-valid grammar for constrained PocketSQL semantic-plan decoding."""
from __future__ import annotations

from dataclasses import dataclass
import re

from pocketsql.model.schema_grounding import IdentifierMapping


CLAUSE_ORDER = {key: index for index, key in enumerate(("T", "S", "A", "D", "J", "F", "G", "O", "L"))}
AGGREGATES = ("COUNT", "SUM", "AVG", "MIN", "MAX")
OPERATORS = (" = ", " > ", " < ", " >= ", " <= ")
DIRECTIONS = ("ASC", "DESC")
MAX_LIST_ITEMS = 8
MAX_FILTERS = 4


@dataclass(frozen=True)
class Status:
    prefix: bool
    complete: bool = False


INVALID = Status(False, False)


def _finite_status(text: str, candidates: tuple[str, ...] | list[str]) -> Status:
    return Status(any(candidate.startswith(text) for candidate in candidates), text in candidates)


def _marker_body(text: str, marker: str) -> tuple[str | None, Status | None]:
    if marker.startswith(text):
        return None, Status(True, False)
    if not text.startswith(marker):
        return None, INVALID
    return text[len(marker) :], None


def _list_status(text: str, marker: str, candidates: tuple[str, ...], limit: int = MAX_LIST_ITEMS) -> Status:
    body, early = _marker_body(text, marker)
    if early is not None:
        return early
    parts = body.split(",")
    if len(parts) > limit or any(part not in candidates for part in parts[:-1]):
        return INVALID
    final = _finite_status(parts[-1], candidates)
    return Status(final.prefix, final.complete)


def _fields_status(text: str, marker: str, fields: tuple[tuple[str, ...], ...]) -> Status:
    body, early = _marker_body(text, marker)
    if early is not None:
        return early
    pieces = body.split(" ")
    if len(pieces) > len(fields) or any(not piece for piece in pieces[:-1]):
        return INVALID
    if any(piece not in candidates for piece, candidates in zip(pieces[:-1], fields)):
        return INVALID
    current = _finite_status(pieces[-1], fields[len(pieces) - 1])
    complete = len(pieces) == len(fields) and current.complete
    return Status(current.prefix, complete)


def _quoted_literal_status(text: str) -> Status:
    if not text.startswith("'") or len(text) > 66:
        return INVALID
    index = 1
    while index < len(text):
        character = text[index]
        if character in "\r\n;|":
            return INVALID
        if character == "'":
            if index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            return Status(index == len(text) - 1, index == len(text) - 1)
        index += 1
    return Status(True, False)


def _literal_status(text: str, value_slots: tuple[str, ...]) -> Status:
    finite = _finite_status(text, value_slots) if value_slots else INVALID
    quoted = _quoted_literal_status(text) if text.startswith("'") else INVALID
    numeric_prefix = bool(re.fullmatch(r"-?(?:\d+)?(?:\.\d*)?", text)) and text not in {"", "-", ".", "-."}
    numeric_complete = bool(re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", text))
    return Status(finite.prefix or quoted.prefix or numeric_prefix, finite.complete or quoted.complete or numeric_complete)


def _predicate_status(text: str, references: tuple[str, ...], value_slots: tuple[str, ...]) -> Status:
    prefix = False
    complete = False
    for reference in references:
        if reference.startswith(text):
            prefix = True
        if not text.startswith(reference):
            continue
        remainder = text[len(reference) :]
        for operator in OPERATORS:
            if operator.startswith(remainder):
                prefix = True
            if not remainder.startswith(operator):
                continue
            literal = _literal_status(remainder[len(operator) :], value_slots)
            prefix = prefix or literal.prefix
            complete = complete or literal.complete
    return Status(prefix, complete)


def _split_unquoted(text: str, delimiter: str) -> list[str] | None:
    parts = []
    start = 0
    quoted = False
    index = 0
    while index < len(text):
        if text[index] == "'":
            if quoted and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        if not quoted and text.startswith(delimiter, index):
            parts.append(text[start:index])
            start = index + len(delimiter)
            index = start
            continue
        index += 1
    parts.append(text[start:])
    return parts


def _separated_status(text: str, delimiter: str, item_status, limit: int) -> Status:
    parts = _split_unquoted(text, delimiter)
    if parts is None or len(parts) > limit:
        return INVALID
    if any(not item_status(part).complete for part in parts[:-1]):
        return INVALID
    current = item_status(parts[-1])
    prefix = current.prefix
    if not prefix:
        for suffix_length in range(1, len(delimiter)):
            suffix = delimiter[:suffix_length]
            if text.endswith(suffix) and item_status(text[:-suffix_length]).complete:
                prefix = True
                break
    return Status(prefix, current.complete)


@dataclass(frozen=True)
class SemanticPlanGrammar:
    tables: tuple[str, ...]
    references: tuple[str, ...]
    selection_references: tuple[str, ...]
    value_slots: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping: IdentifierMapping) -> "SemanticPlanGrammar":
        tables = tuple(sorted(set(mapping.table_to_slot.values())))
        references = set(mapping.column_to_slot.values())
        selection_references = {"*", *references}
        for raw_column, owners in mapping.column_to_tables.items():
            column = mapping.column_to_slot[raw_column]
            for owner in owners:
                table = mapping.table_to_slot[owner]
                references.add(f"{table}.{column}")
                selection_references.add(f"{table}.{column}")
        for table in tables:
            selection_references.add(f"{table}.*")
        return cls(
            tables,
            tuple(sorted(references)),
            tuple(sorted(selection_references)),
            tuple(binding.slot for binding in mapping.literals),
        )

    def _clause_status(self, text: str, key: str) -> Status:
        if key == "T":
            return _finite_status(text, tuple(f"T {table}" for table in self.tables))
        if key == "S":
            return _list_status(text, "S ", self.selection_references)
        if key == "A":
            targets = ("*", *self.references)
            status = _fields_status(
                text,
                "A ",
                (AGGREGATES, targets, tuple(str(position) for position in range(MAX_LIST_ITEMS + 1))),
            )
            if text.startswith("A "):
                pieces = text[2:].split(" ")
                if len(pieces) >= 2 and pieces[0] in AGGREGATES and pieces[0] != "COUNT" and pieces[1] == "*":
                    return INVALID
            return status
        if key == "D":
            return _finite_status(text, ("D",))
        if key == "J":
            return _fields_status(text, "J ", (self.tables, self.references, self.references))
        if key == "F":
            marker = next((candidate for candidate in ("F AND ", "F OR ") if text.startswith(candidate)), None)
            if marker is None:
                return Status(any(candidate.startswith(text) for candidate in ("F AND ", "F OR ")), False)
            body = text[len(marker) :]
            return _separated_status(
                body,
                " & ",
                lambda item: _predicate_status(item, self.references, self.value_slots),
                MAX_FILTERS,
            )
        if key == "G":
            return _list_status(text, "G ", self.references)
        if key == "O":
            return _fields_status(text, "O ", (self.references, DIRECTIONS))
        if key == "L":
            body, early = _marker_body(text, "L ")
            if early is not None:
                return early
            prefix = bool(re.fullmatch(r"\d*", body))
            complete = bool(body and body.isdigit() and int(body) > 0)
            return Status(prefix, complete)
        return INVALID

    @staticmethod
    def _allowed_keys(completed: tuple[str, ...]) -> tuple[str, ...]:
        if not completed:
            return ("T",)
        last_order = CLAUSE_ORDER[completed[-1]]
        has_target = "S" in completed or "A" in completed
        keys = tuple(key for key, order in CLAUSE_ORDER.items() if order > last_order)
        if not has_target:
            keys = tuple(key for key in keys if key in {"S", "A"})
        return keys

    def _exact_status(self, text: str) -> tuple[bool, bool, bool]:
        parts = _split_unquoted(text, " | ")
        if parts is None:
            return False, False, False
        completed: list[str] = []
        for clause in parts[:-1]:
            key = clause[:1]
            if key not in self._allowed_keys(tuple(completed)) or not self._clause_status(clause, key).complete:
                return False, False, False
            completed.append(key)
        current = parts[-1]
        viable = False
        complete = False
        clause_complete = False
        for key in self._allowed_keys(tuple(completed)):
            status = self._clause_status(current, key)
            viable = viable or status.prefix
            if status.complete:
                clause_complete = True
                candidate_keys = (*completed, key)
                complete = complete or ("S" in candidate_keys or "A" in candidate_keys)
        return viable or complete, complete, clause_complete

    def status(self, text: str) -> Status:
        prefix, complete, _ = self._exact_status(text)
        if prefix:
            return Status(True, complete)
        # Tokenizers emit the canonical clause separator one piece at a time.
        # A completed clause followed by " " or " |" is therefore a valid
        # prefix even though the next clause has not started yet.
        for suffix in (" |", " "):
            if not text.endswith(suffix):
                continue
            base = text[: -len(suffix)]
            base_prefix, _, clause_complete = self._exact_status(base)
            if base_prefix and clause_complete:
                return Status(True, False)
        return INVALID

    def is_prefix(self, text: str) -> bool:
        return self.status(text).prefix

    def is_complete(self, text: str) -> bool:
        return self.status(text).complete
