from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
CREATE_TABLE = re.compile(rf"\bCREATE\s+TABLE\s+({IDENTIFIER})\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
TOKEN = re.compile(IDENTIFIER)
SQL_WORDS = {
    "select",
    "distinct",
    "from",
    "inner",
    "join",
    "on",
    "where",
    "and",
    "or",
    "group",
    "by",
    "order",
    "asc",
    "desc",
    "limit",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "as",
    "null",
}


def _split_fields(body: str) -> list[str]:
    fields: list[str] = []
    start = 0
    depth = 0
    quoted = False
    for index, character in enumerate(body):
        if character == "'":
            quoted = not quoted
        elif not quoted and character == "(":
            depth += 1
        elif not quoted and character == ")":
            depth -= 1
        elif not quoted and character == "," and depth == 0:
            fields.append(body[start:index].strip())
            start = index + 1
    fields.append(body[start:].strip())
    return [field for field in fields if field]


def _replace_outside_literals(text: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return text
    lookup = {name.casefold(): replacement for name, replacement in replacements.items()}
    alternatives = "|".join(re.escape(name) for name in sorted(replacements, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9_])({alternatives})(?![A-Za-z0-9_])", re.IGNORECASE)
    parts = re.split(r"('(?:''|[^'])*')", text)
    for index in range(0, len(parts), 2):
        parts[index] = pattern.sub(lambda match: lookup[match.group(0).casefold()], parts[index])
    return "".join(parts)


def _singularize(word: str) -> str:
    lowered = word.casefold()
    if lowered.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if lowered.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if lowered.endswith("s") and not lowered.endswith(("ss", "us")):
        return word[:-1]
    return word


def _pluralize(word: str) -> str:
    lowered = word.casefold()
    if lowered.endswith("s"):
        return word
    if lowered.endswith("y") and len(word) > 1 and lowered[-2] not in "aeiou":
        return word[:-1] + "ies"
    if lowered.endswith(("sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"


def _identifier_aliases(identifier: str) -> set[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier).casefold()
    words = [word for word in snake_case.split("_") if word]
    if words and words[0] in {"tbl", "col"}:
        words = words[1:]
    if len(words) > 1 and words[-1] in {"data", "value"}:
        words = words[:-1]
    if not words:
        return set()
    aliases = {" ".join(words)}
    aliases.add(" ".join((*words[:-1], _singularize(words[-1]))))
    aliases.add(" ".join((*words[:-1], _pluralize(words[-1]))))
    if len(words) > 1 and words[-1] in {"id", "key", "identifier"}:
        # People commonly pluralize the entity noun instead of (or as well as)
        # the identifier noun: customer_id -> customers_id / customers_ids.
        # Retain underscore forms because they are often pasted verbatim into a
        # natural-language question.
        for entity in {_singularize(words[-2]), _pluralize(words[-2])}:
            for identifier_word in {_singularize(words[-1]), _pluralize(words[-1])}:
                variant = (*words[:-2], entity, identifier_word)
                aliases.add(" ".join(variant))
                aliases.add("_".join(variant))
    return {alias for alias in aliases if alias}


@dataclass(frozen=True)
class IdentifierMapping:
    table_to_slot: dict[str, str]
    column_to_slot: dict[str, str]
    column_to_tables: dict[str, tuple[str, ...]]
    primary_column_table: dict[str, str]

    @property
    def raw_to_slot(self) -> dict[str, str]:
        overlap = set(self.table_to_slot) & set(self.column_to_slot)
        if overlap:
            raise ValueError(f"Table and column identifiers overlap and cannot be canonicalized safely: {sorted(overlap)}")
        return {**self.table_to_slot, **self.column_to_slot}

    @property
    def slot_to_raw(self) -> dict[str, str]:
        return {slot: raw for raw, slot in self.raw_to_slot.items()}

    @property
    def allowed_slots(self) -> set[str]:
        return set(self.slot_to_raw)

    def canonicalize(self, text: str) -> str:
        return _replace_outside_literals(text, self.raw_to_slot)

    def canonicalize_question(self, question: str) -> str:
        """Resolve exact identifiers plus simple human singular/plural aliases."""
        replacements = dict(self.table_to_slot)
        replacements.update(
            {
                raw: self._question_column_slot(raw, raw)
                for raw in self.column_to_slot
            }
        )
        canonical = _replace_outside_literals(question, replacements)
        candidates: dict[str, set[str]] = {}
        for raw, slot in self.table_to_slot.items():
            for alias in _identifier_aliases(raw):
                if alias.casefold() == raw.casefold():
                    continue
                candidates.setdefault(alias, set()).add(slot)
        for raw, slot in self.column_to_slot.items():
            for alias in _identifier_aliases(raw):
                if alias.casefold() == raw.casefold():
                    continue
                candidates.setdefault(alias, set()).add(self._question_column_slot(raw, alias))
        unambiguous = {alias: next(iter(slots)) for alias, slots in candidates.items() if len(slots) == 1}
        return _replace_outside_literals(canonical, unambiguous)

    def _question_column_slot(self, column: str, phrase: str) -> str:
        """Keep a table hint when it is embedded inside a column mention.

        Without this, phrases such as ``order ids`` collapse to only ``column4``
        and discard the fact that the user named the orders table.  Shared keys
        such as ``customer_id`` are resolved to the owner whose table noun is
        present in the phrase, preferring its primary-key owner as a tie-breaker.
        """
        column_words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column).casefold().split("_")
        if not column_words or column_words[-1] not in {"id", "key", "identifier"}:
            return self.column_to_slot[column]
        normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", phrase).replace("_", " ").casefold()
        owners = self.column_to_tables.get(column, ())
        mentioned = [
            table
            for table in owners
            if any(
                re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", normalized)
                for alias in _identifier_aliases(table)
            )
        ]
        if not mentioned:
            return self.column_to_slot[column]
        primary = self.primary_column_table.get(column)
        table = primary if primary in mentioned else mentioned[0]
        return f"{self.table_to_slot[table]} {self.column_to_slot[column]}"

    def restore(self, text: str) -> str:
        return _replace_outside_literals(text, self.slot_to_raw)

    def accepts_sql(self, sql: str) -> bool:
        without_literals = re.sub(r"'(?:''|[^'])*'", "''", sql)
        return all(token.casefold() in SQL_WORDS or token in self.allowed_slots for token in TOKEN.findall(without_literals))


def identifier_mapping(schema_sql: str, slot_strategy: str = "ordered") -> IdentifierMapping:
    if slot_strategy not in {"ordered", "permuted"}:
        raise ValueError("slot_strategy must be 'ordered' or 'permuted'")
    tables: list[str] = []
    columns: list[str] = []
    column_to_tables: dict[str, list[str]] = {}
    primary_column_table: dict[str, str] = {}
    for match in CREATE_TABLE.finditer(schema_sql):
        table_name, body = match.groups()
        if table_name not in tables:
            tables.append(table_name)
        for field in _split_fields(body):
            column_match = re.match(rf"({IDENTIFIER})\b", field)
            if not column_match:
                continue
            column_name = column_match.group(1)
            if column_name.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                continue
            if column_name not in columns:
                columns.append(column_name)
            column_to_tables.setdefault(column_name, []).append(table_name)
            if "PRIMARY KEY" in field.upper():
                primary_column_table[column_name] = table_name
    if not tables or not columns:
        raise ValueError("Schema canonicalization requires CREATE TABLE statements with unquoted identifiers")
    if slot_strategy == "permuted":
        def permuted(names: list[str], namespace: str) -> list[str]:
            return sorted(
                names,
                key=lambda name: hashlib.sha256(
                    f"{schema_sql}\0{namespace}\0{name.casefold()}".encode("utf-8")
                ).digest(),
            )

        tables = permuted(tables, "table")
        columns = permuted(columns, "column")
    return IdentifierMapping(
        {name: f"table{index}" for index, name in enumerate(tables)},
        {name: f"column{index}" for index, name in enumerate(columns)},
        {name: tuple(owners) for name, owners in column_to_tables.items()},
        primary_column_table,
    )


def canonicalize_inputs(
    schema_sql: str,
    question: str,
    slot_strategy: str = "ordered",
) -> tuple[str, str, IdentifierMapping]:
    mapping = identifier_mapping(schema_sql, slot_strategy)
    return mapping.canonicalize(schema_sql), mapping.canonicalize_question(question), mapping


def canonicalize_record(record: dict, slot_strategy: str = "ordered") -> dict:
    schema, question, mapping = canonicalize_inputs(record["schema_sql"], record["question"], slot_strategy)
    return {**record, "schema_sql": schema, "question": question, "sql": mapping.canonicalize(record["sql"])}
