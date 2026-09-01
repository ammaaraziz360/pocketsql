from __future__ import annotations

from dataclasses import dataclass, replace
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

LOCATION_IDENTIFIER_WORDS = {
    "area",
    "campus",
    "city",
    "district",
    "market",
    "neighborhood",
    "region",
    "sector",
    "territory",
    "zone",
}


@dataclass(frozen=True)
class LiteralBinding:
    value: str | int | float
    slot: str

    @property
    def question_text(self) -> str:
        return str(self.value)

    @property
    def sql_text(self) -> str:
        if isinstance(self.value, str):
            return "'" + self.value.replace("'", "''") + "'"
        return str(self.value)


@dataclass(frozen=True)
class ForeignKeyBinding:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


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
    if len(words) > 1:
        aliases.update({"-".join(words), "/".join(words)})
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


def _identifier_label(identifier: str) -> str:
    """Expose an identifier's readable words without making it the output key."""
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier).casefold()
    words = [word for word in snake_case.split("_") if word]
    return " ".join(words) or identifier.casefold()


def _extract_question_literals(question: str, excluded_words: set[str] | None = None) -> list[str | int | float]:
    """Extract explicit filter/limit values after identifiers have become slots."""
    found: list[tuple[int, str | int | float]] = []
    occupied: list[tuple[int, int]] = []
    excluded_words = excluded_words or set()

    def add(start: int, end: int, value: str | int | float) -> None:
        if any(left < end and start < right for left, right in occupied):
            return
        if isinstance(value, str):
            if value.casefold() in excluded_words:
                return
            if re.fullmatch(r"(?:table|column|value)\d+", value, re.IGNORECASE):
                return
            # Prepositions in ordinary requests are not always introducing a
            # literal: "in each status category" and "for every customer"
            # must not bind ``each`` or ``every`` as value slots.
            if value.casefold() in {
                "a",
                "all",
                "an",
                "and",
                "any",
                "as",
                "by",
                "each",
                "every",
                "for",
                "from",
                "in",
                "is",
                "our",
                "or",
                "record",
                "records",
                "row",
                "rows",
                "the",
                "there",
                "total",
                "where",
                "whose",
                "with",
            }:
                return
        occupied.append((start, end))
        found.append((start, value))

    for match in re.finditer(r"'((?:''|[^'])*)'", question):
        add(match.start(), match.end(), match.group(1).replace("''", "'"))
    for match in re.finditer(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])", question):
        raw = match.group(0)
        add(match.start(), match.end(), float(raw) if "." in raw else int(raw))

    comparison = re.compile(
        r"\bcolumn\d+\b\s+(?:set\s+to|marked\s+as|equal\s+to|equals|matches|exceeds|"
        r"is\s+(?:above|over|greater\s+than|more\s+than|below|under|less\s+than|at\s+least|at\s+most|"
        r"no\s+less\s+than|no\s+more\s+than)|"
        r"is(?!\s+(?:above|over|greater\s+than|more\s+than|below|under|less\s+than|at\s+least|at\s+most|"
        r"no\s+less\s+than|no\s+more\s+than)))"
        r"\s+([A-Za-z][A-Za-z0-9_-]*)",
        re.IGNORECASE,
    )
    for match in comparison.finditer(question):
        add(match.start(1), match.end(1), match.group(1))

    # Compact casual requests commonly omit an explicit equality word:
    # "customers with the name John", "the customer named John", and
    # "orders for customer John". Identifiers are slots by this point, which
    # makes these otherwise ambiguous constructions conservative to bind.
    nominal_comparison = re.compile(
        r"\b(?:with\s+(?:the\s+)?column\d+|"
        r"(?:table|column)\d+\s+(?:named|called)|"
        r"(?:for|with|to)\s+(?:the\s+)?table\d+)\s+"
        r"([A-Za-z][A-Za-z0-9_-]*)",
        re.IGNORECASE,
    )
    for match in nominal_comparison.finditer(question):
        add(match.start(1), match.end(1), match.group(1))

    # Natural location shorthand such as "customer city from Houston". Table
    # and column slots are explicitly excluded so ordinary "from customers"
    # phrasing cannot become a value.
    location = re.compile(
        r"\b(?:from|in|near|around|located\s+in|based\s+in|for)\s+"
        r"([A-Za-z][A-Za-z0-9_-]*)\b(?=\s*(?:[?.!,]|$|\bby\b))",
        re.IGNORECASE,
    )
    for match in location.finditer(question):
        add(match.start(1), match.end(1), match.group(1))

    unique: list[str | int | float] = []
    seen: set[tuple[type, str]] = set()
    for _, value in sorted(found, key=lambda item: item[0]):
        key = (type(value), str(value).casefold())
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


@dataclass(frozen=True)
class IdentifierMapping:
    table_to_slot: dict[str, str]
    column_to_slot: dict[str, str]
    column_to_tables: dict[str, tuple[str, ...]]
    primary_column_table: dict[str, str]
    foreign_keys: tuple[ForeignKeyBinding, ...] = ()
    literals: tuple[LiteralBinding, ...] = ()

    @property
    def raw_to_slot(self) -> dict[str, str]:
        overlap = set(self.table_to_slot) & set(self.column_to_slot)
        if overlap:
            raise ValueError(f"Table and column identifiers overlap and cannot be canonicalized safely: {sorted(overlap)}")
        return {**self.table_to_slot, **self.column_to_slot}

    @property
    def slot_to_raw(self) -> dict[str, str]:
        return {
            **{slot: raw for raw, slot in self.table_to_slot.items()},
            **{slot: raw for raw, slot in self.column_to_slot.items()},
        }

    @property
    def allowed_slots(self) -> set[str]:
        return set(self.slot_to_raw) | {binding.slot for binding in self.literals}

    def schema_linking_legend(
        self,
        question: str = "",
        canonical_question: str = "",
        max_tables: int = 5,
        max_columns: int = 8,
    ) -> str:
        """Map the most relevant stable slots to readable schema names.

        Question-mentioned slots are always retained. Lexically related labels
        come next, followed by a small schema sample. The cap keeps large
        real-world schemas inside the model's local context window.
        """
        table_links = sorted(
            ((slot, _identifier_label(raw)) for raw, slot in self.table_to_slot.items()),
            key=lambda item: int(item[0][5:]),
        )[:max_tables]
        all_column_links = sorted(
            ((slot, _identifier_label(raw)) for raw, slot in self.column_to_slot.items()),
            key=lambda item: int(item[0][6:]),
        )
        mentioned_slots = set(re.findall(r"\bcolumn\d+\b", canonical_question, re.IGNORECASE))
        question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))

        def relevance(item: tuple[str, str]) -> tuple[int, int]:
            slot, label = item
            label_words = set(label.split())
            if slot in mentioned_slots:
                priority = 0
            elif label_words & question_words:
                priority = 1
            else:
                priority = 2
            return priority, int(slot[6:])

        column_links = sorted(all_column_links, key=relevance)[:max_columns]
        column_links.sort(key=lambda item: int(item[0][6:]))
        links = ";".join(f"{slot}={label}" for slot, label in (*table_links, *column_links))
        return f"SCHEMA LINKS {links};"

    def declared_joins(self, first_table_slot: str, second_table_slot: str) -> tuple[tuple[str, str], ...]:
        """Return declared foreign-key joins between two canonical table slots."""
        joins = []
        for foreign_key in self.foreign_keys:
            child_table = self.table_to_slot.get(foreign_key.child_table)
            parent_table = self.table_to_slot.get(foreign_key.parent_table)
            if {child_table, parent_table} != {first_table_slot, second_table_slot}:
                continue
            child_column = self.column_to_slot.get(foreign_key.child_column)
            parent_column = self.column_to_slot.get(foreign_key.parent_column)
            if child_table and parent_table and child_column and parent_column:
                joins.append(
                    (
                        f"{child_table}.{child_column}",
                        f"{parent_table}.{parent_column}",
                    )
                )
        return tuple(dict.fromkeys(joins))

    def canonicalize(self, text: str) -> str:
        return _replace_outside_literals(text, self.raw_to_slot)

    def canonicalize_schema(self, schema_sql: str) -> str:
        """Canonicalize DDL while keeping table and column namespaces distinct."""
        table_lookup = {raw.casefold(): slot for raw, slot in self.table_to_slot.items()}
        column_lookup = {raw.casefold(): slot for raw, slot in self.column_to_slot.items()}
        if not set(table_lookup) & set(column_lookup):
            return self.canonicalize(schema_sql)

        def replace_table(raw: str) -> str:
            return table_lookup.get(raw.casefold(), raw)

        def replace_column(raw: str) -> str:
            return column_lookup.get(raw.casefold(), raw)

        def canonical_statement(match: re.Match) -> str:
            table_name, body = match.groups()
            fields = []
            for field in _split_fields(body):
                column_match = re.match(rf"({IDENTIFIER})\b", field)
                if column_match and column_match.group(1).upper() not in {
                    "PRIMARY",
                    "FOREIGN",
                    "UNIQUE",
                    "CHECK",
                    "CONSTRAINT",
                }:
                    start, end = column_match.span(1)
                    field = field[:start] + replace_column(column_match.group(1)) + field[end:]
                field = re.sub(
                    rf"\bREFERENCES\s+({IDENTIFIER})\s*\(\s*({IDENTIFIER})\s*\)",
                    lambda reference: (
                        f"REFERENCES {replace_table(reference.group(1))}"
                        f"({replace_column(reference.group(2))})"
                    ),
                    field,
                    flags=re.IGNORECASE,
                )
                fields.append(field)
            return f"CREATE TABLE {replace_table(table_name)} ({', '.join(fields)});"

        return CREATE_TABLE.sub(canonical_statement, schema_sql)

    def with_literals(self, values: list[str | int | float], question: str) -> "IdentifierMapping":
        """Attach stable valueN slots ordered by their first question mention."""
        def position(value: str | int | float) -> int:
            raw = str(value)
            # A bare ``find("4")`` can accidentally locate the digit in
            # ``column4`` before the actual LIMIT 4 mention.  Use lexical
            # boundaries so slot indices never influence value-slot order.
            pattern = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(raw)}(?![A-Za-z0-9_.])", re.IGNORECASE)
            match = pattern.search(question)
            return match.start() if match else -1

        positions = {(type(value), str(value).casefold()): position(value) for value in values}
        ordered = sorted(
            enumerate(values),
            key=lambda item: (
                positions[(type(item[1]), str(item[1]).casefold())] < 0,
                positions[(type(item[1]), str(item[1]).casefold())],
                item[0],
            ),
        )
        bindings: list[LiteralBinding] = []
        seen: set[tuple[type, str]] = set()
        for _, value in ordered:
            key = (type(value), str(value).casefold())
            if key in seen:
                continue
            seen.add(key)
            bindings.append(LiteralBinding(value, f"value{len(bindings)}"))
        return replace(self, literals=tuple(bindings))

    def canonicalize_sql(self, sql: str) -> str:
        table_lookup = {raw.casefold(): slot for raw, slot in self.table_to_slot.items()}
        column_lookup = {raw.casefold(): slot for raw, slot in self.column_to_slot.items()}
        if set(table_lookup) & set(column_lookup):
            canonical = re.sub(
                rf"\b(FROM|JOIN)\s+({IDENTIFIER})\b",
                lambda match: f"{match.group(1)} {table_lookup.get(match.group(2).casefold(), match.group(2))}",
                sql,
                flags=re.IGNORECASE,
            )
            canonical = re.sub(
                rf"\b({IDENTIFIER})\s*\.",
                lambda match: f"{table_lookup.get(match.group(1).casefold(), match.group(1))}.",
                canonical,
            )
            canonical = _replace_outside_literals(canonical, self.column_to_slot)
        else:
            canonical = self.canonicalize(sql)
        for binding in sorted(self.literals, key=lambda item: len(item.sql_text), reverse=True):
            if isinstance(binding.value, str):
                pattern = re.compile(re.escape(binding.sql_text), re.IGNORECASE)
            else:
                pattern = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(binding.sql_text)}(?![A-Za-z0-9_.])")
            canonical = pattern.sub(binding.slot, canonical)
        return canonical

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
        canonical = _replace_outside_literals(canonical, unambiguous)
        literal_replacements = {binding.question_text: binding.slot for binding in self.literals}
        canonical = _replace_outside_literals(canonical, literal_replacements)
        return self._ground_implicit_location(canonical)

    def _ground_implicit_location(self, question: str) -> str:
        """Expose an implicit location column when natural language omits it.

        Identifier canonicalization deliberately hides raw names from the
        model.  Without this schema-linking hint, a request such as "count
        customers in Boston" contains no information that distinguishes the
        shuffled city slot from every other TEXT slot.
        """
        if not self.literals:
            return question
        location_candidates = []
        for raw, slot in self.column_to_slot.items():
            normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw).casefold()
            words = {word for word in normalized.split("_") if word not in {"col", "data", "value"}}
            if words & LOCATION_IDENTIFIER_WORDS:
                location_candidates.append(slot)
        if len(set(location_candidates)) != 1:
            return question
        location_slot = location_candidates[0]
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(location_slot)}(?![A-Za-z0-9_])", question):
            return question
        for binding in self.literals:
            if re.search(
                rf"\b(?:from|in|near|around|located\s+in|based\s+in|for)\s+{re.escape(binding.slot)}\b",
                question,
                re.IGNORECASE,
            ):
                punctuation = question[-1] if question.endswith(("?", ".", "!")) else ""
                stem = question[:-1] if punctuation else question
                return f"{stem} by {location_slot}{punctuation}"
        return question

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
        literal_slots = {binding.slot: binding.sql_text for binding in self.literals}
        return _replace_outside_literals(_replace_outside_literals(text, literal_slots), self.slot_to_raw)

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
    foreign_keys: list[ForeignKeyBinding] = []
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
            reference = re.search(
                rf"\bREFERENCES\s+({IDENTIFIER})\s*\(\s*({IDENTIFIER})\s*\)",
                field,
                re.IGNORECASE,
            )
            if reference:
                parent_table, parent_column = reference.groups()
                foreign_keys.append(
                    ForeignKeyBinding(table_name, column_name, parent_table, parent_column)
                )
    if not tables or not columns:
        raise ValueError("Schema canonicalization requires CREATE TABLE statements with unquoted identifiers")
    if slot_strategy == "permuted":
        # Slot permutations must depend on schema structure, not presentation.
        # The normalized form intentionally matches Schema.sql() output, so
        # existing synthetic-training assignments remain unchanged while
        # pretty-printed and single-line user DDL map identically.
        fingerprint = _schema_fingerprint(schema_sql)

        def permuted(names: list[str], namespace: str) -> list[str]:
            return sorted(
                names,
                key=lambda name: hashlib.sha256(
                    f"{fingerprint}\0{namespace}\0{name.casefold()}".encode("utf-8")
                ).digest(),
            )

        tables = permuted(tables, "table")
        columns = permuted(columns, "column")
    return IdentifierMapping(
        {name: f"table{index}" for index, name in enumerate(tables)},
        {name: f"column{index}" for index, name in enumerate(columns)},
        {name: tuple(owners) for name, owners in column_to_tables.items()},
        primary_column_table,
        tuple(foreign_keys),
    )


def _schema_fingerprint(schema_sql: str) -> str:
    """Canonical DDL text for stable permuted-slot hashing."""
    statements = []
    for match in CREATE_TABLE.finditer(schema_sql):
        table_name, body = match.groups()
        fields = [" ".join(field.split()) for field in _split_fields(body)]
        statements.append(f"CREATE TABLE {table_name} ({', '.join(fields)});")
    return "\n".join(statements)


def canonicalize_inputs(
    schema_sql: str,
    question: str,
    slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
    schema_linking_hints: bool = False,
    schema_linking_max_tables: int = 5,
    schema_linking_max_columns: int = 8,
) -> tuple[str, str, IdentifierMapping]:
    mapping = identifier_mapping(schema_sql, slot_strategy)
    identifier_question = mapping.canonicalize_question(question)
    if canonicalize_literals:
        mapping = _with_question_literals(mapping, identifier_question)
    schema_source = _schema_fingerprint(schema_sql) if slot_strategy == "permuted" else schema_sql
    canonical_schema = mapping.canonicalize_schema(schema_source)
    canonical_question = mapping.canonicalize_question(question)
    if schema_linking_hints:
        canonical_schema += "\n" + mapping.schema_linking_legend(
            question,
            canonical_question,
            schema_linking_max_tables,
            schema_linking_max_columns,
        )
    return canonical_schema, canonical_question, mapping


def canonicalize_record(
    record: dict,
    slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
    schema_linking_hints: bool = False,
    schema_linking_max_tables: int = 5,
    schema_linking_max_columns: int = 8,
) -> dict:
    mapping = identifier_mapping(record["schema_sql"], slot_strategy)
    if canonicalize_literals:
        identifier_question = mapping.canonicalize_question(record["question"])
        mapping = _with_question_literals(mapping, identifier_question)
    canonical_schema = mapping.canonicalize_schema(
        _schema_fingerprint(record["schema_sql"]) if slot_strategy == "permuted" else record["schema_sql"]
    )
    canonical_question = mapping.canonicalize_question(record["question"])
    if schema_linking_hints:
        canonical_schema += "\n" + mapping.schema_linking_legend(
            record["question"],
            canonical_question,
            schema_linking_max_tables,
            schema_linking_max_columns,
        )
    canonical = {
        **record,
        "schema_sql": canonical_schema,
        "question": canonical_question,
        "sql": mapping.canonicalize_sql(record["sql"]),
    }
    if "query_plan" in record and "table" in record["query_plan"]:
        canonical["query_plan"] = _canonicalize_query_plan(record["query_plan"], mapping)
    return canonical


def _canonicalize_query_plan(plan: dict, mapping: IdentifierMapping) -> dict:
    """Apply the same reversible slots to the structured supervision target."""
    table_lookup = {raw.casefold(): slot for raw, slot in mapping.table_to_slot.items()}
    column_lookup = {raw.casefold(): slot for raw, slot in mapping.column_to_slot.items()}

    def table(value: str) -> str:
        return table_lookup.get(value.casefold(), value)

    def reference(value: str) -> str:
        qualifier, separator, column = value.rpartition(".")
        if separator:
            return f"{table(qualifier)}.{column_lookup.get(column.casefold(), column)}"
        if value == "*":
            return value
        return column_lookup.get(value.casefold(), value)

    canonical = dict(plan)
    for key in ("table", "join_table"):
        if canonical.get(key):
            canonical[key] = table(canonical[key])
    for key in ("aggregate_column", "order_by"):
        if canonical.get(key):
            canonical[key] = reference(canonical[key])
    for key in ("columns", "group_by"):
        canonical[key] = [reference(item) for item in canonical.get(key, ())]
    if canonical.get("join_on"):
        canonical["join_on"] = [reference(item) for item in canonical["join_on"]]

    def canonical_value(value: str | int | float) -> str | int | float:
        for binding in mapping.literals:
            if type(value) is type(binding.value) and str(value).casefold() == str(binding.value).casefold():
                return binding.slot
        return value

    canonical["filters"] = [
        {
            **item,
            "column": reference(item["column"]),
            "value": canonical_value(item["value"]),
        }
        for item in canonical.get("filters", ())
    ]
    return canonical


def _with_question_literals(mapping: IdentifierMapping, identifier_question: str) -> IdentifierMapping:
    """Bind only values inference can recover from the question itself.

    Gold query plans can contain ordinary text values that the conservative
    inference extractor cannot identify reliably. Leaving those values quoted
    in both the training target and inference output teaches direct copying
    without creating a train/inference slot mismatch. Values the extractor can
    identify still use reversible ``valueN`` slots.
    """
    identifier_aliases = {
        alias.casefold()
        for identifier in (*mapping.table_to_slot, *mapping.column_to_slot)
        for alias in _identifier_aliases(identifier)
    }
    return mapping.with_literals(
        _extract_question_literals(identifier_question, identifier_aliases),
        identifier_question,
    )
