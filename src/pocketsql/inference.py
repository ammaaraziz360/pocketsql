from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sqlite3

import mlx.core as mx

from pocketsql.data.validate import is_read_only_select
from pocketsql.data.render_sql import render_sql
from pocketsql.model.schema_grounding import LOCATION_IDENTIFIER_WORDS, IdentifierMapping, canonicalize_inputs
from pocketsql.model.semantic_grammar import SemanticPlanGrammar
from pocketsql.model.semantic_plan import SemanticPlanError, VALUE_SLOT_RE, parse_semantic_plan, serialize_semantic_plan
from pocketsql.model.tokenizer import TokenizerProtocol, load_tokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import load_checkpoint


def _generation_limit(model, max_tokens: int | None) -> int:
    limit = max_tokens if max_tokens is not None else getattr(model, "generation_max_tokens", 128)
    if limit < 1:
        raise ValueError("max_tokens must be positive")
    return limit


def _semantic_constraint(model, mapping: IdentifierMapping | None) -> SemanticPlanGrammar | None:
    if (
        getattr(model, "target_format", "sql") == "semantic_plan"
        and getattr(model, "constrain_semantic_plan", True)
        and mapping is not None
    ):
        return SemanticPlanGrammar.from_mapping(mapping)
    return None


def _token_pieces(tokenizer: TokenizerProtocol) -> tuple[str, ...]:
    return tuple(tokenizer.decode([identifier]) for identifier in range(tokenizer.vocab_size))


def _constrained_next_token(
    logits,
    target: str,
    grammar: SemanticPlanGrammar | None,
    tokenizer: TokenizerProtocol,
    pieces: tuple[str, ...] | None = None,
) -> int:
    if grammar is None:
        return int(mx.argmax(logits).item())
    pieces = pieces or _token_pieces(tokenizer)
    terminal_ids = {tokenizer.eos_id, tokenizer.sql_end_id}
    forbidden_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.sql_start_id}
    scores = logits.tolist()
    for identifier in sorted(range(len(scores)), key=scores.__getitem__, reverse=True):
        piece = pieces[identifier]
        if identifier in terminal_ids:
            if grammar.is_complete(target):
                return identifier
            continue
        if identifier in forbidden_ids or not piece or "<" in piece:
            continue
        if grammar.is_prefix(target + piece):
            return identifier
    # End safely and let the ordinary target validator reject the partial plan
    # instead of escaping the grammar or emitting an unsafe query.
    return tokenizer.sql_end_id


def _grounded_inputs(model, schema: str, question: str) -> tuple[str, str, IdentifierMapping | None]:
    if not getattr(model, "canonicalize_identifiers", False):
        return schema, question, None
    return canonicalize_inputs(
        schema,
        question,
        getattr(model, "identifier_slot_strategy", "ordered"),
        getattr(model, "canonicalize_literals", False),
        getattr(model, "schema_linking_hints", False),
    )


def _valid_for_schema(schema_sql: str, sql: str) -> bool:
    """Ask SQLite to resolve every table and column without running the query."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        connection.execute(f"EXPLAIN QUERY PLAN {sql}")
        return True
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def _repair_implicit_location_filter(
    sql: str,
    mapping: IdentifierMapping,
    grounded_question: str | None,
) -> str:
    """Repair one unambiguous, schema-invalid location predicate.

    A model can identify the selected location column and table but still put
    the location literal against a column owned by another table.  Only repair
    that failure when the question uses location shorthand, the SQL is a
    single-table query, and the selected table owns exactly one location-like
    column.  Valid model output and ambiguous cases are left untouched.
    """
    if not grounded_question or re.search(r"\b(?:JOIN|UNION|INTERSECT|EXCEPT)\b", sql, re.IGNORECASE):
        return sql
    from_tables = re.findall(r"\bFROM\s+(table\d+)\b", sql, re.IGNORECASE)
    if len(from_tables) != 1:
        return sql
    table_slot = from_tables[0]
    raw_table = mapping.slot_to_raw.get(table_slot)
    if raw_table not in mapping.table_to_slot:
        return sql

    predicates = list(
        re.finditer(
            r"(?:(table\d+)\s*\.\s*)?(column\d+)\s*=\s*(value\d+)\b",
            sql,
            re.IGNORECASE,
        )
    )
    if len(predicates) != 1:
        return sql
    predicate = predicates[0]
    qualifier, column_slot, value_slot = predicate.groups()
    if qualifier and qualifier.casefold() != table_slot.casefold():
        return sql
    if not re.search(
        rf"\b(?:from|in|near|around|located\s+in|based\s+in|for)\s+{re.escape(value_slot)}\b",
        grounded_question,
        re.IGNORECASE,
    ):
        return sql

    raw_column = mapping.slot_to_raw.get(column_slot)
    if raw_column and raw_table in mapping.column_to_tables.get(raw_column, ()):
        return sql

    location_slots = []
    for candidate, owners in mapping.column_to_tables.items():
        words = set(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", candidate).casefold().split("_"))
        if raw_table in owners and words & LOCATION_IDENTIFIER_WORDS:
            location_slots.append(mapping.column_to_slot[candidate])
    if len(set(location_slots)) != 1:
        return sql
    location_slot = location_slots[0]
    if not re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(location_slot)}(?![A-Za-z0-9_])",
        grounded_question,
    ):
        return sql
    start, end = predicate.span(2)
    return sql[:start] + location_slot + sql[end:]


def _finish_sql(
    sql: str,
    mapping: IdentifierMapping | None,
    schema_sql: str | None = None,
    grounded_question: str | None = None,
) -> str:
    if mapping is not None:
        sql = _repair_implicit_location_filter(sql, mapping, grounded_question)
        if not mapping.accepts_sql(sql):
            return ""
        sql = mapping.restore(sql)
    if not is_read_only_select(sql):
        return ""
    if schema_sql is not None and not _valid_for_schema(schema_sql, sql):
        return ""
    return sql


def _finish_target(
    target: str,
    model,
    mapping: IdentifierMapping | None,
    schema_sql: str,
    grounded_question: str,
) -> str:
    """Turn either a legacy SQL target or a semantic target into checked SQL."""
    if getattr(model, "target_format", "sql") == "sql":
        return _finish_sql(target, mapping, schema_sql, grounded_question)
    if getattr(model, "target_format", "sql") != "semantic_plan":
        return ""
    try:
        plan = parse_semantic_plan(target)
        if mapping is not None:
            plan = _ground_semantic_plan(plan, mapping, grounded_question)
        if mapping is not None and plan.filters:
            grounded_literals = {binding.slot for binding in mapping.literals}

            def supported(value: str | int | float) -> bool:
                if isinstance(value, str) and VALUE_SLOT_RE.fullmatch(value):
                    return value in grounded_literals
                raw = str(value)
                return bool(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])",
                        grounded_question,
                        re.IGNORECASE,
                    )
                )

            # A filter value absent from the request is necessarily a
            # hallucination. Removing only those predicates is a conservative
            # grounding constraint; mentioned literals and value slots remain.
            plan = replace(plan, filters=tuple(item for item in plan.filters if supported(item.value)))
        restored = mapping.restore(serialize_semantic_plan(plan)) if mapping is not None else serialize_semantic_plan(plan)
        sql = render_sql(parse_semantic_plan(restored))
    except SemanticPlanError:
        return ""
    return _finish_sql(sql, None, schema_sql)


def _ground_semantic_plan(plan, mapping: IdentifierMapping, question: str):
    """Apply conservative schema/question constraints to a decoded plan."""
    # Repeating a projection can never add information in a non-aggregate
    # SELECT, so collapse accidental duplicates while preserving order.
    columns = tuple(dict.fromkeys(plan.columns))

    if not plan.join_table and not plan.aggregate:
        # Remove spans that describe filter predicates, then regard remaining
        # column slots as explicit projections. This separates, for example,
        # "show city where name is John" from "show customers named John".
        projection_text = re.sub(
            r"(?:\bwith\s+(?:the\s+)?)?\bcolumn\d+\b\s*"
            r"(?:(?:set\s+to|marked\s+as|equal\s+to|equals|matches|exceeds|is(?:\s+(?:above|over|"
            r"greater\s+than|more\s+than|below|under|less\s+than|at\s+least|at\s+most|"
            r"no\s+less\s+than|no\s+more\s+than))?)\s+)?value\d+\b",
            " ",
            question,
            flags=re.IGNORECASE,
        )
        projection_text = re.sub(r"\bby\s+column\d+\b", " ", projection_text, flags=re.IGNORECASE)
        requested = tuple(dict.fromkeys(re.findall(r"\bcolumn\d+\b", projection_text, re.IGNORECASE)))
        if requested:
            columns = requested

    filter_hints = _explicit_filter_column_hints(question)
    if filter_hints and plan.filters:
        plan = replace(
            plan,
            filters=tuple(
                replace(item, column=filter_hints.get(str(item.value).casefold(), item.column))
                for item in plan.filters
            ),
        )

    mentioned_tables = tuple(dict.fromkeys(re.findall(r"\btable\d+\b", question, re.IGNORECASE)))
    if not plan.join_table and len(mentioned_tables) == 1:
        plan = replace(plan, table=mentioned_tables[0])

    if plan.join_table:
        joined_tables = {plan.table, plan.join_table}
        # In an adjacent table noun phrase ("customer orders"), the second
        # noun is the requested entity and the first supplies relationship
        # context. Do not apply this to prepositional forms such as
        # "orders for customers".
        compound = re.search(r"\b(table\d+)\s+(table\d+)\b", question, re.IGNORECASE)
        if compound and set(compound.groups()) == joined_tables:
            context_table, selected_table = compound.groups()
            if len(columns) == 1 and columns[0] in {f"{plan.table}.*", f"{plan.join_table}.*"}:
                columns = (f"{selected_table}.*",)
            plan = replace(plan, table=selected_table, join_table=context_table)

        # A column that exists on exactly one joined table cannot legally be
        # qualified by the other table. Correct only this unambiguous case.
        grounded_filters = []
        for item in plan.filters:
            _, separator, column_slot = item.column.rpartition(".")
            if not separator:
                column_slot = item.column
            raw_column = mapping.slot_to_raw.get(column_slot)
            owners = mapping.column_to_tables.get(raw_column, ()) if raw_column else ()
            owner_slots = [mapping.table_to_slot[owner] for owner in owners if mapping.table_to_slot[owner] in joined_tables]
            column = f"{owner_slots[0]}.{column_slot}" if len(owner_slots) == 1 else item.column
            grounded_filters.append(replace(item, column=column))
        plan = replace(plan, filters=tuple(grounded_filters))

    return replace(plan, columns=columns)


def _explicit_filter_column_hints(question: str) -> dict[str, str]:
    """Recover only direct column/value pairings stated in a grounded question.

    This deliberately ignores loose proximity. For example, ``average price
    before 2020`` may imply a year predicate without naming the year column;
    treating the nearby projection as the filter would be worse than leaving
    the model's decision alone.
    """
    column = r"column\d+"
    value = r"value\d+"
    comparison = (
        r"(?:is|equals|matches|equal\s+to|set\s+to|marked\s+as|above|over|exceeds|"
        r"greater\s+than|more\s+than|below|under|less\s+than|at\s+least|at\s+most|"
        r"no\s+less\s+than|no\s+more\s+than)"
    )
    candidates: dict[str, set[str]] = {}

    def add(column_slot: str, value_slot: str) -> None:
        candidates.setdefault(value_slot.casefold(), set()).add(column_slot)

    for match in re.finditer(
        rf"\b({column})\b\s+(?:{comparison}\s+)?\b({value})\b",
        question,
        re.IGNORECASE,
    ):
        add(match.group(1), match.group(2))
    for match in re.finditer(
        rf"\b({value})\b\s+(?:by\s+)?\b({column})\b",
        question,
        re.IGNORECASE,
    ):
        add(match.group(2), match.group(1))
    for match in re.finditer(
        rf"\b({column})\b\s+(?:{comparison}\s+)?\b({value})\b\s+(?:and|or)\s+\b({value})\b",
        question,
        re.IGNORECASE,
    ):
        add(match.group(1), match.group(2))
        add(match.group(1), match.group(3))
    return {value_slot: next(iter(columns)) for value_slot, columns in candidates.items() if len(columns) == 1}


def generate_sql(model, schema: str, question: str, tokenizer: TokenizerProtocol | None = None, max_tokens: int | None = None) -> str:
    tokenizer = tokenizer or load_tokenizer()
    source_schema = schema
    schema, question, mapping = _grounded_inputs(model, schema, question)
    prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= model.config.context_length:
        raise ValueError(f"prompt uses {len(tokens)} tokens but model context length is {model.config.context_length}")
    budget = min(_generation_limit(model, max_tokens), model.config.context_length - len(tokens))
    grammar = _semantic_constraint(model, mapping)
    pieces = _token_pieces(tokenizer) if grammar is not None else None
    generated: list[int] = []
    for _ in range(budget):
        logits = model(mx.array([tokens]))
        target_prefix = tokenizer.decode(generated)
        next_token = _constrained_next_token(logits[0, -1], target_prefix, grammar, tokenizer, pieces)
        tokens.append(next_token)
        generated.append(next_token)
        if next_token in (tokenizer.eos_id, tokenizer.sql_end_id):
            break
    text = tokenizer.decode(tokens)
    target = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
    finished = _finish_target(target, model, mapping, source_schema, question)
    if not finished:
        target_name = getattr(model, "target_format", "sql")
        raise ValueError(f"model output is not a valid {target_name} target: {target!r}")
    return finished


def generate_sql_batch(
    model,
    schemas: list[str],
    questions: list[str],
    tokenizer: TokenizerProtocol | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Generate safe SQL for a batch of prompts using right padding under causal attention.

    Each row's next-token logits are read at its last non-padding position, so future
    right-padding cannot influence the generated result. Invalid generations are
    returned as empty strings, matching evaluation's existing failure behavior.
    """
    if len(schemas) != len(questions):
        raise ValueError("schemas and questions must have the same length")
    if not schemas:
        return []
    tokenizer = tokenizer or load_tokenizer()
    max_tokens = _generation_limit(model, max_tokens)
    grounded = [_grounded_inputs(model, schema, question) for schema, question in zip(schemas, questions)]
    grammars = [_semantic_constraint(model, mapping) for _, _, mapping in grounded]
    pieces = _token_pieces(tokenizer) if any(grammar is not None for grammar in grammars) else None
    prompts = [
        tokenizer.encode(f"<bos><schema>{schema}</schema><question>{question}</question><sql>")
        for schema, question, _ in grounded
    ]
    sequences = [list(prompt) for prompt in prompts]
    groups: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        groups.setdefault(len(prompt), []).append(index)
    for prompt_length, indices in groups.items():
        available = model.config.context_length - prompt_length
        if available > 0:
            _generate_with_cache(
                model,
                sequences,
                indices,
                tokenizer,
                min(max_tokens, available),
                prompt_length,
                grammars,
                pieces,
            )

    outputs = []
    for tokens, source_schema, (_, grounded_question, mapping) in zip(sequences, schemas, grounded):
        text = tokenizer.decode(tokens)
        target = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
        outputs.append(_finish_target(target, model, mapping, source_schema, grounded_question))
    return outputs


def _generate_with_cache(
    model,
    sequences: list[list[int]],
    indices: list[int],
    tokenizer: TokenizerProtocol,
    max_tokens: int,
    prompt_length: int,
    grammars: list[SemanticPlanGrammar | None] | None = None,
    pieces: tuple[str, ...] | None = None,
) -> None:
    prompt_tokens = mx.array([sequences[index] for index in indices])
    logits, cache = model.forward_with_cache(prompt_tokens)
    mx.eval(logits, *[value for block_cache in cache for value in block_cache])
    active = [True] * len(indices)
    for _ in range(max_tokens):
        next_tokens = []
        for row, index in enumerate(indices):
            if not active[row]:
                next_tokens.append(tokenizer.pad_id)
                continue
            target_prefix = tokenizer.decode(sequences[index][prompt_length:])
            grammar = grammars[index] if grammars is not None else None
            next_tokens.append(
                _constrained_next_token(logits[row, -1], target_prefix, grammar, tokenizer, pieces)
            )
        inputs = []
        for row, (index, next_token) in enumerate(zip(indices, next_tokens)):
            if active[row]:
                sequences[index].append(next_token)
                if next_token in (tokenizer.eos_id, tokenizer.sql_end_id):
                    active[row] = False
            inputs.append(next_token if active[row] else tokenizer.pad_id)
        if not any(active):
            break
        logits, cache = model.forward_with_cache(mx.array(inputs)[:, None], cache)
        mx.eval(logits, *[value for block_cache in cache for value in block_cache])


def _generate_without_cache(model, sequences: list[list[int]], indices: list[int], tokenizer: TokenizerProtocol, max_tokens: int) -> None:
    """Preserve sliding-window behavior for unusually long prompts."""
    active = list(indices)
    for _ in range(max_tokens):
        if not active:
            break
        windows = [sequences[index][-model.config.context_length :] for index in active]
        width = max(len(window) for window in windows)
        batch = [window + [tokenizer.pad_id] * (width - len(window)) for window in windows]
        logits = model(mx.array(batch))
        next_tokens = [int(mx.argmax(logits[row, len(window) - 1]).item()) for row, window in enumerate(windows)]
        remaining = []
        for index, next_token in zip(active, next_tokens):
            sequences[index].append(next_token)
            if next_token not in (tokenizer.eos_id, tokenizer.sql_end_id):
                remaining.append(index)
        active = remaining


def load_model_from_checkpoint(checkpoint: str, tokenizer: TokenizerProtocol | None = None) -> PocketSQLTransformer:
    tokenizer = tokenizer or load_tokenizer(Path(checkpoint))
    metadata_path = Path(checkpoint)
    config = json.loads((metadata_path / "metadata.json").read_text(encoding="utf-8"))["config"]
    model = PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=config["layers"], hidden_dim=config["hidden_dim"], heads=config["heads"], ffn_dim=config["ffn_dim"], context_length=config["context_length"]))
    model.generation_max_tokens = config.get("generation_max_tokens", 128)
    model.canonicalize_identifiers = config.get("canonicalize_identifiers", False)
    model.identifier_slot_strategy = config.get("identifier_slot_strategy", "ordered")
    model.canonicalize_literals = config.get("canonicalize_literals", False)
    model.target_format = config.get("target_format", "sql")
    model.constrain_semantic_plan = config.get("constrain_semantic_plan", model.target_format == "semantic_plan")
    model.schema_linking_hints = config.get("schema_linking_hints", False)
    load_checkpoint(metadata_path, model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override the generation cap stored in the checkpoint.")
    parser.add_argument(
        "--unconstrained-semantic-plan",
        action="store_true",
        help="Disable semantic-plan grammar constraints for legacy score reproduction.",
    )
    args = parser.parse_args()
    tokenizer = load_tokenizer(Path(args.checkpoint))
    model = load_model_from_checkpoint(args.checkpoint, tokenizer)
    if args.unconstrained_semantic_plan:
        model.constrain_semantic_plan = False
    print(generate_sql(model, args.schema, args.question, tokenizer, args.max_tokens))


if __name__ == "__main__":
    main()
