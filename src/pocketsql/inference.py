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
from pocketsql.model.factorized import (
    FactorizedPocketSQLTransformer,
    FactorizedSchemaConfig,
    decode_schema_link_logits,
)
from pocketsql.model.semantic_grammar import SemanticPlanGrammar
from pocketsql.model.semantic_plan import SemanticPlanError, VALUE_SLOT_RE, parse_semantic_plan, serialize_semantic_plan
from pocketsql.model.structured import (
    StructuredPocketSQLTransformer,
    StructuredQueryConfig,
    decode_literal_logits,
    decode_operation_logits,
    prompt_layout,
    structured_query_plan,
)
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
        getattr(model, "schema_linking_max_tables", 5),
        getattr(model, "schema_linking_max_columns", 8),
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
    factorized_links: dict | None = None,
    factorized_confidence_threshold: float | None = None,
) -> str:
    """Turn either a legacy SQL target or a semantic target into checked SQL."""
    if getattr(model, "target_format", "sql") == "sql":
        return _finish_sql(target, mapping, schema_sql, grounded_question)
    if getattr(model, "target_format", "sql") != "semantic_plan":
        return ""
    try:
        plan = parse_semantic_plan(target)
        if mapping is not None and factorized_links is not None:
            plan = _apply_factorized_schema_links(
                plan,
                factorized_links,
                mapping,
                factorized_confidence_threshold,
            )
        if mapping is not None:
            plan = _ground_semantic_plan(plan, mapping, grounded_question)
        if mapping is not None and plan.filters:
            grounded_literals = {binding.slot for binding in mapping.literals}

            def supported(value: str | int | float) -> bool:
                if isinstance(value, str) and VALUE_SLOT_RE.fullmatch(value):
                    return value in grounded_literals
                raw = str(value)
                canonical_raw = mapping.canonicalize_question(raw)
                return any(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])",
                        grounded_question,
                        re.IGNORECASE,
                    )
                    for candidate in {raw, canonical_raw}
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


def _apply_factorized_schema_links(
    plan,
    links: dict,
    mapping: IdentifierMapping,
    confidence_threshold: float | None = None,
):
    """Substitute schema roles, optionally keeping every low-confidence role."""
    confidence = links.get("confidence", {})

    def confident(role: str, index: int | None = None) -> bool:
        if confidence_threshold is None:
            return True
        value = confidence.get(role, 0.0)
        if index is not None:
            value = value[index] if index < len(value) else 0.0
        return float(value) >= confidence_threshold

    predicted_table = f"table{links['table']}"
    predicted_join = f"table{links['join_table']}" if links["join_table"] is not None else None
    replace_scope = confident("table_join")
    table = predicted_table if replace_scope else plan.table
    join_table = predicted_join if replace_scope else plan.join_table
    joined = join_table is not None
    star_column = links["star_column"]
    predicted_scope = {predicted_table, predicted_join} - {None}
    selected_scope = {table, join_table} - {None}
    role_scope_matches = predicted_scope == selected_scope

    def reference(owner: int, column: int, allow_star: bool = False) -> str:
        owner_slot = f"table{owner}"
        if allow_star and column == star_column:
            return f"{owner_slot}.*" if joined else "*"
        column_slot = f"column{column}"
        return f"{owner_slot}.{column_slot}" if joined else column_slot

    columns = tuple(
        reference(links["projection_owner"][index], links["projection_column"][index], True)
        if role_scope_matches
        and index < len(links["projection_column"])
        and confident("projection", index)
        else item
        for index, item in enumerate(plan.columns)
    )
    aggregate_column = plan.aggregate_column
    if aggregate_column and role_scope_matches and confident("aggregate"):
        aggregate_column = reference(links["aggregate_owner"], links["aggregate_column"])
    filters = tuple(
        replace(
            item,
            column=reference(links["filter_owner"][index], links["filter_column"][index]),
        )
        if role_scope_matches
        and index < len(links["filter_column"])
        and confident("filter", index)
        else item
        for index, item in enumerate(plan.filters)
    )
    group_by = tuple(
        reference(links["group_owner"][index], links["group_column"][index])
        if role_scope_matches
        and index < len(links["group_column"])
        and confident("group", index)
        else item
        for index, item in enumerate(plan.group_by)
    )
    order_by = plan.order_by
    if order_by and role_scope_matches and confident("order"):
        order_by = reference(links["order_owner"], links["order_column"])

    join_on = plan.join_on if joined else None
    if join_table:
        declared = mapping.declared_joins(table, join_table)
        if len(declared) == 1:
            join_on = declared[0]
        elif replace_scope and confident("join_column"):
            join_on = (
                reference(links["table"], links["join_column"][0]),
                reference(links["join_table"], links["join_column"][1]),
            )
    return replace(
        plan,
        table=table,
        join_table=join_table,
        join_on=join_on,
        columns=columns,
        aggregate_column=aggregate_column,
        filters=filters,
        group_by=group_by,
        order_by=order_by,
    )


def _factorized_link_predictions(model, prompts, grounded, tokenizer: TokenizerProtocol | None = None):
    if not isinstance(model, FactorizedPocketSQLTransformer) or not getattr(
        model, "use_factorized_schema_links", True
    ):
        return [None] * len(prompts)
    predictions = [None] * len(prompts)
    if getattr(model, "factorized_schema_linking_hints", False):
        if tokenizer is None:
            raise ValueError("factorized schema label prompts require a tokenizer")
        labeled_prompts = []
        for original_prompt, (schema, question, mapping) in zip(prompts, grounded):
            labeled_prompt = tokenizer.encode(
                f"<bos><schema>{schema}\n"
                f"{mapping.schema_linking_legend('', question, getattr(model, 'schema_linking_max_tables', 5), getattr(model, 'schema_linking_max_columns', 8))}"
                f"</schema><question>{question}</question><sql>"
            )
            # Readable labels are an auxiliary schema-linking channel.  Large
            # real-world schemas can make that channel longer than the model's
            # position table even when the canonical generation prompt fits.
            # Falling back to the canonical prompt keeps inference safe and
            # preserves the pre-v15 behaviour for those records.
            if len(labeled_prompt) > model.config.context_length:
                labeled_prompt = original_prompt
            labeled_prompts.append(labeled_prompt)
        prompts = labeled_prompts
    groups: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        groups.setdefault(len(prompt), []).append(index)
    for indices in groups.values():
        mappings = [grounded[index][2] for index in indices]
        if any(mapping is None for mapping in mappings):
            raise ValueError("factorized schema linking requires canonicalized identifiers")
        prompt_batch = mx.array([prompts[index] for index in indices])
        if isinstance(model, StructuredPocketSQLTransformer):
            prompt_texts = [tokenizer.decode(prompts[index]) for index in indices]
            layouts = [
                prompt_layout(
                    prompt_text,
                    tokenizer,
                    grounded[index][2],
                    model.schema_config,
                    model.structured_config,
                )
                for index, prompt_text in zip(indices, prompt_texts)
            ]
            layout_keys = (
                "table_positions",
                "table_mask",
                "column_positions",
                "column_mask",
                "question_mask",
            )
            layout_batch = {
                name: mx.array([layout[name] for layout in layouts]) for name in layout_keys
            }
            layout_batch["prompt_positions"] = mx.array(
                [layout["prompt_position"] for layout in layouts]
            )
            logits, operation_logits, literal_logits = model.structured_logits(
                prompt_batch, layout_batch
            )
            mx.eval(
                *logits.values(),
                *operation_logits.values(),
                *literal_logits.values(),
            )
            operations = decode_operation_logits(operation_logits)
            literals = decode_literal_logits(
                literal_logits,
                prompt_texts,
                layouts,
                model.structured_config.max_literal_span_tokens,
            )
        else:
            logits = model.schema_link_logits(prompt_batch)
            mx.eval(*logits.values())
            operations = [None] * len(indices)
            literals = [None] * len(indices)
        decoded = decode_schema_link_logits(logits, mappings, model.schema_config)
        for index, links, operation, literal_slots, mapping in zip(
            indices, decoded, operations, literals, mappings
        ):
            if operation is not None and literal_slots is not None:
                links["structured_plan"] = structured_query_plan(
                    operation,
                    links,
                    literal_slots,
                    mapping,
                    model.structured_config,
                )
                links["structured_operation"] = operation
                links["structured_literal_values"] = literal_slots
            predictions[index] = links
    return predictions


def _finish_generated_target(
    target: str,
    model,
    mapping: IdentifierMapping | None,
    schema_sql: str,
    grounded_question: str,
    factorized_links: dict | None,
) -> str:
    if factorized_links is None:
        return _finish_target(target, model, mapping, schema_sql, grounded_question)
    structured_plan = factorized_links.get("structured_plan")
    if isinstance(model, StructuredPocketSQLTransformer):
        mode = getattr(model, "structured_plan_mode", "fallback")
        structured_sql = ""
        if structured_plan is not None:
            try:
                structured_target = serialize_semantic_plan(structured_plan)
            except SemanticPlanError:
                structured_target = ""
            if structured_target:
                structured_sql = _finish_target(
                    structured_target,
                    model,
                    mapping,
                    schema_sql,
                    grounded_question,
                )
        if mode == "replace":
            return structured_sql
        baseline = _finish_target(target, model, mapping, schema_sql, grounded_question)
        if mode == "prefer":
            return structured_sql or baseline
        if mode == "fallback":
            return baseline or structured_sql
        if mode != "disabled":
            raise ValueError(f"unsupported structured_plan_mode: {mode!r}")
    mode = getattr(model, "factorized_schema_mode", "fallback")
    if mode == "fallback":
        baseline = _finish_target(target, model, mapping, schema_sql, grounded_question)
        return baseline or _finish_target(
            target,
            model,
            mapping,
            schema_sql,
            grounded_question,
            factorized_links,
        )
    if mode == "replace":
        return _finish_target(
            target,
            model,
            mapping,
            schema_sql,
            grounded_question,
            factorized_links,
        )
    if mode == "confidence":
        threshold = getattr(model, "factorized_schema_confidence_threshold", 0.9)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("factorized_schema_confidence_threshold must be between zero and one")
        baseline = _finish_target(target, model, mapping, schema_sql, grounded_question)
        if baseline:
            return baseline
        return _finish_target(
            target,
            model,
            mapping,
            schema_sql,
            grounded_question,
            factorized_links,
            threshold,
        )
    if mode == "disabled":
        return _finish_target(target, model, mapping, schema_sql, grounded_question)
    raise ValueError(f"unsupported factorized_schema_mode: {mode!r}")


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

    if plan.aggregate or plan.group_by:
        # Aggregate/group joins remain model-directed for now. Their table
        # semantics are more subtle than plain relationship projections, and
        # the deterministic planner must not trade away proven v12 behavior.
        if plan.join_table:
            joined_tables = {plan.table, plan.join_table}
            plan = replace(
                plan,
                filters=tuple(
                    replace(item, column=_qualify_join_reference(item.column, joined_tables, mapping))
                    for item in plan.filters
                ),
            )
        return replace(plan, columns=columns)

    referenced_tables = _plan_reference_tables(plan, columns, mapping)
    inferred_pair: tuple[str, str] | None = None
    inferred_join: tuple[str, str] | None = None
    decoded_pair = (plan.table, plan.join_table) if plan.join_table else ()
    decoded_joins = mapping.declared_joins(*decoded_pair) if len(decoded_pair) == 2 else ()
    if len(decoded_joins) == 1:
        inferred_pair = decoded_pair
        inferred_join = decoded_joins[0]
    elif plan.join_table:
        candidates = (
            mentioned_tables if len(mentioned_tables) == 2 else (),
            referenced_tables if len(referenced_tables) == 2 else (),
        )
    else:
        # Projection grounding above has already removed unrequested model
        # columns, so two remaining unique owner tables are direct evidence
        # that the requested projection requires a join.
        candidates = (
            mentioned_tables if len(mentioned_tables) == 2 else (),
            referenced_tables if len(referenced_tables) == 2 else (),
        )
    for candidate in candidates if inferred_pair is None else ():
        if len(candidate) != 2 or candidate[0] == candidate[1]:
            continue
        joins = mapping.declared_joins(candidate[0], candidate[1])
        if len(joins) == 1:
            inferred_pair = (candidate[0], candidate[1])
            inferred_join = joins[0]
            break

    if inferred_pair and inferred_join:
        base_table = plan.table if plan.table in inferred_pair else inferred_pair[0]
        # In an adjacent table noun phrase ("customer orders"), the second
        # noun is the requested entity and the first supplies relationship
        # context. Do not apply this to prepositional forms such as
        # "orders for customers".
        compound = re.search(r"\b(table\d+)\s+(table\d+)\b", question, re.IGNORECASE)
        if compound and set(compound.groups()) == set(inferred_pair):
            _, base_table = compound.groups()
        else:
            wildcard_tables = tuple(
                reference.split(".", 1)[0]
                for reference in columns
                if reference.endswith(".*") and reference.split(".", 1)[0] in inferred_pair
            )
            if len(wildcard_tables) == 1:
                base_table = wildcard_tables[0]
        join_table = inferred_pair[1] if base_table == inferred_pair[0] else inferred_pair[0]
        if compound and len(columns) == 1 and columns[0].endswith(".*"):
            columns = (f"{base_table}.*",)
        plan = replace(
            plan,
            table=base_table,
            join_table=join_table,
            join_on=inferred_join,
        )

    if plan.join_table:
        joined_tables = {plan.table, plan.join_table}
        declared = mapping.declared_joins(plan.table, plan.join_table)
        if not declared:
            raise SemanticPlanError("decoded join tables are not connected by a declared foreign key")
        decoded_join = tuple(plan.join_on or ())
        if len(declared) > 1 and not any(decoded_join in {join, join[::-1]} for join in declared):
            raise SemanticPlanError("decoded join is ambiguous across multiple declared foreign keys")
        if len(declared) == 1:
            plan = replace(plan, join_on=declared[0])

        # A column owned by exactly one joined table is safe to qualify or to
        # repair when the decoder attached the other table.
        columns = tuple(_qualify_join_reference(item, joined_tables, mapping) for item in columns)
        aggregate_column = (
            _qualify_join_reference(plan.aggregate_column, joined_tables, mapping)
            if plan.aggregate_column
            else None
        )
        filters = tuple(
            replace(item, column=_qualify_join_reference(item.column, joined_tables, mapping))
            for item in plan.filters
        )
        group_by = tuple(_qualify_join_reference(item, joined_tables, mapping) for item in plan.group_by)
        order_by = (
            _qualify_join_reference(plan.order_by, joined_tables, mapping)
            if plan.order_by
            else None
        )
        plan = replace(
            plan,
            aggregate_column=aggregate_column,
            filters=filters,
            group_by=group_by,
            order_by=order_by,
        )

    return replace(plan, columns=columns)


def _reference_owner_tables(reference: str, mapping: IdentifierMapping) -> tuple[str, ...]:
    qualifier, separator, column_slot = reference.rpartition(".")
    if separator and column_slot == "*":
        return (qualifier,) if qualifier in mapping.table_to_slot.values() else ()
    if not separator:
        column_slot = reference
    raw_column = mapping.slot_to_raw.get(column_slot)
    if not raw_column:
        return ()
    owners = tuple(
        mapping.table_to_slot[owner]
        for owner in mapping.column_to_tables.get(raw_column, ())
        if owner in mapping.table_to_slot
    )
    if len(owners) == 1:
        return owners
    if separator and qualifier in owners:
        return (qualifier,)
    return ()


def _plan_reference_tables(plan, columns: tuple[str, ...], mapping: IdentifierMapping) -> tuple[str, ...]:
    references = [*columns, *plan.group_by]
    if plan.aggregate_column:
        references.append(plan.aggregate_column)
    if plan.order_by:
        references.append(plan.order_by)
    references.extend(item.column for item in plan.filters)
    tables = []
    for reference in references:
        for table in _reference_owner_tables(reference, mapping):
            if table not in tables:
                tables.append(table)
    return tuple(tables)


def _qualify_join_reference(reference: str, joined_tables: set[str], mapping: IdentifierMapping) -> str:
    qualifier, separator, column_slot = reference.rpartition(".")
    if separator and column_slot == "*":
        return reference
    if not separator:
        column_slot = reference
    raw_column = mapping.slot_to_raw.get(column_slot)
    owners = mapping.column_to_tables.get(raw_column, ()) if raw_column else ()
    owner_slots = [
        mapping.table_to_slot[owner]
        for owner in owners
        if mapping.table_to_slot.get(owner) in joined_tables
    ]
    return f"{owner_slots[0]}.{column_slot}" if len(owner_slots) == 1 else reference


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
    factorized_links = _factorized_link_predictions(
        model, [tokens], [(schema, question, mapping)], tokenizer
    )[0]
    if isinstance(model, StructuredPocketSQLTransformer) and getattr(
        model, "structured_plan_mode", "fallback"
    ) == "replace":
        finished = _finish_generated_target(
            "", model, mapping, source_schema, question, factorized_links
        )
        if not finished:
            raise ValueError("structured heads did not produce a valid semantic_plan target")
        return finished
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
    finished = _finish_generated_target(
        target,
        model,
        mapping,
        source_schema,
        question,
        factorized_links,
    )
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
    outputs, _ = generate_sql_batch_with_targets(
        model,
        schemas,
        questions,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    return outputs


def generate_sql_batch_with_targets(
    model,
    schemas: list[str],
    questions: list[str],
    tokenizer: TokenizerProtocol | None = None,
    max_tokens: int | None = None,
) -> tuple[list[str], list[str]]:
    """Generate checked SQL and retain each raw decoded target for diagnostics."""
    outputs, targets, _ = generate_sql_batch_with_targets_and_links(
        model,
        schemas,
        questions,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    return outputs, targets


def generate_sql_batch_with_targets_and_links(
    model,
    schemas: list[str],
    questions: list[str],
    tokenizer: TokenizerProtocol | None = None,
    max_tokens: int | None = None,
) -> tuple[list[str], list[str], list[dict | None]]:
    """Retain checked SQL, raw targets, and factorized schema-role predictions."""
    if len(schemas) != len(questions):
        raise ValueError("schemas and questions must have the same length")
    if not schemas:
        return [], [], []
    tokenizer = tokenizer or load_tokenizer()
    max_tokens = _generation_limit(model, max_tokens)
    grounded = [_grounded_inputs(model, schema, question) for schema, question in zip(schemas, questions)]
    grammars = [_semantic_constraint(model, mapping) for _, _, mapping in grounded]
    pieces = _token_pieces(tokenizer) if any(grammar is not None for grammar in grammars) else None
    prompts = [
        tokenizer.encode(f"<bos><schema>{schema}</schema><question>{question}</question><sql>")
        for schema, question, _ in grounded
    ]
    factorized_predictions = _factorized_link_predictions(model, prompts, grounded, tokenizer)
    if isinstance(model, StructuredPocketSQLTransformer) and getattr(
        model, "structured_plan_mode", "fallback"
    ) == "replace":
        outputs = [
            _finish_generated_target(
                "",
                model,
                mapping,
                source_schema,
                grounded_question,
                factorized_links,
            )
            for source_schema, (_, grounded_question, mapping), factorized_links in zip(
                schemas, grounded, factorized_predictions
            )
        ]
        return outputs, [""] * len(outputs), factorized_predictions
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
    targets = []
    for tokens, source_schema, (_, grounded_question, mapping), factorized_links in zip(
        sequences,
        schemas,
        grounded,
        factorized_predictions,
    ):
        text = tokenizer.decode(tokens)
        target = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
        targets.append(target)
        outputs.append(
            _finish_generated_target(
                target,
                model,
                mapping,
                source_schema,
                grounded_question,
                factorized_links,
            )
        )
    return outputs, targets, factorized_predictions


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
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, layers=config["layers"], hidden_dim=config["hidden_dim"], heads=config["heads"], ffn_dim=config["ffn_dim"], context_length=config["context_length"])
    architecture = config.get("model_architecture", "autoregressive")
    if architecture == "factorized_schema":
        model = FactorizedPocketSQLTransformer(
            model_config,
            FactorizedSchemaConfig(
                max_table_slots=config.get("max_table_slots", 16),
                max_column_slots=config.get("max_column_slots", 64),
                max_projection_slots=config.get("max_projection_slots", 4),
                max_filter_slots=config.get("max_filter_slots", 4),
                max_group_slots=config.get("max_group_slots", 2),
            ),
        )
    elif architecture == "structured_v18":
        model = StructuredPocketSQLTransformer(
            model_config,
            FactorizedSchemaConfig(
                max_table_slots=config.get("max_table_slots", 16),
                max_column_slots=config.get("max_column_slots", 64),
                max_projection_slots=config.get("max_projection_slots", 4),
                max_filter_slots=config.get("max_filter_slots", 4),
                max_group_slots=config.get("max_group_slots", 2),
            ),
            StructuredQueryConfig(
                max_literal_span_tokens=config.get("max_literal_span_tokens", 12),
                max_limit_value=config.get("max_limit_value", 100),
            ),
        )
    elif architecture == "autoregressive":
        model = PocketSQLTransformer(model_config)
    else:
        raise ValueError(f"unsupported model_architecture in checkpoint: {architecture!r}")
    model.model_architecture = architecture
    model.factorized_schema_mode = config.get("factorized_schema_mode", "fallback")
    model.factorized_schema_confidence_threshold = config.get(
        "factorized_schema_confidence_threshold", 0.9
    )
    model.generation_max_tokens = config.get("generation_max_tokens", 128)
    model.canonicalize_identifiers = config.get("canonicalize_identifiers", False)
    model.identifier_slot_strategy = config.get("identifier_slot_strategy", "ordered")
    model.canonicalize_literals = config.get("canonicalize_literals", False)
    model.target_format = config.get("target_format", "sql")
    model.constrain_semantic_plan = config.get("constrain_semantic_plan", model.target_format == "semantic_plan")
    model.schema_linking_hints = config.get("schema_linking_hints", False)
    model.factorized_schema_linking_hints = config.get("factorized_schema_linking_hints", False)
    model.schema_linking_max_tables = config.get("schema_linking_max_tables", 5)
    model.schema_linking_max_columns = config.get("schema_linking_max_columns", 8)
    model.structured_plan_mode = config.get("structured_plan_mode", "fallback")
    load_checkpoint(metadata_path, model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override the generation cap stored in the checkpoint.")
    parser.add_argument(
        "--factorized-schema-mode",
        choices=("fallback", "replace", "confidence", "disabled"),
        default=None,
        help="Override how factorized schema pointers are applied for this run.",
    )
    parser.add_argument(
        "--factorized-schema-confidence-threshold",
        type=float,
        default=None,
        help="Minimum constrained pointer confidence in confidence mode.",
    )
    parser.add_argument(
        "--structured-plan-mode",
        choices=("replace", "prefer", "fallback", "disabled"),
        default=None,
        help="Override how v18 structured plans are combined with autoregressive plans.",
    )
    parser.add_argument(
        "--unconstrained-semantic-plan",
        action="store_true",
        help="Disable semantic-plan grammar constraints for legacy score reproduction.",
    )
    args = parser.parse_args()
    tokenizer = load_tokenizer(Path(args.checkpoint))
    model = load_model_from_checkpoint(args.checkpoint, tokenizer)
    if args.factorized_schema_mode is not None:
        model.factorized_schema_mode = args.factorized_schema_mode
    if args.factorized_schema_confidence_threshold is not None:
        if not 0.0 <= args.factorized_schema_confidence_threshold <= 1.0:
            raise SystemExit("--factorized-schema-confidence-threshold must be between zero and one")
        model.factorized_schema_confidence_threshold = args.factorized_schema_confidence_threshold
    if args.structured_plan_mode is not None:
        model.structured_plan_mode = args.structured_plan_mode
    if args.unconstrained_semantic_plan:
        model.constrain_semantic_plan = False
    print(generate_sql(model, args.schema, args.question, tokenizer, args.max_tokens))


if __name__ == "__main__":
    main()
