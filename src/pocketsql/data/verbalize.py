from __future__ import annotations

import random
import re

from .query_ast import QueryPlan


VERBS = ("show", "list", "find", "which")

OPERATOR_WORDS = {
    "=": ("is", "equals", "matches"),
    ">": ("is greater than", "is above", "exceeds"),
    "<": ("is less than", "is below"),
    ">=": ("is at least", "is no less than"),
    "<=": ("is at most", "is no more than"),
}


def _words(identifier: str) -> list[str]:
    identifier = identifier.rsplit(".", 1)[-1]
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier).casefold()
    words = [word for word in snake_case.split("_") if word]
    if words and words[0] in {"tbl", "col"}:
        words = words[1:]
    if len(words) > 1 and words[-1] in {"data", "value"}:
        words = words[:-1]
    return words or [identifier.casefold()]


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _plural(word: str) -> str:
    if word.endswith("s"):
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"


def _label(identifier: str, number: str | None = None) -> str:
    words = _words(identifier)
    if number == "singular":
        words[-1] = _singular(words[-1])
    elif number == "plural":
        words[-1] = _plural(words[-1])
    return " ".join(words)


def humanize_identifier(identifier: str, number: str | None = None) -> str:
    """Convert generated snake/camel/prefixed identifiers into question text."""
    return _label(identifier, number)


def _casual_predicates(plan: QueryPlan, rng: random.Random, heldout: bool) -> str:
    phrases = []
    for item in plan.filters:
        column = _label(item.column)
        if item.operator == "=":
            wording = "is" if heldout else rng.choice(("is", "equals", "set to", "marked as"))
        elif item.operator == ">":
            wording = "above" if heldout else rng.choice(("over", "above", "more than", "greater than"))
        elif item.operator == "<":
            wording = "below" if heldout else rng.choice(("under", "below", "less than"))
        elif item.operator == ">=":
            wording = "no lower than" if heldout else rng.choice(("at least", "no less than", "not below"))
        else:
            wording = "no higher than" if heldout else rng.choice(("at most", "no more than", "not above"))
        phrase = f"{column} {wording} {item.value}"
        phrases.append(phrase)
    connector = " or " if plan.filter_connector == "OR" else (" while " if heldout else rng.choice((" and ", " while ", " plus ")))
    return connector.join(phrases)


def casual_verbalize(plan: QueryPlan, rng: random.Random, heldout: bool = False) -> str:
    """Render conversational language independently of the original templates."""
    table_one = _label(plan.table, "singular")
    table_many = _label(plan.table, "plural")
    columns = [_label(column, "plural") for column in plan.columns]
    column_words = " and ".join(columns)
    predicates = _casual_predicates(plan, rng, heldout)

    def choose(training: tuple[str, ...], development: str) -> str:
        return development if heldout else rng.choice(training)

    if plan.family == "count_text_filter":
        value = str(plan.filters[0].value)
        words = choose(
            (
                f"show me the number of {table_many} from {value}",
                f"how many {table_many} are from {value}",
                f"count the {table_many} in {value}",
                f"give me the number of {table_many} located in {value}",
                f"what is the {table_one} count for {value}",
            ),
            f"how many {table_many} do we have around {value}",
        )
        predicates = ""
    elif plan.family == "join_text_filter":
        related_one = _label(plan.join_table or "record", "singular")
        related_many = _label(plan.join_table or "records", "plural")
        value = str(plan.filters[0].value)
        words = choose(
            (
                f"show me {table_many} from {related_one} from {value}",
                f"list {table_many} belonging to {related_many} in {value}",
                f"which {table_many} are connected to {related_many} located in {value}",
                f"get {table_many} for {related_many} based in {value}",
                f"show every {table_one} associated with one {related_one} from {value}",
            ),
            f"pull up {table_many} tied to {related_many} around {value}",
        )
        predicates = ""
    elif plan.family == "join_count_text_filter":
        related_many = _label(plan.join_table or "records", "plural")
        value = str(plan.filters[0].value)
        words = choose(
            (
                f"show me the number of {table_many} from {related_many} in {value}",
                f"how many {table_many} belong to {related_many} from {value}",
                f"count {table_many} connected to {related_many} in {value}",
                f"give me the {table_one} count for {related_many} located in {value}",
            ),
            f"how many {table_many} are tied to {related_many} around {value}",
        )
        predicates = ""
    elif plan.family == "join_aggregate_text_filter":
        related_many = _label(plan.join_table or "records", "plural")
        value = str(plan.filters[0].value)
        target = _label(plan.aggregate_column or "value")
        operation = {
            "SUM": f"sum of the {target}",
            "AVG": f"average {target}",
            "MIN": f"lowest {target}",
            "MAX": f"highest {target}",
        }[plan.aggregate or "SUM"]
        words = choose(
            (
                f"show me the {operation} for {table_many} from {related_many} in {value}",
                f"what is the {operation} among {table_many} linked to {related_many} from {value}",
                f"find the {operation} for {table_many} belonging to {related_many} located in {value}",
            ),
            f"give me the {operation} for {table_many} tied to {related_many} around {value}",
        )
        predicates = ""
    elif plan.family == "text_filter":
        location_filter = plan.filters[0]
        location = _label(location_filter.column)
        value = str(location_filter.value)
        if location_filter.column in plan.columns:
            words = choose(
                (
                    f"what are the {table_one} {column_words} from {value}",
                    f"show me {column_words} for {table_many} in {value}",
                    f"list {column_words} for {table_many} located in {value}",
                    f"give me {column_words} from {table_many} based in {value}",
                ),
                f"pull up {column_words} for {table_many} around {value}",
            )
        else:
            words = choose(
                (
                    f"show me {column_words} for {table_many} where {location} is {value}",
                    f"list {column_words} from {table_many} with {location} equal to {value}",
                    f"give me {column_words} for {table_many} located in {value} by {location}",
                ),
                f"pull up {column_words} for {table_many} whose {location} is {value}",
            )
        predicates = ""
    elif plan.group_by:
        group = " and ".join(_label(column) for column in plan.group_by)
        words = choose(
            (
                f"give me a {table_one} count for every {group}",
                f"count {table_many} separately for each {group}",
                f"break the number of {table_many} down by {group}",
                f"for every {group}, tell me how many {table_many} there are",
                f"group {table_many} by {group} and count them",
                f"give me the number of {table_many} in every {group} category",
                f"return a separate {table_one} count for each {group}",
                f"how many {table_many} are there in each {group} category",
            ),
            f"how many {table_many} fall into each {group}",
        )
    elif plan.aggregate == "COUNT":
        words = choose(
            (
                f"how many {table_many} do we have",
                f"count all the {table_many}",
                f"tell me the number of {table_many}",
                f"what is our total {table_one} count",
                f"give me the headcount for {table_many}",
                f"tell me the total number of {table_many}",
                f"give me one overall count of {table_many}",
                f"how many {table_many} are there altogether",
            ),
            f"give me a headcount of all {table_many}",
        )
    elif plan.aggregate:
        target = _label(plan.aggregate_column or "value")
        train = {
            "SUM": (
                f"add up {target} across all {table_many}",
                f"give me the total {target} for {table_many}",
                f"how much {target} do all {table_many} have combined",
                f"sum the {target} from every {table_one}",
            ),
            "AVG": (
                f"what is the average {target} for a {table_one}",
                f"find the mean {target} across {table_many}",
                f"what {target} does a typical {table_one} have",
                f"average the {target} for all {table_many}",
            ),
            "MIN": (
                f"show me the lowest {target} among {table_many}",
                f"find the minimum {target} for {table_many}",
                f"what is the smallest {target} in {table_many}",
            ),
            "MAX": (
                f"show me the highest {target} among {table_many}",
                f"find the maximum {target} for {table_many}",
                f"what is the largest {target} in {table_many}",
            ),
        }
        dev = {
            "SUM": f"how much {target} is there in total for {table_many}",
            "AVG": f"roughly what {target} does a typical {table_one} have",
            "MIN": f"which is the smallest {target} recorded for {table_many}",
            "MAX": f"which is the largest {target} recorded for {table_many}",
        }
        words = dev[plan.aggregate] if heldout else rng.choice(train[plan.aggregate])
    elif plan.join_table:
        other_many = _label(plan.join_table, "plural")
        words = choose(
            (
                f"show me {column_words} for every {table_one} with matching {other_many}",
                f"combine each {table_one} with its {other_many} and list {column_words}",
                f"connect {table_many} to {other_many} so I can see {column_words}",
                f"for every {table_one}, get {column_words} from the related {other_many}",
                f"join {table_many} with {other_many} and return {column_words}",
                f"I need {column_words} across matching {table_many} and {other_many}",
                f"bring back {column_words} by matching each {table_one} with related {other_many}",
                f"use the link between {table_many} and {other_many} to show {column_words}",
                f"follow each {table_one}'s relationship to {other_many} and return {column_words}",
            ),
            f"pull up {column_words} by connecting each {table_one} to its {other_many}",
        )
    elif plan.distinct:
        words = choose(
            (
                f"what different {column_words} are represented among {table_many}",
                f"list the unique {column_words} from {table_many}",
                f"show each distinct {column_words} used by {table_many}",
                f"remove duplicate {column_words} when listing {table_many}",
                f"what {column_words} appear in {table_many} without repeats",
                f"tell me which {column_words} have appeared anywhere among {table_many}",
                f"which {column_words} occur among {table_many}, without duplicates",
                f"return every different {column_words} seen in {table_many}",
            ),
            f"which {column_words} occur at least once for {table_many}",
        )
    elif len(columns) == 1:
        words = choose(
            (
                f"show me {table_one} {columns[0]}",
                f"give me the {columns[0]} for every {table_one}",
                f"list all {columns[0]} from {table_many}",
                f"can you pull up each {table_one}'s {columns[0]}",
                f"I'd like the {columns[0]} for our {table_many}",
                f"what are the {table_one} {columns[0]}?",
            ),
            f"I'd like to see the {columns[0]} belonging to our {table_many}",
        )
    else:
        words = choose(
            (
                f"can I see each {table_one}'s {column_words}",
                f"show me the {column_words} for all {table_many}",
                f"list every {table_one} with its {column_words}",
                f"give me {column_words} from {table_many}",
                f"pull the {column_words} associated with each {table_one}",
                f"what are our {table_one} {column_words}",
                f"I want the {column_words} attached to every {table_one}",
            ),
            f"pull up the {column_words} belonging to every {table_one}",
        )

    if predicates:
        words += (" as long as " if heldout else rng.choice((" where ", " when ", " but only when ", " that have "))) + predicates
    if plan.order_by:
        direction = "highest first" if plan.descending else "lowest first"
        order = _label(plan.order_by)
        words += (
            f", sorted by {order} with {direction}"
            if heldout
            else rng.choice((f", order those by {order} with {direction}", f", sort on {order}, {direction}", f", arranged by {order}, {direction}"))
        )
    if plan.limit:
        words += (
            f", and only give me {plan.limit}"
            if heldout
            else rng.choice((f", just the first {plan.limit}", f", limit that to {plan.limit}", f", only return {plan.limit}"))
        )
    return words


def compositional_verbalize(plan: QueryPlan, rng: random.Random, heldout: bool = False) -> str:
    """Describe independently combined operations without relying on ``family``."""
    table_one = _label(plan.table, "singular")
    table_many = _label(plan.table, "plural")
    joined_many = _label(plan.join_table, "plural") if plan.join_table else ""
    wildcard = len(plan.columns) == 1 and plan.columns[0].endswith(".*")
    column_words = " and ".join(_label(column, "plural") for column in plan.columns)

    if plan.group_by:
        groups = " and ".join(_label(column) for column in plan.group_by)
        words = (
            f"give me a tally of {table_many} for each {groups}"
            if heldout
            else rng.choice(
                (
                    f"count {table_many} separately for every {groups}",
                    f"for each {groups}, show the number of {table_many}",
                    f"break down the {table_one} count by {groups}",
                )
            )
        )
    elif plan.aggregate == "COUNT":
        words = (
            f"give me a tally of {table_many}"
            if heldout
            else rng.choice(
                (
                    f"how many {table_many} are there",
                    f"show me the number of {table_many}",
                    f"count all {table_many}",
                    f"what is the {table_one} count",
                )
            )
        )
    elif plan.aggregate:
        target = _label(plan.aggregate_column or "value")
        training = {
            "SUM": (f"add up {target} for {table_many}", f"show the total {target} across {table_many}"),
            "AVG": (f"find the average {target} for {table_many}", f"show the mean {target} across {table_many}"),
            "MIN": (f"find the smallest {target} among {table_many}", f"show the minimum {target} for {table_many}"),
            "MAX": (f"find the largest {target} among {table_many}", f"show the maximum {target} for {table_many}"),
        }
        development = {
            "SUM": f"how much {target} do {table_many} have combined",
            "AVG": f"what {target} does a typical {table_one} have",
            "MIN": f"what is the lowest recorded {target} for {table_many}",
            "MAX": f"what is the highest recorded {target} for {table_many}",
        }
        words = development[plan.aggregate] if heldout else rng.choice(training[plan.aggregate])
    elif plan.distinct:
        words = (
            f"which {column_words} occur at least once among {table_many}"
            if heldout
            else rng.choice(
                (
                    f"show the unique {column_words} from {table_many}",
                    f"list distinct {column_words} for {table_many}",
                    f"what different {column_words} appear among {table_many}",
                )
            )
        )
    elif wildcard:
        words = (
            f"pull up every {table_one} record"
            if heldout
            else rng.choice((f"show me all {table_many}", f"list every {table_one}", f"return all {table_many}"))
        )
    else:
        words = (
            f"pull up {column_words} belonging to {table_many}"
            if heldout
            else rng.choice(
                (
                    f"show me {column_words} from {table_many}",
                    f"list {column_words} for every {table_one}",
                    f"give me {column_words} across {table_many}",
                    f"what {column_words} belong to {table_many}",
                )
            )
        )

    if plan.join_table:
        relation = (
            f" tied to matching {joined_many}"
            if heldout
            else rng.choice(
                (
                    f" linked to their {joined_many}",
                    f" joined with matching {joined_many}",
                    f" connected to the related {joined_many}",
                )
            )
        )
        words += relation

    if plan.filters:
        predicates = []
        for item in plan.filters:
            column = _label(item.column)
            if heldout:
                wording = {
                    "=": "matches",
                    ">": "is above",
                    "<": "is below",
                    ">=": "is no less than",
                    "<=": "is no more than",
                }[item.operator]
            else:
                wording = rng.choice(OPERATOR_WORDS[item.operator])
            predicates.append(f"{column} {wording} {item.value}")
        connector = " or " if plan.filter_connector == "OR" else (" while " if heldout else rng.choice((" and ", " while ", " plus ")))
        clause = connector.join(predicates)
        words += (" provided that " if heldout else rng.choice((" where ", " when ", " but only when "))) + clause

    if plan.order_by:
        order = _label(plan.order_by)
        direction = "highest first" if plan.descending else "lowest first"
        words += (
            f", arranged on {order} with {direction}"
            if heldout
            else rng.choice((f", sort by {order} with {direction}", f", order those on {order}, {direction}"))
        )
    if plan.limit is not None:
        words += (
            f", and stop after {plan.limit}"
            if heldout
            else rng.choice((f", only return {plan.limit}", f", limit that to {plan.limit}", f", just the first {plan.limit}"))
        )
    return words


def predicate_words(plan: QueryPlan, rng: random.Random) -> str:
    phrases = [f"{item.column} {rng.choice(OPERATOR_WORDS[item.operator])} {item.value}" for item in plan.filters]
    return f" {rng.choice((plan.filter_connector.lower(), 'as well as' if plan.filter_connector == 'AND' else 'or else'))} ".join(phrases)


def with_filters(words: str, plan: QueryPlan, rng: random.Random) -> str:
    if not plan.filters:
        return words
    return words + rng.choice((" where ", " with ", " for rows where ")) + predicate_words(plan, rng)


def verbalize(plan: QueryPlan, rng: random.Random, style: str = "classic") -> str:
    if style not in {"classic", "casual", "mixed", "heldout", "compositional", "compositional_heldout"}:
        raise ValueError("question style must be classic, casual, mixed, heldout, compositional, or compositional_heldout")
    if style == "compositional":
        return compositional_verbalize(plan, rng)
    if style == "compositional_heldout":
        return compositional_verbalize(plan, rng, heldout=True)
    if style == "heldout":
        return casual_verbalize(plan, rng, heldout=True)
    if style == "casual" or (style == "mixed" and rng.random() < 0.8):
        return casual_verbalize(plan, rng)
    verb = rng.choice(VERBS)
    if plan.family == "count_text_filter":
        value = plan.filters[0].value
        table = _label(plan.table, "plural")
        return rng.choice(
            (
                f"show me the number of {table} from {value}",
                f"count {table} in {value}",
                f"how many {table} are located in {value}",
            )
        )
    if plan.family in {"join_text_filter", "join_count_text_filter", "join_aggregate_text_filter"}:
        table = _label(plan.table, "plural")
        related = _label(plan.join_table or "records", "plural")
        value = plan.filters[0].value
        if plan.family == "join_text_filter":
            return rng.choice(
                (
                    f"show me {table} from {related} in {value}",
                    f"list {table} belonging to {related} located in {value}",
                    f"return {table} linked to {related} from {value}",
                )
            )
        if plan.family == "join_count_text_filter":
            return rng.choice(
                (
                    f"show me the number of {table} from {related} in {value}",
                    f"count {table} belonging to {related} located in {value}",
                    f"how many {table} are linked to {related} from {value}",
                )
            )
        target = _label(plan.aggregate_column or "value")
        calculation = {
            "SUM": f"sum of {target}",
            "AVG": f"average {target}",
            "MIN": f"smallest {target}",
            "MAX": f"largest {target}",
        }[plan.aggregate or "SUM"]
        return rng.choice(
            (
                f"show the {calculation} for {table} from {related} in {value}",
                f"what is the {calculation} among {table} linked to {related} from {value}",
            )
        )
    if plan.family == "text_filter":
        location_filter = plan.filters[0]
        columns = " and ".join(plan.columns)
        if location_filter.column in plan.columns:
            return rng.choice(
                (
                    f"what are the {plan.table} {columns} from {location_filter.value}",
                    f"{verb} {columns} from {plan.table} in {location_filter.value}",
                    f"return {columns} for {plan.table} located in {location_filter.value}",
                )
            )
        return rng.choice(
            (
                f"{verb} {columns} from {plan.table} where {location_filter.column} is {location_filter.value}",
                f"return {columns} for {plan.table} with {location_filter.column} equals {location_filter.value}",
            )
        )
    if plan.group_by:
        words = rng.choice(
            (
                f"{verb} the count of {plan.table} grouped by " + " and ".join(plan.group_by),
                f"break down {plan.table} counts by " + " and ".join(plan.group_by),
                f"for each " + " and ".join(plan.group_by) + f", return the number of {plan.table}",
            )
        )
        return with_filters(words, plan, rng)
    if plan.aggregate == "COUNT":
        words = rng.choice((f"how many {plan.table} are there", f"count the {plan.table}", f"what is the number of {plan.table}"))
        return with_filters(words, plan, rng)
    if plan.aggregate:
        adjective = {"SUM": "total", "AVG": "average", "MIN": "smallest", "MAX": "largest"}[plan.aggregate]
        target = plan.aggregate_column or "value"
        words = rng.choice((f"{verb} the {adjective} {target} for {plan.table}", f"what is the {adjective} {target} in {plan.table}", f"calculate {plan.aggregate.lower()} of {target} for {plan.table}"))
        return with_filters(words, plan, rng)
    columns = " and ".join(plan.columns)
    words = rng.choice((f"{verb} {columns} from {plan.table}", f"return {columns} for all {plan.table}", f"I need {columns} in {plan.table}"))
    if plan.distinct:
        words = rng.choice((f"{verb} distinct {columns} from {plan.table}", f"return unique {columns} in {plan.table}", f"which different {columns} appear in {plan.table}"))
    words = with_filters(words, plan, rng)
    if plan.join_table:
        words += rng.choice((f" with matching {plan.join_table}", f" joined to {plan.join_table}", f" by linking {plan.table} and {plan.join_table}"))
    if plan.limit:
        words += rng.choice((f", limited to {plan.limit}", f", returning only {plan.limit}", f", with a limit of {plan.limit}"))
    return words
