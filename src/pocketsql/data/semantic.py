"""Build a natural-language-focused semantic-plan pilot and frozen gate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3

from .generate import dataset_quality_report
from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import DOMAIN_VOCAB, Schema, make_schema
from .validate import validate_sql
from .verbalize import humanize_identifier


HUMAN_NAMES = ("Max", "John", "Alice", "Jordan", "Grace", "Sam", "Maya", "Leo", "Nora", "Omar", "Priya", "Diego")
SOURCE_WEIGHTS = {"focused": 18, "composition": 13, "replay": 7, "gretel": 1, "wikisql": 1}
JOINED_AGGREGATE_INTENTS = frozenset(
    {
        "relationship_join_sum_name_location",
        "relationship_join_average_location_status",
        "relationship_join_maximum_name",
        "relationship_join_count_name_location",
    }
)


def _labels(schema: Schema) -> dict[str, str]:
    parent, child = schema.tables
    return {
        "parent_one": humanize_identifier(parent.name, "singular"),
        "parent_many": humanize_identifier(parent.name, "plural"),
        "child_one": humanize_identifier(child.name, "singular"),
        "child_many": humanize_identifier(child.name, "plural"),
        "name": humanize_identifier(schema.role("name")[1].name),
        "location": humanize_identifier(schema.role("location")[1].name),
        "amount": humanize_identifier(schema.role("amount")[1].name),
        "status": humanize_identifier(schema.role("status")[1].name),
        "child_id": humanize_identifier(schema.role("child_id")[1].name),
    }


def _semantic_cases(
    schema: Schema,
    people: list[tuple[int, str, str]],
    rng: random.Random,
    heldout: bool,
    include_joined_aggregates: bool = False,
) -> list[tuple[str, QueryPlan, str]]:
    def _choose(
        _rng: random.Random,
        training: tuple[str, ...],
        development: str | tuple[str, ...],
    ) -> str:
        if not heldout:
            return rng.choice(training)
        return rng.choice(development) if isinstance(development, tuple) else development

    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_id = schema.role("child_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    labels = _labels(schema)
    person_id, person_name, person_city = people[0]
    second_id, second_name, second_city = people[1]
    del person_id, second_id
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    child_all = (f"{child.name}.*",)
    parent_all = (f"{parent.name}.*",)
    parent_name = f"{parent.name}.{name}"
    parent_location = f"{parent.name}.{location}"
    child_status = f"{child.name}.{status}"
    child_amount = f"{child.name}.{amount}"

    cases = [
        (
            "contrast_parent_projection",
            QueryPlan("semantic_parent_projection", parent.name, (name,)),
            _choose(
                rng,
                (
                    f"list the {labels['name']} for every {labels['parent_one']}",
                    f"give me all {labels['parent_many']} {labels['name']}",
                    f"what are the {labels['name']} of our {labels['parent_many']}",
                    f"show the {labels['name']} for each {labels['parent_one']}",
                    f"show me every {labels['parent_one']} {labels['name']}",
                    f"display {labels['parent_many']} by {labels['name']}",
                    f"get all {labels['name']} from {labels['parent_many']}",
                    f"I need the {labels['name']} for the {labels['parent_many']}",
                ),
                (
                    f"show me {labels['parent_one']} names",
                    f"show all {labels['parent_one']} names",
                ),
            ),
        ),
        (
            "contrast_named_parent_rows",
            QueryPlan(
                "semantic_named_parent_rows",
                parent.name,
                parent_all,
                filters=(Filter(name, "=", person_name),),
            ),
            _choose(
                rng,
                (
                    f"list every {labels['parent_one']} whose {labels['name']} is {person_name}",
                    f"find the {labels['parent_one']} with {labels['name']} equal to {person_name}",
                    f"return {labels['parent_many']} where {labels['name']} matches {person_name}",
                    f"show me {labels['parent_many']} where {labels['name']} equals {person_name}",
                    f"show the {labels['parent_one']} with {labels['name']} matching {person_name}",
                    f"get {labels['parent_many']} whose {labels['name']} is {person_name}",
                    f"display {labels['parent_many']} when {labels['name']} equals {person_name}",
                    f"I need the {labels['parent_one']} named {person_name}",
                ),
                (
                    f"show me {labels['parent_many']} with the {labels['name']} {person_name}",
                    f"show {labels['parent_many']} where {labels['name']} is {person_name}",
                ),
            ),
        ),
        (
            "relationship_join_name",
            QueryPlan(
                "semantic_join_name",
                child.name,
                child_all,
                filters=(Filter(parent_name, "=", person_name),),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"list {labels['child_many']} belonging to a {labels['parent_one']} whose {labels['name']} is {person_name}",
                    f"find {labels['child_many']} associated with the {labels['parent_one']} named {person_name}",
                    f"get {labels['child_many']} for {labels['parent_many']} where {labels['name']} equals {person_name}",
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name}",
                    f"show {labels['parent_one']} {labels['child_many']} with {labels['name']} matching {person_name}",
                    f"list {labels['parent_one']} {labels['child_many']} when {labels['name']} equals {person_name}",
                    f"get the {labels['child_many']} for {labels['parent_one']} {person_name}",
                    f"display {labels['child_many']} whose {labels['parent_one']} has {labels['name']} {person_name}",
                ),
                (
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} is {person_name}",
                    f"show {labels['child_many']} for the {labels['parent_one']} whose {labels['name']} is {person_name}",
                ),
            ),
        ),
        (
            "relationship_join_name_location",
            QueryPlan(
                "semantic_join_name_location",
                child.name,
                child_all,
                filters=(Filter(parent_name, "=", person_name), Filter(parent_location, "=", person_city)),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"list {labels['child_many']} for the {labels['parent_one']} whose {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                    f"find {labels['child_many']} tied to {labels['parent_many']} with {labels['name']} equal to {person_name} and {labels['location']} equal to {person_city}",
                    f"get {labels['child_many']} belonging to the {labels['parent_one']} named {person_name} in {person_city}",
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                    f"show {labels['child_many']} for {labels['parent_many']} with {labels['name']} matching {person_name} and {labels['location']} matching {person_city}",
                    f"list {labels['parent_one']} {labels['child_many']} when {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                    f"get {labels['child_many']} for {person_name} from {person_city}",
                    f"display {labels['child_many']} whose {labels['parent_one']} has {labels['name']} {person_name} and {labels['location']} {person_city}",
                ),
                (
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                    f"show {labels['child_many']} where the {labels['parent_one']} {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                ),
            ),
        ),
        (
            "relationship_join_location",
            QueryPlan(
                "semantic_join_location",
                child.name,
                child_all,
                filters=(Filter(parent_location, "=", second_city),),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"which {labels['child_many']} belong to {labels['parent_many']} whose {labels['location']} is {second_city}",
                    f"get {labels['child_many']} associated with {labels['parent_many']} located in {second_city}",
                    f"list {labels['child_many']} for {labels['parent_many']} where {labels['location']} equals {second_city}",
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['location']} equals {second_city}",
                    f"show {labels['child_many']} for {labels['parent_many']} with {labels['location']} matching {second_city}",
                    f"list {labels['parent_one']} {labels['child_many']} when {labels['location']} equals {second_city}",
                    f"get {labels['child_many']} for {labels['parent_many']} in {second_city}",
                    f"display {labels['child_many']} whose {labels['parent_one']} is located in {second_city}",
                ),
                (
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['location']} is {second_city}",
                    f"show {labels['child_many']} for {labels['parent_many']} whose {labels['location']} is {second_city}",
                ),
            ),
        ),
        (
            "relationship_join_parent_child_filters",
            QueryPlan(
                "semantic_join_parent_child_filters",
                child.name,
                child_all,
                filters=(Filter(parent_name, "=", second_name), Filter(child_status, "=", "shipped")),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"show {labels['child_many']} for the {labels['parent_one']} whose {labels['name']} is {second_name} when {labels['status']} is shipped",
                    f"find shipped {labels['child_many']} linked to {labels['parent_many']} where {labels['name']} equals {second_name}",
                    f"list {labels['child_many']} belonging to {labels['parent_many']} with {labels['name']} {second_name} and {labels['status']} shipped",
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {second_name} and {labels['status']} equals shipped",
                    f"show {labels['child_many']} for {labels['parent_many']} with {labels['name']} matching {second_name} and {labels['status']} matching shipped",
                    f"list {labels['parent_one']} {labels['child_many']} when {labels['name']} equals {second_name} and {labels['status']} equals shipped",
                    f"get shipped {labels['child_many']} for {labels['parent_one']} {second_name}",
                    f"display {labels['child_many']} whose {labels['parent_one']} has {labels['name']} {second_name} and whose {labels['status']} is shipped",
                ),
                (
                    f"show me {labels['parent_one']} {labels['child_many']} where {labels['name']} is {second_name} and {labels['status']} is shipped",
                    f"show {labels['child_many']} for the {labels['parent_one']} whose {labels['name']} is {second_name} when {labels['status']} equals shipped",
                ),
            ),
        ),
        (
            "relationship_join_count",
            QueryPlan(
                "semantic_join_count",
                child.name,
                (),
                aggregate="COUNT",
                filters=(Filter(parent_name, "=", person_name),),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"count the {labels['child_many']} belonging to the {labels['parent_one']} whose {labels['name']} is {person_name}",
                    f"how many {labels['child_many']} are linked to {labels['parent_many']} where {labels['name']} equals {person_name}",
                    f"give me the number of {labels['child_many']} for {labels['parent_one']} {person_name}",
                    f"count {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name}",
                    f"how many {labels['parent_one']} {labels['child_many']} have {labels['name']} matching {person_name}",
                    f"tell me the count of {labels['child_many']} when the {labels['parent_one']} {labels['name']} equals {person_name}",
                    f"get the number of {labels['child_many']} belonging to {person_name}",
                    f"display a count of {labels['child_many']} for the {labels['parent_one']} named {person_name}",
                ),
                (
                    f"how many {labels['parent_one']} {labels['child_many']} have {labels['name']} {person_name}",
                    f"how many {labels['child_many']} belong to the {labels['parent_one']} whose {labels['name']} is {person_name}",
                ),
            ),
        ),
        (
            "contrast_maximum_aggregate",
            QueryPlan(
                "semantic_maximum_aggregate",
                child.name,
                (),
                aggregate="MAX",
                aggregate_column=amount,
            ),
            _choose(
                rng,
                (
                    f"what is the maximum {labels['amount']} across {labels['child_many']}",
                    f"find the largest {labels['amount']} among all {labels['child_many']}",
                    f"show me the highest {labels['amount']} for {labels['child_many']}",
                    f"what is the biggest {labels['amount']} across the {labels['child_many']}",
                    f"get the maximum {labels['amount']} from {labels['child_many']}",
                    f"display the largest {labels['amount']} in {labels['child_many']}",
                    f"which {labels['amount']} is highest among {labels['child_many']}",
                    f"tell me the top {labels['amount']} for all {labels['child_many']}",
                ),
                (
                    f"what is the greatest {labels['amount']} across {labels['child_many']}",
                    f"show the greatest {labels['amount']} among the {labels['child_many']}",
                ),
            ),
        ),
        (
            "contrast_projection_with_name_filter",
            QueryPlan(
                "semantic_projection_name_filter",
                parent.name,
                (name, location),
                filters=(Filter(name, "=", person_name),),
            ),
            _choose(
                rng,
                (
                    f"show the {labels['name']} and {labels['location']} for {labels['parent_many']} where {labels['name']} is {person_name}",
                    f"give me {labels['name']} plus {labels['location']} from the {labels['parent_one']} named {person_name}",
                    f"return {labels['name']} and {labels['location']} when {labels['name']} equals {person_name}",
                    f"show me {labels['name']} and {labels['location']} where {labels['name']} matches {person_name}",
                    f"show {labels['name']} with {labels['location']} for {labels['parent_many']} whose {labels['name']} equals {person_name}",
                    f"list {labels['name']} plus {labels['location']} when {labels['name']} is {person_name}",
                    f"get the {labels['name']} and {labels['location']} for {person_name}",
                    f"display {labels['name']} and {labels['location']} filtered by {labels['name']} equal to {person_name}",
                ),
                (
                    f"show me {labels['name']} and {labels['location']} where {labels['name']} is {person_name}",
                    f"show the {labels['name']} and {labels['location']} for the {labels['parent_one']} whose {labels['name']} is {person_name}",
                ),
            ),
        ),
        (
            "composition_single_table_filters",
            QueryPlan(
                "semantic_single_table_filters",
                child.name,
                (child_id,),
                filters=(Filter(status, "=", "shipped"), Filter(amount, ">", 25)),
            ),
            _choose(
                rng,
                (
                    f"list {labels['child_id']} where {labels['status']} is shipped and {labels['amount']} is greater than 25",
                    f"show {labels['child_id']} for {labels['child_many']} with {labels['status']} shipped plus {labels['amount']} above 25",
                    f"find {labels['child_id']} when {labels['status']} equals shipped and {labels['amount']} exceeds 25",
                    f"show me {labels['child_id']} where {labels['status']} equals shipped and {labels['amount']} is above 25",
                    f"show {labels['child_id']} when {labels['status']} matches shipped and {labels['amount']} exceeds 25",
                    f"list {labels['child_id']} with {labels['status']} equal to shipped and {labels['amount']} greater than 25",
                    f"get {labels['child_id']} for shipped {labels['child_many']} over 25 {labels['amount']}",
                    f"display {labels['child_id']} filtered to {labels['status']} shipped and {labels['amount']} above 25",
                ),
                (
                    f"list {labels['child_id']} where {labels['status']} is shipped and {labels['amount']} is over 25",
                    f"show {labels['child_id']} when {labels['status']} is shipped and {labels['amount']} is greater than 25",
                ),
            ),
        ),
        (
            "contrast_parent_projection_pair",
            QueryPlan("semantic_parent_projection_pair", parent.name, (name, location)),
            _choose(
                rng,
                (
                    f"list {labels['name']} and {labels['location']} for all {labels['parent_many']}",
                    f"show every {labels['parent_one']} {labels['name']} with its {labels['location']}",
                    f"give me the {labels['name']} and {labels['location']} columns from {labels['parent_many']}",
                    f"show me {labels['parent_one']} {labels['name']} plus {labels['location']}",
                    f"show all {labels['parent_many']} by {labels['name']} and {labels['location']}",
                    f"list the {labels['name']} with the {labels['location']} for each {labels['parent_one']}",
                    f"get {labels['name']} and {labels['location']} from {labels['parent_many']}",
                    f"display every {labels['parent_one']} using {labels['name']} and {labels['location']}",
                ),
                (
                    f"show me {labels['parent_one']} {labels['name']} and {labels['location']}",
                    f"show all {labels['parent_many']} with {labels['name']} and {labels['location']}",
                ),
            ),
        ),
        (
            "relationship_join_project_child_id",
            QueryPlan(
                "semantic_join_project_child_id",
                child.name,
                (f"{child.name}.{child_id}",),
                filters=(Filter(parent_name, "=", second_name),),
                join_table=parent.name,
                join_on=join_on,
            ),
            _choose(
                rng,
                (
                    f"list the {labels['child_id']} belonging to the {labels['parent_one']} whose {labels['name']} is {second_name}",
                    f"find {labels['child_id']} for {labels['child_many']} linked to {labels['parent_many']} where {labels['name']} equals {second_name}",
                    f"return {labels['child_id']} from {labels['child_many']} associated with {labels['parent_one']} {second_name}",
                    f"show me {labels['child_id']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {second_name}",
                    f"show {labels['child_id']} from {labels['child_many']} when the {labels['parent_one']} {labels['name']} matches {second_name}",
                    f"list {labels['parent_one']} {labels['child_id']} with {labels['name']} equal to {second_name}",
                    f"get the {labels['child_id']} for {labels['parent_one']} {second_name}",
                    f"display {labels['child_id']} whose {labels['parent_one']} has {labels['name']} {second_name}",
                ),
                (
                    f"show me {labels['child_id']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} is {second_name}",
                    f"show the {labels['child_id']} belonging to the {labels['parent_one']} whose {labels['name']} is {second_name}",
                ),
            ),
        ),
    ]
    if include_joined_aggregates:
        cases.extend(
            [
                (
                    "relationship_join_sum_name_location",
                    QueryPlan(
                        "semantic_join_sum_name_location",
                        child.name,
                        (),
                        aggregate="SUM",
                        aggregate_column=child_amount,
                        filters=(Filter(parent_name, "=", person_name), Filter(parent_location, "=", person_city)),
                        join_table=parent.name,
                        join_on=join_on,
                    ),
                    _choose(
                        rng,
                        (
                            f"sum the {labels['amount']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                            f"add up {labels['amount']} from {labels['child_many']} belonging to the {labels['parent_one']} whose {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                            f"calculate the total {labels['amount']} across {labels['child_many']} for {labels['parent_one']} {person_name} in {person_city}",
                            f"sum {labels['amount']} where {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                            f"how much {labels['amount']} is there in total from {labels['child_many']} when {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                            f"give me the combined {labels['amount']} for {labels['parent_many']} with {labels['name']} matching {person_name} and {labels['location']} matching {person_city}",
                            f"total the {labels['amount']} on {labels['child_many']} whose {labels['parent_one']} has {labels['name']} {person_name} and {labels['location']} {person_city}",
                            f"calculate total {labels['amount']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                        ),
                        (
                            f"show me total {labels['amount']} where {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                            f"what is the total {labels['amount']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                        ),
                    ),
                ),
                (
                    "relationship_join_average_location_status",
                    QueryPlan(
                        "semantic_join_average_location_status",
                        child.name,
                        (),
                        aggregate="AVG",
                        aggregate_column=child_amount,
                        filters=(Filter(parent_location, "=", second_city), Filter(child_status, "=", "shipped")),
                        join_table=parent.name,
                        join_on=join_on,
                    ),
                    _choose(
                        rng,
                        (
                            f"average the {labels['amount']} for shipped {labels['child_many']} from {labels['parent_many']} where {labels['location']} equals {second_city}",
                            f"find the average {labels['amount']} of {labels['child_many']} with {labels['status']} shipped and {labels['parent_one']} {labels['location']} {second_city}",
                            f"what is the mean {labels['amount']} for {labels['child_many']} where {labels['location']} is {second_city} and {labels['status']} is shipped",
                            f"calculate average {labels['amount']} when {labels['location']} equals {second_city} and {labels['status']} equals shipped",
                            f"show the average {labels['amount']} across shipped {labels['child_many']} belonging to {labels['parent_many']} in {second_city}",
                            f"get mean {labels['amount']} for {labels['parent_one']} {labels['child_many']} with {labels['location']} matching {second_city} and {labels['status']} matching shipped",
                            f"average {labels['amount']} from {labels['child_many']} whose {labels['parent_one']} is in {second_city} and whose {labels['status']} is shipped",
                            f"give me average {labels['amount']} where {labels['location']} equals {second_city} and {labels['status']} equals shipped",
                        ),
                        (
                            f"show me the average {labels['amount']} where {labels['location']} is {second_city} and {labels['status']} is shipped",
                            f"what is the average {labels['amount']} for shipped {labels['parent_one']} {labels['child_many']} in {second_city}",
                        ),
                    ),
                ),
                (
                    "relationship_join_maximum_name",
                    QueryPlan(
                        "semantic_join_maximum_name",
                        child.name,
                        (),
                        aggregate="MAX",
                        aggregate_column=child_amount,
                        filters=(Filter(parent_name, "=", second_name),),
                        join_table=parent.name,
                        join_on=join_on,
                    ),
                    _choose(
                        rng,
                        (
                            f"find the maximum {labels['amount']} among {labels['child_many']} for the {labels['parent_one']} whose {labels['name']} is {second_name}",
                            f"show the highest {labels['amount']} on {labels['child_many']} belonging to {labels['parent_one']} {second_name}",
                            f"what is the largest {labels['amount']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {second_name}",
                            f"get max {labels['amount']} where {labels['name']} equals {second_name}",
                            f"calculate the greatest {labels['amount']} across {labels['child_many']} linked to {labels['parent_many']} with {labels['name']} {second_name}",
                            f"give me the top {labels['amount']} for {labels['child_many']} whose {labels['parent_one']} is named {second_name}",
                            f"maximum {labels['amount']} from {labels['child_many']} when {labels['name']} matches {second_name}",
                            f"find highest {labels['amount']} for {labels['parent_one']} {second_name} {labels['child_many']}",
                        ),
                        (
                            f"show me the greatest {labels['amount']} where {labels['name']} is {second_name}",
                            f"what is the maximum {labels['amount']} for {labels['parent_one']} {labels['child_many']} where {labels['name']} is {second_name}",
                        ),
                    ),
                ),
                (
                    "relationship_join_count_name_location",
                    QueryPlan(
                        "semantic_join_count_name_location",
                        child.name,
                        (),
                        aggregate="COUNT",
                        filters=(Filter(parent_name, "=", person_name), Filter(parent_location, "=", person_city)),
                        join_table=parent.name,
                        join_on=join_on,
                    ),
                    _choose(
                        rng,
                        (
                            f"count {labels['parent_one']} {labels['child_many']} where {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                            f"how many {labels['child_many']} belong to the {labels['parent_one']} whose {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                            f"give me the number of {labels['child_many']} for {labels['parent_one']} {person_name} in {person_city}",
                            f"count {labels['child_many']} where {labels['name']} matches {person_name} and {labels['location']} matches {person_city}",
                            f"tell me how many {labels['child_many']} are linked to {labels['parent_many']} with {labels['name']} {person_name} and {labels['location']} {person_city}",
                            f"get the count of {labels['child_many']} when {labels['name']} equals {person_name} and {labels['location']} equals {person_city}",
                            f"how many {labels['parent_one']} {labels['child_many']} have {labels['name']} {person_name} and {labels['location']} {person_city}",
                            f"display a count of {labels['child_many']} for the {labels['parent_one']} named {person_name} in {person_city}",
                        ),
                        (
                            f"show me the number of {labels['child_many']} where {labels['name']} is {person_name} and {labels['location']} is {person_city}",
                            f"how many {labels['parent_one']} {labels['child_many']} are for {person_name} in {person_city}",
                        ),
                    ),
                ),
            ]
        )
    return cases


def _schema_records(
    schema: Schema,
    schema_index: int,
    rng: random.Random,
    heldout: bool,
    seed: int,
    include_joined_aggregates: bool = False,
) -> list[dict]:
    connection = sqlite3.connect(":memory:")
    populate(connection, schema, rng)
    parent = schema.tables[0]
    parent_id = schema.role("parent_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    status = schema.role("status")[1].name
    names = list(HUMAN_NAMES)
    rng.shuffle(names)
    identifiers = [row[0] for row in connection.execute(f"SELECT {parent_id} FROM {parent.name} ORDER BY {parent_id}")]
    connection.executemany(
        f"UPDATE {parent.name} SET {name} = ? WHERE {parent_id} = ?",
        list(zip(names, identifiers)),
    )
    connection.commit()
    people = [
        (identifier, person_name, city)
        for identifier, person_name, city in connection.execute(
            f"SELECT {parent_id}, {name}, {location} FROM {parent.name} ORDER BY {parent_id}"
        )
    ]
    shipped_person = connection.execute(
        f"SELECT p.{parent_id}, p.{name}, p.{location} "
        f"FROM {parent.name} AS p INNER JOIN {schema.tables[1].name} AS c "
        f"ON c.{child_fk} = p.{parent_id} WHERE c.{status} = 'shipped' LIMIT 1"
    ).fetchone()
    if shipped_person is None:
        raise RuntimeError(f"No shipped relationship available for {schema.schema_id}")
    people[1] = shipped_person
    database_sql = "\n".join(connection.iterdump())
    records = []
    for case_index, (intent, plan, question) in enumerate(
        _semantic_cases(schema, people, rng, heldout, include_joined_aggregates)
    ):
        sql = render_sql(plan)
        valid, rows = validate_sql(connection, sql)
        if not valid or not rows:
            raise RuntimeError(f"Invalid semantic example {intent} for {schema.schema_id}: {sql}")
        records.append(
            {
                "id": f"{schema.schema_id}_{case_index:02d}",
                "schema_id": schema.schema_id,
                "schema_sql": schema.sql(),
                "database_sql": database_sql,
                "question": question,
                "sql": sql,
                "query_plan": plan.normalized(),
                "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None) + int(len(plan.filters) > 1),
                "seed": seed,
                "semantic_intent": intent,
                "evaluation_only": heldout,
            }
        )
    connection.close()
    return records


def build_focused_splits(
    schemas: int,
    seed: int,
    heldout: bool = False,
    schema_prefix: str = "semantic",
    include_joined_aggregates: bool = False,
) -> dict[str, list[dict]]:
    if schemas < 3 and not heldout:
        raise ValueError("at least three schemas are needed for train/validation/test splits")
    rng = random.Random(seed)
    by_schema = []
    for schema_index in range(schemas):
        schema = make_schema(schema_index, rng, DOMAIN_VOCAB, schema_prefix)
        by_schema.append(_schema_records(schema, schema_index, rng, heldout, seed, include_joined_aggregates))
    if heldout:
        return {"natural_gate": [record for records in by_schema for record in records]}
    rng.shuffle(by_schema)
    train_end = max(1, int(schemas * 0.8))
    validation_end = max(train_end + 1, int(schemas * 0.9))
    train_end = min(train_end, schemas - 2)
    validation_end = min(max(validation_end, train_end + 1), schemas - 1)
    return {
        "train": [record for records in by_schema[:train_end] for record in records],
        "validation": [record for records in by_schema[train_end:validation_end] for record in records],
        "test": [record for records in by_schema[validation_end:] for record in records],
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _quotas(total: int) -> dict[str, int]:
    weight_total = sum(SOURCE_WEIGHTS.values())
    raw = {name: total * weight / weight_total for name, weight in SOURCE_WEIGHTS.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(quotas.values())
    for name in sorted(raw, key=lambda item: raw[item] - quotas[item], reverse=True)[:remaining]:
        quotas[name] += 1
    return quotas


def mix_sources(sources: dict[str, list[dict]], total: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    selected = []
    quotas = _quotas(total)
    for source, count in quotas.items():
        candidates = list(sources[source])
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"source {source!r} has {len(candidates)} records but needs {count}")
        for record in candidates[:count]:
            selected.append(
                {
                    **record,
                    "id": f"{source}:{record.get('id', len(selected))}",
                    "schema_id": f"{source}:{record.get('schema_id', 'unknown')}",
                    "semantic_source": source,
                }
            )
    rng.shuffle(selected)
    return selected, quotas


def write_semantic_pilot(
    output: Path,
    schemas: int,
    gate_schemas: int,
    seed: int,
    composition: Path,
    replay: Path,
    gretel: Path,
    wikisql: Path,
    mixed_train_records: int,
    mixed_validation_records: int,
    include_joined_aggregates: bool = False,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    focused = build_focused_splits(
        schemas,
        seed,
        schema_prefix="semantic_train",
        include_joined_aggregates=include_joined_aggregates,
    )
    gate = build_focused_splits(
        gate_schemas,
        seed + 1,
        heldout=True,
        schema_prefix="semantic_gate",
        include_joined_aggregates=include_joined_aggregates,
    )["natural_gate"]
    for name, records in (*focused.items(), ("natural_gate", gate)):
        _write_jsonl(output / f"{name}.jsonl", records)
    joined_aggregate_counts: dict[str, int] = {}
    if include_joined_aggregates:
        for name, records in (*focused.items(), ("natural_gate", gate)):
            selected = [record for record in records if record["semantic_intent"] in JOINED_AGGREGATE_INTENTS]
            artifact_name = f"joined_aggregate_{'gate' if name == 'natural_gate' else name}"
            _write_jsonl(output / f"{artifact_name}.jsonl", selected)
            joined_aggregate_counts[artifact_name] = len(selected)

    train_sources = {
        "focused": focused["train"],
        "composition": _load_jsonl(composition / "train.jsonl"),
        "replay": _load_jsonl(replay / "train.jsonl"),
        "gretel": _load_jsonl(gretel / "train.jsonl"),
        "wikisql": _load_jsonl(wikisql / "train.jsonl"),
    }
    validation_sources = {
        "focused": focused["validation"],
        "composition": _load_jsonl(composition / "validation.jsonl"),
        "replay": _load_jsonl(replay / "validation.jsonl"),
        "gretel": _load_jsonl(gretel / "validation.jsonl"),
        "wikisql": _load_jsonl(wikisql / "validation.jsonl"),
    }
    mixed_train, train_quotas = mix_sources(train_sources, mixed_train_records, seed + 2)
    mixed_validation, validation_quotas = mix_sources(validation_sources, mixed_validation_records, seed + 3)
    _write_jsonl(output / "mixed_train.jsonl", mixed_train)
    _write_jsonl(output / "mixed_validation.jsonl", mixed_validation)

    report = dataset_quality_report({**focused, "natural_gate": gate})
    report.update(
        {
            "target_format": "semantic_plan",
            "include_joined_aggregates": include_joined_aggregates,
            "joined_aggregate_counts": joined_aggregate_counts,
            "natural_gate_evaluation_only": True,
            "intent_counts": dict(sorted(Counter(record["semantic_intent"] for record in gate).items())),
            "mix": {
                "weights": SOURCE_WEIGHTS,
                "train_records": len(mixed_train),
                "train_source_counts": train_quotas,
                "validation_records": len(mixed_validation),
                "validation_source_counts": validation_quotas,
            },
            "sources": {
                "composition": str(composition),
                "replay": str(replay),
                "gretel": str(gretel),
                "wikisql": str(wikisql),
            },
        }
    )
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **{name: len(records) for name, records in focused.items()},
        "natural_gate": len(gate),
        "mixed_train": len(mixed_train),
        "mixed_validation": len(mixed_validation),
        **joined_aggregate_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=1000)
    parser.add_argument("--gate-schemas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--composition", type=Path, default=Path("data/generated-composition-v9-pilot"))
    parser.add_argument("--replay", type=Path, default=Path("data/generated-position-robust-v8-replay"))
    parser.add_argument("--gretel", type=Path, default=Path("data/gretel-pilot-v2"))
    parser.add_argument("--wikisql", type=Path, default=Path("data/wikisql-pilot-v1"))
    parser.add_argument("--mixed-train-records", type=int, default=20000)
    parser.add_argument("--mixed-validation-records", type=int, default=2500)
    parser.add_argument(
        "--include-joined-aggregates",
        action="store_true",
        help="Add SUM/AVG/MAX/COUNT joins with parent and cross-table filters.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_semantic_pilot(
                args.output,
                args.schemas,
                args.gate_schemas,
                args.seed,
                args.composition,
                args.replay,
                args.gretel,
                args.wikisql,
                args.mixed_train_records,
                args.mixed_validation_records,
                args.include_joined_aggregates,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
