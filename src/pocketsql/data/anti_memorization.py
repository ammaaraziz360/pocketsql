"""Build a paired evaluation gate for separating copying from generalization.

Each pair keeps the schema, database, and gold query fixed.  The ``direct``
question names the relevant schema identifiers, while the ``paraphrase``
question expresses the same intent with domain synonyms.  Half of the intents
also use operation combinations withheld by the v9 composition curriculum.

This file is evaluation-only.  Its output must never be used for training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sqlite3

from .challenge import _load_reference_identifiers
from .composition import held_out_composition
from .generate import schema_identifiers
from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import CITY_VALUES, STATUS_VALUES, Column, Schema, Table
from .validate import validate_sql
from .verbalize import humanize_identifier


_VARIANTS = 6

# The identifier vocabulary and the question vocabulary are intentionally
# different.  For example, the schema says ``transport_charge`` while the
# paraphrase asks for a ``shipping price``.  Humans can connect the concepts,
# but exact schema-token matching cannot solve the paraphrased question alone.
_DOMAINS = (
    {
        "domain": "freight",
        "parent_table": ("haulers", "forwarders", "fleet_houses", "cargo_firms", "route_owners", "carrier_groups"),
        "child_table": ("waybills", "consignments", "load_sheets", "dispatch_batches", "cargo_entries", "haul_notices"),
        "parent_id": ("hauler_ref", "forwarder_ref", "fleet_ref", "cargo_firm_ref", "route_owner_ref", "carrier_group_ref"),
        "child_id": ("waybill_ref", "consignment_ref", "load_sheet_ref", "dispatch_batch_ref", "cargo_entry_ref", "haul_notice_ref"),
        "name": ("registered_title", "legal_caption", "trading_label", "official_heading", "registry_caption", "contract_title"),
        "location": ("home_territory", "operating_sector", "base_district", "origin_market", "service_zone", "coverage_sector"),
        "amount": ("transport_charge", "haulage_value", "carriage_price", "routing_fee", "load_expense", "movement_cost"),
        "status": ("journey_phase", "fulfillment_stage", "movement_state", "routing_phase", "haul_condition", "transit_stage"),
        "aliases": {
            "parent_one": "transport business",
            "parent_many": "transport businesses",
            "child_one": "delivery record",
            "child_many": "delivery records",
            "name": "business name",
            "location": "base area",
            "amount": "shipping price",
            "status": "delivery progress",
        },
    },
    {
        "domain": "energy",
        "parent_table": ("grid_firms", "utilities", "plant_owners", "power_houses", "network_operators", "generation_firms"),
        "child_table": ("telemetry_batches", "meter_logs", "yield_entries", "output_samples", "supply_snapshots", "wattage_logs"),
        "parent_id": ("grid_firm_ref", "utility_ref", "plant_owner_ref", "power_house_ref", "network_operator_ref", "generation_firm_ref"),
        "child_id": ("telemetry_ref", "meter_log_ref", "yield_entry_ref", "output_sample_ref", "supply_snapshot_ref", "wattage_log_ref"),
        "name": ("registered_title", "charter_caption", "operator_label", "official_heading", "registry_caption", "licensed_title"),
        "location": ("supply_district", "grid_sector", "plant_territory", "network_market", "service_zone", "coverage_sector"),
        "amount": ("wattage_total", "generation_value", "yield_quantity", "output_measure", "supply_volume", "production_quantity"),
        "status": ("measurement_phase", "sampling_state", "yield_condition", "output_stage", "supply_phase", "production_state"),
        "aliases": {
            "parent_one": "power company",
            "parent_many": "power companies",
            "child_one": "energy reading",
            "child_many": "energy readings",
            "name": "company name",
            "location": "coverage area",
            "amount": "power output",
            "status": "reading condition",
        },
    },
    {
        "domain": "healthcare",
        "parent_table": ("medical_practices", "clinics", "treatment_centres", "health_units", "therapy_houses", "diagnostic_firms"),
        "child_table": ("consultations", "appointments", "care_sessions", "patient_calls", "treatment_slots", "exam_notices"),
        "parent_id": ("practice_ref", "clinic_ref", "treatment_centre_ref", "health_unit_ref", "therapy_house_ref", "diagnostic_firm_ref"),
        "child_id": ("consultation_ref", "appointment_ref", "care_session_ref", "patient_call_ref", "treatment_slot_ref", "exam_notice_ref"),
        "name": ("official_title", "registry_caption", "licensed_label", "charter_heading", "registered_caption", "practice_title"),
        "location": ("treatment_district", "care_sector", "practice_territory", "patient_market", "service_zone", "coverage_sector"),
        "amount": ("billing_charge", "consultation_value", "care_price", "treatment_fee", "session_expense", "exam_cost"),
        "status": ("appointment_phase", "consultation_state", "care_condition", "treatment_stage", "session_phase", "exam_state"),
        "aliases": {
            "parent_one": "care provider",
            "parent_many": "care providers",
            "child_one": "patient visit",
            "child_many": "patient visits",
            "name": "facility name",
            "location": "service area",
            "amount": "visit price",
            "status": "booking condition",
        },
    },
    {
        "domain": "publishing",
        "parent_table": ("imprints", "press_houses", "editorial_firms", "media_groups", "print_owners", "content_houses"),
        "child_table": ("editions", "manuscripts", "print_batches", "release_notices", "volume_entries", "publication_slates"),
        "parent_id": ("imprint_ref", "press_house_ref", "editorial_firm_ref", "media_group_ref", "print_owner_ref", "content_house_ref"),
        "child_id": ("edition_ref", "manuscript_ref", "print_batch_ref", "release_notice_ref", "volume_entry_ref", "publication_slate_ref"),
        "name": ("registered_title", "legal_caption", "editorial_label", "official_heading", "registry_caption", "licensed_title"),
        "location": ("distribution_market", "circulation_sector", "editorial_territory", "reader_district", "service_zone", "coverage_sector"),
        "amount": ("production_budget", "printing_value", "edition_price", "release_fee", "volume_expense", "publication_cost"),
        "status": ("release_phase", "printing_state", "edition_condition", "publication_stage", "volume_phase", "content_state"),
        "aliases": {
            "parent_one": "media company",
            "parent_many": "media companies",
            "child_one": "released work",
            "child_many": "released works",
            "name": "publisher name",
            "location": "home area",
            "amount": "creation cost",
            "status": "publication condition",
        },
    },
)


def _make_schema(index: int, rng: random.Random) -> tuple[Schema, dict[str, str]]:
    domain = _DOMAINS[index % len(_DOMAINS)]
    variant = (index // len(_DOMAINS)) % _VARIANTS
    parent_name = domain["parent_table"][variant]
    child_name = domain["child_table"][variant]
    parent_id = domain["parent_id"][variant]
    child_id = domain["child_id"][variant]
    parent_columns = [
        Column(parent_id, "INTEGER", True, role="parent_id"),
        Column(domain["name"][variant], "TEXT", role="name"),
        Column(domain["location"][variant], "TEXT", role="location"),
        Column(f"parent_memo_{index}", "TEXT", role="parent_extra_0"),
    ]
    child_columns = [
        Column(child_id, "INTEGER", True, role="child_id"),
        Column(parent_id, "INTEGER", references=(parent_name, parent_id), role="parent_fk"),
        Column(domain["amount"][variant], "REAL", role="amount"),
        Column(domain["status"][variant], "TEXT", role="status"),
        Column(f"child_memo_{index}", "TEXT", role="child_extra_0"),
    ]
    rng.shuffle(parent_columns)
    rng.shuffle(child_columns)
    schema = Schema(
        f"anti_memory_{index:04d}",
        domain["domain"],
        (Table(parent_name, tuple(parent_columns)), Table(child_name, tuple(child_columns))),
    )
    return schema, domain["aliases"]


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
    }


def _cases(schema: Schema, aliases: dict[str, str], variant: int) -> list[tuple[str, QueryPlan, str, str]]:
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status_column = schema.role("status")[1].name
    city = CITY_VALUES[variant % len(CITY_VALUES)]
    status_value = STATUS_VALUES[variant % len(STATUS_VALUES)]
    threshold = (50, 80, 100, 125)[variant % 4]
    limit = 2 + variant % 4
    direct = _labels(schema)
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    parent_location = f"{parent.name}.{location}"
    child_status = f"{child.name}.{status_column}"
    child_amount = f"{child.name}.{amount}"

    return [
        (
            "project_parent_name",
            QueryPlan("anti_project_parent_name", parent.name, (name,)),
            f"List the {direct['name']} from all {direct['parent_many']}",
            f"Give me the {aliases['name']} for every {aliases['parent_one']}",
        ),
        (
            "count_parent_location",
            QueryPlan(
                "anti_count_parent_location",
                parent.name,
                (),
                aggregate="COUNT",
                filters=(Filter(location, "=", city),),
            ),
            f"How many {direct['parent_many']} have {direct['location']} equal to {city}",
            f"How many {aliases['parent_many']} operate around {city}",
        ),
        (
            "join_rows_location",
            QueryPlan(
                "anti_join_rows_location",
                child.name,
                (f"{child.name}.*",),
                filters=(Filter(parent_location, "=", city),),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Show all {direct['child_many']} whose {direct['parent_one']} has {direct['location']} equal to {city}",
            f"Show every {aliases['child_one']} connected to a {aliases['parent_one']} based in {city}",
        ),
        (
            "sum_child_status",
            QueryPlan(
                "anti_sum_child_status",
                child.name,
                (),
                aggregate="SUM",
                aggregate_column=amount,
                filters=(Filter(status_column, "=", status_value),),
            ),
            f"Add up {direct['amount']} for {direct['child_many']} whose {direct['status']} is {status_value}",
            f"How much {aliases['amount']} do the {aliases['child_many']} marked {status_value} have altogether",
        ),
        (
            "join_count_multi_filter",
            QueryPlan(
                "anti_join_count_multi_filter",
                child.name,
                (),
                aggregate="COUNT",
                filters=(Filter(parent_location, "=", city), Filter(child_status, "=", status_value)),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Count {direct['child_many']} where {direct['parent_one']} {direct['location']} is {city} and {direct['status']} is {status_value}",
            f"How many {aliases['child_many']} belong to {aliases['parent_many']} based in {city} and are marked {status_value}",
        ),
        (
            "join_sum_multi_filter",
            QueryPlan(
                "anti_join_sum_multi_filter",
                child.name,
                (),
                aggregate="SUM",
                aggregate_column=child_amount,
                filters=(Filter(parent_location, "=", city), Filter(child_status, "=", status_value)),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Total {direct['amount']} for {direct['child_many']} where {direct['parent_one']} {direct['location']} is {city} and {direct['status']} is {status_value}",
            f"What is the combined {aliases['amount']} for {aliases['child_many']} connected to {aliases['parent_many']} based in {city} and marked {status_value}",
        ),
        (
            "distinct_filter_limit",
            QueryPlan(
                "anti_distinct_filter_limit",
                parent.name,
                (name,),
                distinct=True,
                filters=(Filter(location, "=", city),),
                limit=limit,
            ),
            f"List {limit} distinct {direct['name']} from {direct['parent_many']} where {direct['location']} is {city}",
            f"Give me up to {limit} different {aliases['name']} for {aliases['parent_many']} operating around {city}",
        ),
        (
            "group_filter",
            QueryPlan(
                "anti_group_filter",
                child.name,
                (status_column,),
                aggregate="COUNT",
                filters=(Filter(amount, ">", threshold),),
                group_by=(status_column,),
            ),
            f"Count {direct['child_many']} by {direct['status']} where {direct['amount']} is greater than {threshold}",
            f"For {aliases['child_many']} costing more than {threshold}, give a count for each {aliases['status']}",
        ),
    ]


def _identifier_words(schema_sql: str) -> set[str]:
    words: set[str] = set()
    for identifier in schema_identifiers(schema_sql):
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier).casefold()
        words.update(part for part in snake.split("_") if len(part) > 2)
    return words


def _question_words(question: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", question.casefold()) if len(word) > 2}


def write_anti_memorization_dataset(
    output: Path,
    schemas: int = 24,
    seed: int = 424242,
    reference_data: Path | None = None,
) -> dict[str, int]:
    """Write paired direct/paraphrased prompts on unseen schemas."""
    if not 1 <= schemas <= len(_DOMAINS) * _VARIANTS:
        raise ValueError(f"schemas must be between 1 and {len(_DOMAINS) * _VARIANTS}")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict] = []
    for schema_index in range(schemas):
        schema, aliases = _make_schema(schema_index, rng)
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        database_sql = "\n".join(connection.iterdump())
        for case_index, (intent, plan, direct_question, paraphrase_question) in enumerate(
            _cases(schema, aliases, schema_index)
        ):
            sql = render_sql(plan)
            valid, reason = validate_sql(connection, sql)
            if not valid:
                raise RuntimeError(f"Invalid anti-memorization SQL for {schema.schema_id}/{intent}: {reason}")
            novelty = "held_out_composition" if held_out_composition(plan) else "familiar_composition"
            pair_id = f"{schema.schema_id}_{case_index:02d}"
            for track, question in (("direct_identifier", direct_question), ("semantic_paraphrase", paraphrase_question)):
                records.append(
                    {
                        "id": f"{pair_id}_{track}",
                        "schema_id": schema.schema_id,
                        "schema_sql": schema.sql(),
                        "database_sql": database_sql,
                        "question": question,
                        "sql": sql,
                        "query_plan": plan.normalized(),
                        "difficulty": 1
                        + int(plan.aggregate is not None)
                        + int(plan.join_table is not None)
                        + int(len(plan.filters) > 1),
                        "seed": seed,
                        "anti_memorization_pair": pair_id,
                        "anti_memorization_track": track,
                        "composition_novelty": novelty,
                        "intent": intent,
                    }
                )
        connection.close()

    records.sort(key=lambda record: record["id"])
    with (output / "anti_memorization.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    reference_records = []
    if reference_data is not None:
        reference_records = [
            json.loads(line)
            for line in reference_data.read_text(encoding="utf-8").splitlines()
            if line
        ]
    reference_questions = {record["question"].strip().casefold() for record in reference_records}
    reference_schemas = {record["schema_sql"].strip() for record in reference_records}
    reference_identifiers = _load_reference_identifiers(reference_data) if reference_data else set()
    track_overlap: dict[str, list[float]] = {}
    for track in ("direct_identifier", "semantic_paraphrase"):
        track_records = [record for record in records if record["anti_memorization_track"] == track]
        track_overlap[track] = [
            len(_identifier_words(record["schema_sql"]) & _question_words(record["question"]))
            / max(len(_question_words(record["question"])), 1)
            for record in track_records
        ]
    benchmark_identifiers = set().union(*(schema_identifiers(record["schema_sql"]) for record in records))
    report = {
        "profile": "paired_anti_memorization_v1",
        "training_use_allowed": False,
        "seed": seed,
        "records": len(records),
        "pairs": len(records) // 2,
        "schemas": schemas,
        "track_counts": dict(sorted(Counter(record["anti_memorization_track"] for record in records).items())),
        "composition_counts": dict(sorted(Counter(record["composition_novelty"] for record in records).items())),
        "intent_counts": dict(sorted(Counter(record["intent"] for record in records).items())),
        "mean_question_schema_token_overlap": {
            track: sum(values) / max(len(values), 1) for track, values in track_overlap.items()
        },
        "reference_data": str(reference_data) if reference_data else None,
        "exact_question_overlap_vs_reference": sum(
            record["question"].strip().casefold() in reference_questions for record in records
        ),
        "exact_schema_overlap_vs_reference": sum(
            record["schema_sql"].strip() in reference_schemas for record in records
        ),
        "unseen_identifier_rate_vs_reference": (
            len(benchmark_identifiers - reference_identifiers) / max(len(benchmark_identifiers), 1)
            if reference_data
            else None
        ),
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"records": len(records), "pairs": len(records) // 2, "schemas": schemas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=24)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--reference-data", type=Path)
    args = parser.parse_args()
    print(write_anti_memorization_dataset(args.output, args.schemas, args.seed, args.reference_data))


if __name__ == "__main__":
    main()
