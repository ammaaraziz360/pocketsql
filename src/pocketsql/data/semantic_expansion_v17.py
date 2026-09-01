"""Build PocketSQL v17's diverse natural-language and composition curriculum.

The corpus deliberately varies three independent axes: schema vocabulary,
question wording, and query-operation composition.  Its fresh gate uses domain
vocabulary and phrasings that are never written to the training or validation
files, so it can distinguish a larger training set from genuine transfer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile

from .composition import composition_signature, held_out_composition
from .generate import schema_identifiers
from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import CITY_VALUES, STATUS_VALUES, Column, Schema, Table
from .validate import validate_sql
from .verbalize import humanize_identifier
from .schema_linking import linking_family, weighted_resample
from .spider import _normalize_text, _schema_key, _sql_key


_TRAIN_DOMAINS = (
    {
        "domain": "procurement",
        "parent_table": ("manufacturers", "makers", "producers", "factories"),
        "child_table": ("deliveries", "supply_runs", "receipts", "dropoffs"),
        "parent_id": ("maker_ref", "producer_key", "factory_ref", "source_key"),
        "child_id": ("delivery_ref", "supply_run_key", "receipt_ref", "dropoff_key"),
        "name": ("company_title", "maker_label", "producer_name", "factory_title"),
        "location": ("operating_area", "source_district", "production_zone", "factory_region"),
        "amount": ("shipping_cost", "supply_value", "receipt_amount", "dropoff_charge"),
        "status": ("delivery_state", "supply_phase", "receipt_status", "dropoff_stage"),
        "aliases": ("supplier", "suppliers", "delivery", "deliveries", "business name", "base area", "delivery cost", "delivery progress"),
    },
    {
        "domain": "workforce",
        "parent_table": ("employers", "departments", "work_groups", "staff_units"),
        "child_table": ("work_logs", "shift_entries", "time_cards", "labor_records"),
        "parent_id": ("employer_ref", "department_key", "work_group_ref", "staff_unit_key"),
        "child_id": ("work_log_ref", "shift_entry_key", "time_card_ref", "labor_record_key"),
        "name": ("employer_title", "department_label", "group_name", "unit_title"),
        "location": ("office_area", "work_district", "staff_zone", "employment_region"),
        "amount": ("hours_total", "shift_value", "time_amount", "labor_hours"),
        "status": ("work_state", "shift_phase", "time_status", "labor_stage"),
        "aliases": ("workplace", "workplaces", "time entry", "time entries", "workplace name", "office area", "hours worked", "work status"),
    },
    {
        "domain": "property",
        "parent_table": ("property_owners", "lessors", "housing_groups", "estate_holders"),
        "child_table": ("rental_agreements", "tenancies", "occupancy_records", "lease_files"),
        "parent_id": ("owner_ref", "lessor_key", "housing_group_ref", "estate_holder_key"),
        "child_id": ("rental_ref", "tenancy_key", "occupancy_ref", "lease_file_key"),
        "name": ("owner_title", "lessor_label", "housing_name", "estate_title"),
        "location": ("property_area", "rental_district", "housing_zone", "estate_region"),
        "amount": ("monthly_rent", "tenancy_value", "occupancy_cost", "lease_charge"),
        "status": ("rental_state", "tenancy_phase", "occupancy_status", "lease_stage"),
        "aliases": ("property owner", "property owners", "rental", "rentals", "owner name", "property area", "rent amount", "rental status"),
    },
    {
        "domain": "hospitality",
        "parent_table": ("lodgings", "inn_groups", "resorts", "guest_houses"),
        "child_table": ("stays", "room_visits", "guest_sessions", "overnights"),
        "parent_id": ("lodging_ref", "inn_group_key", "resort_ref", "guest_house_key"),
        "child_id": ("stay_ref", "room_visit_key", "guest_session_ref", "overnight_key"),
        "name": ("lodging_title", "inn_label", "resort_name", "guest_house_title"),
        "location": ("lodging_area", "inn_district", "resort_zone", "guest_house_region"),
        "amount": ("stay_cost", "room_charge", "guest_spend", "overnight_price"),
        "status": ("stay_state", "room_phase", "guest_status", "overnight_stage"),
        "aliases": ("hotel", "hotels", "guest stay", "guest stays", "hotel name", "hotel area", "stay price", "stay status"),
    },
    {
        "domain": "insurance",
        "parent_table": ("insured_members", "coverage_holders", "benefit_members", "plan_owners"),
        "child_table": ("reimbursements", "benefit_requests", "payout_files", "coverage_cases"),
        "parent_id": ("insured_ref", "coverage_holder_key", "benefit_member_ref", "plan_owner_key"),
        "child_id": ("reimbursement_ref", "benefit_request_key", "payout_file_ref", "coverage_case_key"),
        "name": ("insured_name", "holder_label", "member_title", "owner_name"),
        "location": ("coverage_area", "holder_district", "benefit_zone", "plan_region"),
        "amount": ("claim_amount", "benefit_amount", "payout_total", "coverage_cost"),
        "status": ("reimbursement_state", "benefit_phase", "payout_status", "coverage_stage"),
        "aliases": ("policy member", "policy members", "claim", "claims", "member name", "coverage area", "claim amount", "claim status"),
    },
    {
        "domain": "telecom",
        "parent_table": ("network_firms", "phone_groups", "signal_owners", "service_operators"),
        "child_table": ("service_sessions", "call_batches", "data_records", "connection_logs"),
        "parent_id": ("network_firm_ref", "phone_group_key", "signal_owner_ref", "service_operator_key"),
        "child_id": ("service_session_ref", "call_batch_key", "data_record_ref", "connection_log_key"),
        "name": ("network_title", "phone_group_label", "signal_owner_name", "operator_title"),
        "location": ("network_area", "phone_district", "signal_zone", "service_region"),
        "amount": ("usage_charge", "call_value", "data_amount", "connection_cost"),
        "status": ("session_state", "call_phase", "data_status", "connection_stage"),
        "aliases": ("phone company", "phone companies", "usage record", "usage records", "provider name", "service area", "usage cost", "connection status"),
    },
    {
        "domain": "agriculture",
        "parent_table": ("farms", "grower_groups", "orchards", "field_owners"),
        "child_table": ("crop_batches", "harvest_runs", "produce_lots", "yield_records"),
        "parent_id": ("farm_ref", "grower_group_key", "orchard_ref", "field_owner_key"),
        "child_id": ("crop_batch_ref", "harvest_run_key", "produce_lot_ref", "yield_record_key"),
        "name": ("farm_title", "grower_label", "orchard_name", "field_owner_title"),
        "location": ("farm_area", "grower_district", "orchard_zone", "field_region"),
        "amount": ("crop_weight", "harvest_amount", "produce_total", "yield_value"),
        "status": ("crop_state", "harvest_phase", "produce_status", "yield_stage"),
        "aliases": ("farm business", "farm businesses", "harvest", "harvests", "farm name", "growing area", "harvest amount", "harvest status"),
    },
    {
        "domain": "automotive",
        "parent_table": ("dealerships", "garage_groups", "motor_shops", "vehicle_centres"),
        "child_table": ("service_orders", "repair_visits", "workshop_jobs", "maintenance_files"),
        "parent_id": ("dealership_ref", "garage_group_key", "motor_shop_ref", "vehicle_centre_key"),
        "child_id": ("service_order_ref", "repair_visit_key", "workshop_job_ref", "maintenance_file_key"),
        "name": ("dealer_title", "garage_label", "shop_name", "centre_title"),
        "location": ("dealer_area", "garage_district", "shop_zone", "centre_region"),
        "amount": ("service_cost", "repair_charge", "job_amount", "maintenance_price"),
        "status": ("service_state", "repair_phase", "job_status", "maintenance_stage"),
        "aliases": ("auto shop", "auto shops", "repair job", "repair jobs", "shop name", "shop area", "repair cost", "repair status"),
    },
    {
        "domain": "nonprofit",
        "parent_table": ("charities", "aid_groups", "foundations", "community_funds"),
        "child_table": ("donations", "gift_batches", "grant_entries", "funding_records"),
        "parent_id": ("charity_ref", "aid_group_key", "foundation_ref", "community_fund_key"),
        "child_id": ("donation_ref", "gift_batch_key", "grant_entry_ref", "funding_record_key"),
        "name": ("charity_title", "aid_group_label", "foundation_name", "fund_title"),
        "location": ("charity_area", "aid_district", "foundation_zone", "fund_region"),
        "amount": ("donation_value", "gift_amount", "grant_total", "funding_value"),
        "status": ("donation_state", "gift_phase", "grant_status", "funding_stage"),
        "aliases": ("charity", "charities", "contribution", "contributions", "charity name", "service area", "contribution amount", "funding status"),
    },
    {
        "domain": "travel",
        "parent_table": ("travel_agencies", "tour_groups", "trip_planners", "holiday_firms"),
        "child_table": ("itineraries", "tour_bookings", "trip_files", "holiday_records"),
        "parent_id": ("agency_ref", "tour_group_key", "trip_planner_ref", "holiday_firm_key"),
        "child_id": ("itinerary_ref", "tour_booking_key", "trip_file_ref", "holiday_record_key"),
        "name": ("agency_title", "tour_group_label", "planner_name", "holiday_firm_title"),
        "location": ("agency_area", "tour_district", "planner_zone", "holiday_region"),
        "amount": ("trip_cost", "tour_charge", "booking_amount", "holiday_price"),
        "status": ("trip_state", "tour_phase", "booking_status", "holiday_stage"),
        "aliases": ("travel company", "travel companies", "trip booking", "trip bookings", "agency name", "travel area", "trip price", "booking status"),
    },
)


_GATE_DOMAINS = (
    {
        "domain": "aviation",
        "parent_table": ("air_operators", "flight_houses", "sky_companies", "route_airlines"),
        "child_table": ("flight_legs", "air_journeys", "route_segments", "departure_files"),
        "parent_id": ("air_operator_ref", "flight_house_key", "sky_company_ref", "route_airline_key"),
        "child_id": ("flight_leg_ref", "air_journey_key", "route_segment_ref", "departure_file_key"),
        "name": ("airline_heading", "flight_house_label", "sky_company_name", "airline_title"),
        "location": ("operator_area", "flight_district", "sky_zone", "airline_region"),
        "amount": ("flight_fare", "journey_charge", "segment_amount", "departure_price"),
        "status": ("flight_state", "journey_phase", "segment_status", "departure_stage"),
        "aliases": ("airline", "airlines", "flight", "flights", "airline name", "operating area", "ticket price", "flight status"),
    },
    {
        "domain": "music",
        "parent_table": ("record_labels", "music_houses", "audio_companies", "sound_groups"),
        "child_table": ("album_releases", "music_editions", "audio_drops", "recording_files"),
        "parent_id": ("record_label_ref", "music_house_key", "audio_company_ref", "sound_group_key"),
        "child_id": ("album_release_ref", "music_edition_key", "audio_drop_ref", "recording_file_key"),
        "name": ("label_title", "music_house_label", "audio_company_name", "sound_group_title"),
        "location": ("label_area", "music_district", "audio_zone", "sound_region"),
        "amount": ("release_budget", "edition_cost", "audio_spend", "recording_price"),
        "status": ("release_state", "edition_phase", "audio_status", "recording_stage"),
        "aliases": ("music label", "music labels", "album", "albums", "label name", "home area", "release cost", "release status"),
    },
    {
        "domain": "construction",
        "parent_table": ("building_contractors", "trade_groups", "site_companies", "works_firms"),
        "child_table": ("building_projects", "trade_jobs", "site_packages", "works_orders"),
        "parent_id": ("contractor_ref", "trade_group_key", "site_company_ref", "works_firm_key"),
        "child_id": ("building_project_ref", "trade_job_key", "site_package_ref", "works_order_key"),
        "name": ("contractor_title", "trade_group_label", "site_company_name", "works_firm_title"),
        "location": ("contractor_area", "trade_district", "site_zone", "works_region"),
        "amount": ("project_budget", "job_cost", "package_amount", "works_price"),
        "status": ("project_state", "job_phase", "package_status", "works_stage"),
        "aliases": ("builder", "builders", "construction job", "construction jobs", "builder name", "work area", "project cost", "project status"),
    },
)


def _style(name: str, style: str, table: bool = False) -> str:
    if style == "plain":
        return name
    if style == "camel":
        parts = name.split("_")
        return parts[0] + "".join(part.capitalize() for part in parts[1:])
    if style == "prefixed":
        return ("tbl_" if table else "col_") + name
    return name + ("_data" if table else "_value")


def _make_schema(index: int, rng: random.Random, domains: tuple[dict, ...], prefix: str) -> tuple[Schema, dict]:
    spec = domains[index % len(domains)]
    style = rng.choice(("plain", "camel", "prefixed", "suffixed"))
    vocabulary_variant = rng.randrange(len(spec["parent_table"]))
    parent_name = _style(spec["parent_table"][vocabulary_variant], style, True)
    child_name = _style(spec["child_table"][vocabulary_variant], style, True)
    parent_id = _style(spec["parent_id"][vocabulary_variant], style)
    child_id = _style(spec["child_id"][vocabulary_variant], style)
    name = _style(spec["name"][vocabulary_variant], style)
    location = _style(spec["location"][vocabulary_variant], style)
    amount = _style(spec["amount"][vocabulary_variant], style)
    status = _style(spec["status"][vocabulary_variant], style)
    parent_extra = _style(f"{spec['domain']}_parent_note_{index % 7}", style)
    child_extra = _style(f"{spec['domain']}_child_note_{index % 7}", style)
    parent_columns = [
        Column(parent_id, "INTEGER", True, role="parent_id"),
        Column(name, "TEXT", role="name"),
        Column(location, "TEXT", role="location"),
        Column(parent_extra, "TEXT", role="parent_extra_0"),
    ]
    child_columns = [
        Column(child_id, "INTEGER", True, role="child_id"),
        Column(parent_id, "INTEGER", references=(parent_name, parent_id), role="parent_fk"),
        Column(amount, "REAL", role="amount"),
        Column(status, "TEXT", role="status"),
        Column(child_extra, "TEXT", role="child_extra_0"),
    ]
    rng.shuffle(parent_columns)
    rng.shuffle(child_columns)
    schema = Schema(
        f"{prefix}_{index:04d}",
        spec["domain"],
        (Table(parent_name, tuple(parent_columns)), Table(child_name, tuple(child_columns))),
    )
    return schema, spec


def _unique_schemas(count: int, rng: random.Random, domains: tuple[dict, ...], prefix: str) -> list[tuple[Schema, dict]]:
    selected: list[tuple[Schema, dict]] = []
    seen: set[str] = set()
    candidate = 0
    while len(selected) < count:
        schema, spec = _make_schema(candidate, rng, domains, prefix)
        candidate += 1
        if schema.sql() in seen:
            continue
        seen.add(schema.sql())
        selected.append((replace(schema, schema_id=f"{prefix}_{len(selected):04d}"), spec))
    return selected


def _labels(schema: Schema, spec: dict, semantic: bool) -> dict[str, str]:
    parent, child = schema.tables
    if semantic:
        parent_one, parent_many, child_one, child_many, name, location, amount, status = spec["aliases"]
        return {
            "parent_one": parent_one,
            "parent_many": parent_many,
            "child_one": child_one,
            "child_many": child_many,
            "name": name,
            "location": location,
            "amount": amount,
            "status": status,
            "parent_id": f"{parent_one} reference number",
            "child_id": f"{child_one} reference number",
        }
    return {
        "parent_one": humanize_identifier(parent.name, "singular"),
        "parent_many": humanize_identifier(parent.name, "plural"),
        "child_one": humanize_identifier(child.name, "singular"),
        "child_many": humanize_identifier(child.name, "plural"),
        "name": humanize_identifier(schema.role("name")[1].name),
        "location": humanize_identifier(schema.role("location")[1].name),
        "amount": humanize_identifier(schema.role("amount")[1].name),
        "status": humanize_identifier(schema.role("status")[1].name),
        "parent_id": humanize_identifier(schema.role("parent_id")[1].name),
        "child_id": humanize_identifier(schema.role("child_id")[1].name),
    }


def _cases(schema: Schema, variant: int, name_value: str, city: str) -> list[tuple[str, QueryPlan]]:
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_id = schema.role("child_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    status_value = STATUS_VALUES[variant % len(STATUS_VALUES)]
    threshold = (50, 100, 150, 200)[variant % 4]
    limit = 2 + variant % 4
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    parent_location = f"{parent.name}.{location}"
    parent_name = f"{parent.name}.{name}"
    child_status = f"{child.name}.{status}"
    child_amount = f"{child.name}.{amount}"
    plans = [
        QueryPlan("v17_project_name", parent.name, (name,)),
        QueryPlan("v17_project_id_location", parent.name, (parent_id, location)),
        QueryPlan("v17_parent_named", parent.name, (f"{parent.name}.*",), filters=(Filter(name, "=", name_value),)),
        QueryPlan("v17_name_by_location", parent.name, (name,), filters=(Filter(location, "=", city),)),
        QueryPlan("v17_count_location", parent.name, (), aggregate="COUNT", filters=(Filter(location, "=", city),)),
        QueryPlan("v17_distinct_location", parent.name, (location,), distinct=True),
        QueryPlan("v17_sum_status", child.name, (), aggregate="SUM", aggregate_column=amount, filters=(Filter(status, "=", status_value),)),
        QueryPlan("v17_average_threshold", child.name, (), aggregate="AVG", aggregate_column=amount, filters=(Filter(amount, ">", threshold),)),
        QueryPlan("v17_child_two_filter", child.name, (child_id, amount), filters=(Filter(status, "=", status_value), Filter(amount, ">", threshold))),
        QueryPlan("v17_group_status", child.name, (status,), aggregate="COUNT", group_by=(status,)),
        QueryPlan("v17_group_filter", child.name, (status,), aggregate="COUNT", filters=(Filter(amount, ">", threshold),), group_by=(status,)),
        QueryPlan("v17_join_rows_location", child.name, (f"{child.name}.*",), filters=(Filter(parent_location, "=", city),), join_table=parent.name, join_on=join_on),
        QueryPlan("v17_join_amount_name", child.name, (child_amount,), filters=(Filter(parent_name, "=", name_value),), join_table=parent.name, join_on=join_on),
        QueryPlan("v17_join_count_location_status", child.name, (), aggregate="COUNT", filters=(Filter(parent_location, "=", city), Filter(child_status, "=", status_value)), join_table=parent.name, join_on=join_on),
        QueryPlan("v17_join_sum_location_status", child.name, (), aggregate="SUM", aggregate_column=child_amount, filters=(Filter(parent_location, "=", city), Filter(child_status, "=", status_value)), join_table=parent.name, join_on=join_on),
        QueryPlan("v17_join_sum_name_location", child.name, (), aggregate="SUM", aggregate_column=child_amount, filters=(Filter(parent_name, "=", name_value), Filter(parent_location, "=", city)), join_table=parent.name, join_on=join_on),
        QueryPlan("v17_distinct_filter_limit", parent.name, (name,), distinct=True, filters=(Filter(location, "=", city),), limit=limit),
        QueryPlan("v17_join_projection_status", child.name, (parent_name, child_amount), filters=(Filter(child_status, "=", status_value),), join_table=parent.name, join_on=join_on),
    ]
    return [(plan.family, replace(plan, family=composition_signature(plan))) for plan in plans]


def _question(intent: str, labels: dict[str, str], values: dict[str, object], variant: int) -> str:
    p, ps, c, cs = labels["parent_one"], labels["parent_many"], labels["child_one"], labels["child_many"]
    n, loc, amt, status = labels["name"], labels["location"], labels["amount"], labels["status"]
    name_value, city = values["name"], values["city"]
    state, threshold, limit = values["status"], values["threshold"], values["limit"]
    article = "an" if p[0].casefold() in "aeiou" else "a"
    templates = {
        "v17_project_name": (
            f"show me the {n} for every {p}", f"list all {n} from the {ps}", f"what are the {n} of our {ps}", f"pull up each {p}'s {n}", f"I just need the {n} for {ps}",
        ),
        "v17_project_id_location": (
            f"show the {labels['parent_id']} and {loc} for all {ps}", f"list every {p} with its {labels['parent_id']} and {loc}", f"give me {labels['parent_id']} plus {loc} from {ps}", f"which reference numbers and areas belong to the {ps}", f"I need each {p}'s reference and area",
        ),
        "v17_parent_named": (
            f"show the {p} whose {n} is {name_value}", f"find the {p} named {name_value}", f"pull up the full record for {name_value}", f"which {p} is called {name_value}", f"get me {name_value}'s {p} record",
        ),
        "v17_name_by_location": (
            f"show {n} for {ps} where {loc} is {city}", f"list the {ps} in {city} by {n}", f"which {n} belong to {ps} based in {city}", f"give me names of {ps} operating around {city}", f"who are the {ps} from {city}",
        ),
        "v17_count_location": (
            f"how many {ps} have {loc} equal to {city}", f"count the {ps} located in {city}", f"what is the number of {ps} from {city}", f"tell me how many {ps} operate around {city}", f"{ps} in {city}, how many are there",
        ),
        "v17_distinct_location": (
            f"list the distinct {loc} among {ps}", f"which different {loc} are represented by {ps}", f"show every unique {loc} from the {ps}", f"what areas do the {ps} cover without repeats", f"give me the different places where {ps} are based",
        ),
        "v17_sum_status": (
            f"sum {amt} for {cs} where {status} is {state}", f"add up the {amt} of {state} {cs}", f"what total {amt} belongs to {cs} marked {state}", f"how much do the {state} {cs} amount to altogether", f"total cost for {state} {cs}",
        ),
        "v17_average_threshold": (
            f"find the average {amt} for {cs} above {threshold}", f"what is the mean {amt} when {amt} exceeds {threshold}", f"average the {amt} of {cs} costing more than {threshold}", f"for {cs} over {threshold}, what is the typical {amt}", f"average cost of {cs} above {threshold}",
        ),
        "v17_child_two_filter": (
            f"show {labels['child_id']} and {amt} where {status} is {state} and {amt} is above {threshold}", f"list references and amounts for {state} {cs} over {threshold}", f"which {cs} marked {state} cost more than {threshold}", f"give me the id and amount of every {state} {c} above {threshold}", f"{state} {cs} over {threshold}, with ids and amounts",
        ),
        "v17_group_status": (
            f"count {cs} for each {status}", f"break down the number of {cs} by {status}", f"for every {status}, show how many {cs} there are", f"give a separate {c} count for each {status}", f"counts of {cs} by status",
        ),
        "v17_group_filter": (
            f"count {cs} by {status} where {amt} is above {threshold}", f"for {cs} over {threshold}, give a count for each {status}", f"break down qualifying {cs} by {status}, only using amounts above {threshold}", f"how many {cs} over {threshold} fall into each {status}", f"status counts for {cs} costing more than {threshold}",
        ),
        "v17_join_rows_location": (
            f"show all {cs} whose {p} has {loc} equal to {city}", f"list {cs} belonging to {ps} in {city}", f"which {cs} are tied to {article} {p} based in {city}", f"pull up every {c} connected to {ps} around {city}", f"{cs} from {ps} in {city}",
        ),
        "v17_join_amount_name": (
            f"show {amt} for {cs} whose {p} has {n} {name_value}", f"list the amounts attached to {name_value}'s {cs}", f"what {amt} belongs to {cs} connected to {name_value}", f"pull the costs for every {c} from the {p} called {name_value}", f"{name_value}'s {c} amounts",
        ),
        "v17_join_count_location_status": (
            f"count {cs} where the {p} {loc} is {city} and {status} is {state}", f"how many {state} {cs} belong to {ps} in {city}", f"give the number of {cs} tied to {city} {ps} and marked {state}", f"for {ps} around {city}, count their {state} {cs}", f"number of {state} {cs} from {city} {ps}",
        ),
        "v17_join_sum_location_status": (
            f"sum {amt} for {cs} where the {p} {loc} is {city} and {status} is {state}", f"total the cost of {state} {cs} belonging to {ps} in {city}", f"what combined {amt} comes from {state} {cs} tied to {city} {ps}", f"for {ps} around {city}, add up their {state} {c} amounts", f"total amount of {state} {cs} from {city} {ps}",
        ),
        "v17_join_sum_name_location": (
            f"sum {amt} for {cs} whose {p} {n} is {name_value} and {loc} is {city}", f"what is the total cost of {cs} tied to {name_value} in {city}", f"add up {amt} for the {p} called {name_value} when it is based in {city}", f"how much do {name_value}'s {cs} amount to in {city}", f"{name_value} in {city}: total {c} amount",
        ),
        "v17_distinct_filter_limit": (
            f"list {limit} distinct {n} from {ps} where {loc} is {city}", f"give me up to {limit} different {ps} by name in {city}", f"show {limit} unique names of {ps} based around {city}", f"without duplicates, return at most {limit} {ps} from {city}", f"first {limit} different {p} names in {city}",
        ),
        "v17_join_projection_status": (
            f"show {n} and {amt} for {cs} where {status} is {state}", f"list each {p} name with the amount of its {state} {cs}", f"which {ps} have {state} {cs} and what are their amounts", f"connect {cs} to {ps} and give names plus costs for those marked {state}", f"names and amounts for {state} {cs}",
        ),
    }
    return templates[intent][variant % len(templates[intent])]


def _records_for(
    schemas: list[tuple[Schema, dict]],
    seed: int,
    rng: random.Random,
    tracks: tuple[str, ...],
    phrase_offset: int,
    training_use_allowed: bool,
    gate_phrasing: bool = False,
) -> list[dict]:
    records: list[dict] = []
    names = ("Acme", "Atlas", "Nova", "Summit")
    for schema_index, (schema, spec) in enumerate(schemas):
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        parent = schema.tables[0]
        parent_id = schema.role("parent_id")[1].name
        name_column = schema.role("name")[1].name
        location_column = schema.role("location")[1].name
        name_value = names[schema_index % len(names)]
        connection.execute(
            f"UPDATE {parent.name} SET {name_column} = ? WHERE {parent_id} = 1",
            (name_value,),
        )
        city = connection.execute(
            f"SELECT {location_column} FROM {parent.name} WHERE {parent_id} = 1"
        ).fetchone()[0]
        connection.commit()
        database_sql = "\n".join(connection.iterdump())
        status_value = STATUS_VALUES[schema_index % len(STATUS_VALUES)]
        threshold = (50, 100, 150, 200)[schema_index % 4]
        limit = 2 + schema_index % 4
        values = {"name": name_value, "city": city, "status": status_value, "threshold": threshold, "limit": limit}
        for case_index, (intent, plan) in enumerate(_cases(schema, schema_index, name_value, city)):
            sql = render_sql(plan)
            valid, reason = validate_sql(connection, sql)
            if not valid:
                raise RuntimeError(f"Invalid v17 SQL for {schema.schema_id}/{intent}: {reason}")
            for track_index, track in enumerate(tracks):
                labels = _labels(schema, spec, semantic=track != "direct_identifier")
                if track == "terse_request":
                    template_variant = 4
                elif phrase_offset >= 3:
                    template_variant = phrase_offset
                else:
                    template_variant = (schema_index + case_index + track_index) % 3
                question = _question(intent, labels, values, template_variant)
                if gate_phrasing:
                    question = question.rstrip("?.") + ", please?"
                records.append(
                    {
                        "id": f"{schema.schema_id}_{case_index:02d}_{track}",
                        "schema_id": schema.schema_id,
                        "schema_sql": schema.sql(),
                        "database_sql": database_sql,
                        "question": question,
                        "sql": sql,
                        "query_plan": plan.normalized(),
                        "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None) + int(len(plan.filters) > 1) + int(bool(plan.group_by)),
                        "seed": seed,
                        "semantic_linking_pair": f"{schema.schema_id}_{case_index:02d}",
                        "semantic_linking_track": track,
                        "intent": intent,
                        "composition_signature": composition_signature(plan),
                        "historically_held_out_composition": held_out_composition(plan),
                        "source": {
                            "dataset": "PocketSQL semantic expansion v17",
                            "domain": schema.domain,
                            "human_authored": False,
                            "training_use_allowed": training_use_allowed,
                        },
                    }
                )
        connection.close()
    return records


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_semantic_expansion_dataset(
    output: Path,
    train_schemas: int = 300,
    validation_schemas: int = 36,
    gate_schemas: int = 24,
    seed: int = 171717,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite v17 semantic corpus directory: {output}")
    rng = random.Random(seed)
    regular = _unique_schemas(train_schemas + validation_schemas, rng, _TRAIN_DOMAINS, "semantic_v17")
    gate = _unique_schemas(gate_schemas, rng, _GATE_DOMAINS, "semantic_v17_gate")
    train = _records_for(
        regular[:train_schemas], seed, rng,
        ("direct_identifier", "semantic_paraphrase", "terse_request"), 0, True,
    )
    paired_validation = _records_for(
        regular[train_schemas:], seed, rng,
        ("direct_identifier", "semantic_paraphrase"), 3, False,
    )
    fresh_gate = _records_for(
        gate, seed, rng,
        ("direct_identifier", "semantic_paraphrase"), 4, False, True,
    )
    validation = [r for r in paired_validation if r["semantic_linking_track"] == "semantic_paraphrase"]
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(paired_validation)
    fresh_gate.sort(key=lambda record: record["id"])

    train_schemas_sql = {r["schema_sql"] for r in train}
    validation_schemas_sql = {r["schema_sql"] for r in paired_validation}
    gate_schemas_sql = {r["schema_sql"] for r in fresh_gate}
    if train_schemas_sql & validation_schemas_sql or (train_schemas_sql | validation_schemas_sql) & gate_schemas_sql:
        raise RuntimeError("v17 schema splits overlap")
    train_questions = {r["question"].strip().casefold() for r in train}
    gate_questions = {r["question"].strip().casefold() for r in fresh_gate}
    train_identifiers = set().union(*(schema_identifiers(value) for value in train_schemas_sql))
    gate_identifiers = set().union(*(schema_identifiers(value) for value in gate_schemas_sql))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as name:
        temporary = Path(name)
        _write(temporary / "train.jsonl", train)
        _write(temporary / "validation.jsonl", validation)
        _write(temporary / "paired_validation.jsonl", paired_validation)
        _write(temporary / "fresh_gate.jsonl", fresh_gate)
        report = {
            "profile": "semantic_expansion_v17",
            "seed": seed,
            "training_use_allowed": {"train": True, "validation": False, "paired_validation": False, "fresh_gate": False},
            "records": {"train": len(train), "validation": len(validation), "paired_validation": len(paired_validation), "fresh_gate": len(fresh_gate)},
            "schemas": {"train": train_schemas, "validation": validation_schemas, "fresh_gate": gate_schemas},
            "tracks": dict(sorted(Counter(r["semantic_linking_track"] for r in train).items())),
            "intents": dict(sorted(Counter(r["intent"] for r in train).items())),
            "historically_held_out_training_records": sum(r["historically_held_out_composition"] is not None for r in train),
            "historically_held_out_compositions": dict(sorted(Counter(r["historically_held_out_composition"] for r in train if r["historically_held_out_composition"]).items())),
            "isolation": {
                "train_validation_schema_overlap": len(train_schemas_sql & validation_schemas_sql),
                "training_gate_schema_overlap": len(train_schemas_sql & gate_schemas_sql),
                "training_gate_exact_question_overlap": len(train_questions & gate_questions),
                "training_gate_identifier_overlap": len(train_identifiers & gate_identifiers),
                "fresh_gate_in_training": False,
            },
        }
        (temporary / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return report


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sample(records: list[dict], count: int, seed: int, source: str, evaluation_track: str | None = None) -> list[dict]:
    if count > len(records):
        raise ValueError(f"{source} count {count} exceeds {len(records)} records")
    selected = random.Random(seed).sample(records, count)
    return [
        {
            **record,
            "id": f"v17:{source}:{index}:{record.get('id', index)}",
            "v17_source": source,
            **({"evaluation_track": evaluation_track} if evaluation_track else {}),
        }
        for index, record in enumerate(selected)
    ]


def _uniform_weights(records: list[dict]) -> dict[str, float]:
    return {linking_family(record): 1.0 for record in records}


def _pair_overlap_counts(training: list[dict], reference: list[dict]) -> dict[str, int]:
    """Count exact two-field overlaps used by PocketSQL's contamination policy."""
    indexes = {name: set() for name in ("schema_question", "schema_sql", "question_sql")}
    for record in training:
        schema = _schema_key(record)
        question = _normalize_text(record["question"])
        sql = _sql_key(record)
        indexes["schema_question"].add((schema, question))
        indexes["schema_sql"].add((schema, sql))
        indexes["question_sql"].add((question, sql))
    counts: Counter = Counter()
    for record in reference:
        schema = _schema_key(record)
        question = _normalize_text(record["question"])
        sql = _sql_key(record)
        counts.update(
            name
            for name, key in (
                ("schema_question", (schema, question)),
                ("schema_sql", (schema, sql)),
                ("question_sql", (question, sql)),
            )
            if key in indexes[name]
        )
    return dict(sorted(counts.items()))


def build_v17_mixture(
    output: Path,
    semantic_train_path: Path,
    semantic_validation_path: Path,
    human_train_path: Path,
    human_validation_path: Path,
    synthetic_train_path: Path,
    synthetic_validation_path: Path,
    frozen_human_path: Path,
    fresh_gate_path: Path,
    total_train_records: int = 30000,
    human_train_records: int = 8000,
    human_validation_records: int = 500,
    synthetic_validation_records: int = 400,
    seed: int = 171718,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite v17 mixture directory: {output}")
    semantic_train = _load(semantic_train_path)
    semantic_validation = _load(semantic_validation_path)
    human_train = _load(human_train_path)
    human_validation = _load(human_validation_path)
    synthetic_train = _load(synthetic_train_path)
    synthetic_validation = _load(synthetic_validation_path)
    frozen_human = _load(frozen_human_path)
    fresh_gate = _load(fresh_gate_path)
    if any(not r.get("source", {}).get("training_use_allowed") for r in human_train):
        raise ValueError("human training input contains evaluation-only data")
    if any(r.get("source", {}).get("training_use_allowed") for r in fresh_gate):
        raise ValueError("fresh gate contains training-enabled records")

    train_db = {r.get("source", {}).get("db_id") for r in human_train}
    validation_db = {r.get("source", {}).get("db_id") for r in human_validation}
    frozen_db = {r.get("source", {}).get("db_id") for r in frozen_human}
    if train_db & validation_db or (train_db | validation_db) & frozen_db:
        raise ValueError("human train, validation, and frozen schemas must be disjoint")
    semantic_schema = {r["schema_sql"] for r in semantic_train}
    gate_schema = {r["schema_sql"] for r in fresh_gate}
    if semantic_schema & gate_schema:
        raise ValueError("semantic training schemas overlap the fresh gate")

    selected_human, human_families = weighted_resample(
        human_train, human_train_records, _uniform_weights(human_train), seed, "v17_human"
    )
    selected_human = [{**r, "v17_source": "recovered_human"} for r in selected_human]
    semantic = [{**r, "id": f"v17:semantic:{r['id']}", "v17_source": "diverse_semantic"} for r in semantic_train]
    synthetic_count = total_train_records - len(semantic) - len(selected_human)
    if synthetic_count < 0:
        raise ValueError("total_train_records is smaller than semantic plus human allocations")
    selected_synthetic = _sample(synthetic_train, synthetic_count, seed + 1, "broad_replay")
    selected_human_validation, validation_families = weighted_resample(
        human_validation, human_validation_records, _uniform_weights(human_validation), seed + 2, "v17_human_validation"
    )
    selected_human_validation = [
        {**r, "v17_source": "human_validation", "evaluation_track": "human_validation"}
        for r in selected_human_validation
    ]
    selected_semantic_validation = [
        {**r, "id": f"v17:semantic_validation:{r['id']}", "v17_source": "semantic_validation", "evaluation_track": "semantic_validation"}
        for r in semantic_validation
    ]
    selected_synthetic_validation = _sample(
        synthetic_validation, synthetic_validation_records, seed + 3, "synthetic_validation", "synthetic_validation"
    )
    train = [*semantic, *selected_human, *selected_synthetic]
    validation = [*selected_semantic_validation, *selected_human_validation, *selected_synthetic_validation]
    validation_pair_overlaps = _pair_overlap_counts(train, validation)
    gate_pair_overlaps = _pair_overlap_counts(train, fresh_gate)
    frozen_pair_overlaps = _pair_overlap_counts(train, frozen_human)
    if validation_pair_overlaps or gate_pair_overlaps or frozen_pair_overlaps:
        raise ValueError(
            "v17 record-pair contamination detected: "
            f"validation={validation_pair_overlaps}, gate={gate_pair_overlaps}, "
            f"frozen_human={frozen_pair_overlaps}"
        )
    rng = random.Random(seed + 4)
    rng.shuffle(train)
    rng.shuffle(validation)

    report = {
        "profile": "semantic_expansion_v17_mixture",
        "seed": seed,
        "inputs": {str(p): _sha256(p) for p in (semantic_train_path, semantic_validation_path, human_train_path, human_validation_path, synthetic_train_path, synthetic_validation_path, frozen_human_path, fresh_gate_path)},
        "train": {
            "records": len(train),
            "source_counts": dict(sorted(Counter(r["v17_source"] for r in train).items())),
            "unique_human_records": len(human_train),
            "human_family_counts": human_families,
        },
        "validation": {
            "records": len(validation),
            "source_counts": dict(sorted(Counter(r["evaluation_track"] for r in validation).items())),
            "unique_human_records": len(human_validation),
            "human_family_counts": validation_families,
        },
        "isolation": {
            "human_train_validation_schema_overlap": len(train_db & validation_db),
            "human_train_frozen_schema_overlap": len(train_db & frozen_db),
            "human_validation_frozen_schema_overlap": len(validation_db & frozen_db),
            "semantic_training_fresh_gate_schema_overlap": len(semantic_schema & gate_schema),
            "training_validation_pair_overlaps": validation_pair_overlaps,
            "training_fresh_gate_pair_overlaps": gate_pair_overlaps,
            "training_frozen_human_pair_overlaps": frozen_pair_overlaps,
            "fresh_gate_records_in_training": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as name:
        temporary = Path(name)
        databases = human_train_path.parent / "databases"
        if databases.exists():
            shutil.copytree(databases, temporary / "databases")
        _write(temporary / "mixed_train.jsonl", train)
        _write(temporary / "mixed_validation.jsonl", validation)
        (temporary / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--train-schemas", type=int, default=300)
    generate.add_argument("--validation-schemas", type=int, default=36)
    generate.add_argument("--gate-schemas", type=int, default=24)
    generate.add_argument("--seed", type=int, default=171717)
    mix = subparsers.add_parser("mix")
    mix.add_argument("--output", type=Path, required=True)
    mix.add_argument("--semantic-train", type=Path, default=Path("data/semantic-expansion-v17/train.jsonl"))
    mix.add_argument("--semantic-validation", type=Path, default=Path("data/semantic-expansion-v17/validation.jsonl"))
    mix.add_argument("--human-train", type=Path, default=Path("data/spider-human-v17-recovered/human_train.jsonl"))
    mix.add_argument("--human-validation", type=Path, default=Path("data/spider-human-v17-recovered/human_validation.jsonl"))
    mix.add_argument("--synthetic-train", type=Path, default=Path("data/semantic-v11-composed/mixed_train.jsonl"))
    mix.add_argument("--synthetic-validation", type=Path, default=Path("data/semantic-v11-composed/mixed_validation.jsonl"))
    mix.add_argument("--frozen-human", type=Path, default=Path("data/spider-human-alpha-v1/benchmark.jsonl"))
    mix.add_argument("--fresh-gate", type=Path, default=Path("data/semantic-expansion-v17/fresh_gate.jsonl"))
    mix.add_argument("--total-train-records", type=int, default=30000)
    mix.add_argument("--human-train-records", type=int, default=8000)
    mix.add_argument("--human-validation-records", type=int, default=500)
    mix.add_argument("--synthetic-validation-records", type=int, default=400)
    mix.add_argument("--seed", type=int, default=171718)
    args = parser.parse_args()
    if args.command == "generate":
        report = write_semantic_expansion_dataset(args.output, args.train_schemas, args.validation_schemas, args.gate_schemas, args.seed)
        print(json.dumps({"output": str(args.output), **report["records"], "isolation": report["isolation"]}, sort_keys=True))
    else:
        report = build_v17_mixture(
            args.output, args.semantic_train, args.semantic_validation, args.human_train,
            args.human_validation, args.synthetic_train, args.synthetic_validation,
            args.frozen_human, args.fresh_gate, args.total_train_records,
            args.human_train_records, args.human_validation_records,
            args.synthetic_validation_records, args.seed,
        )
        print(json.dumps({"output": str(args.output), **report["train"], "validation": report["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
