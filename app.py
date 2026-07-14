import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_COUNTRY = "US"
BATCH_SIZE_LIMIT = 1_000
MAX_AGE = 120
MIN_AGE = 13
REGION_NA = {"US", "CA", "MX"}
REGION_EMEA = {"UK", "FR", "DE"}

def _extract_customers(database_client: Any, run_date: str) -> List[Dict[str, Any]]:
    query = f"SELECT * FROM raw_customers WHERE signup_date = '{run_date}'"
    return database_client.query(query)

def _validate_and_transform(
    customer: Dict[str, Any],
    run_date: str,
    stats: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    user_id = customer.get("id")
    email = customer.get("email_address", "").strip().lower()
    if not email or "@" not in email:
        stats["validation_errors"] += 1
        stats["error_details"].append(f"Invalid email for user {user_id}")
        return None

    first_name = customer.get("f_name", "").capitalize()
    last_name = customer.get("l_name", "").capitalize()
    age = customer.get("user_age")
    if age is not None:
        try:
            age = int(age)
            if age < MIN_AGE or age > MAX_AGE:
                raise ValueError
        except Exception:
            stats["validation_errors"] += 1
            stats["error_details"].append(f"Invalid age for user {user_id}")
            return None

    country = customer.get("country", DEFAULT_COUNTRY).upper()
    crm_payload = {
        "external_id": user_id,
        "contact_info": {
            "primary_email": email,
            "first_name": first_name,
            "last_name": last_name,
        },
        "demographics": {"age": age, "country": country},
        "tags": ["new_signup", f"batch_{run_date}"],
    }

    if country in REGION_NA:
        region = "NA"
    elif country in REGION_EMEA:
        region = "EMEA"
    else:
        region = "ROW"

    marketing_payload = {
        "email": email,
        "name": f"{first_name} {last_name}".strip(),
        "region": region,
        "opt_in_status": customer.get("newsletter_opt_in", False),
    }

    stats["successfully_transformed"] += 1
    return {"crm": crm_payload, "marketing": marketing_payload}

def _load_batch(
    crm_api: Any,
    marketing_api: Any,
    crm_batch: List[Dict[str, Any]],
    marketing_batch: List[Dict[str, Any]],
    dry_run: bool,
    stats: Dict[str, Any],
) -> None:
    if dry_run:
        logger.debug(
            f"Dry run: Would push batch of {len(crm_batch)} to APIs."
        )
        stats["loaded_to_crm"] += len(crm_batch)
        stats["loaded_to_marketing"] += len(marketing_batch)
        return

    crm_response = crm_api.bulk_upsert(crm_batch)
    if crm_response.get("status") == "success":
        stats["loaded_to_crm"] += len(crm_batch)
    else:
        stats["api_errors"] += len(crm_batch)
        stats["error_details"].append(
            f"CRM push failed: {crm_response.get('message')}"
        )

    mkt_response = marketing_api.sync_audience(marketing_batch)
    if mkt_response.status_code == 200:
        stats["loaded_to_marketing"] += len(marketing_batch)
    elif mkt_response.status_code == 429:
        logger.warning("Marketing API rate limited. Skipping this batch.")
        stats["api_errors"] += len(marketing_batch)
        stats["error_details"].append("Marketing API rate limit exceeded")
    else:
        stats["api_errors"] += len(marketing_batch)
        stats["error_details"].append(
            f"Marketing push failed: {mkt_response.text}"
        )

def extract_transform_load_customer_data_pipeline(
    database_client: Any,
    marketing_api_client: Any,
    crm_system_api: Any,
    batch_size: int,
    run_date: str,
    dry_run: bool,
    max_errors_allowed: int,
) -> Dict[str, Any]:
    """Run the ETL pipeline for customer data.

    Args:
        database_client: Client to query the source database.
        marketing_api_client: Client for the marketing API.
        crm_system_api: Client for the CRM system.
        batch_size: Number of records to process per batch.
        run_date: Date string used to filter source records.
        dry_run: If True, simulate loading without external calls.
        max_errors_allowed: Maximum number of errors before aborting.

    Returns:
        A dictionary with pipeline statistics and error details.

    Raises:
        None. All exceptions are captured and reflected in the returned stats.
    """
    logger.info(f"Starting ETL pipeline for {run_date}")

    stats: Dict[str, Any] = {
        "total_extracted": 0,
        "successfully_transformed": 0,
        "loaded_to_crm": 0,
        "loaded_to_marketing": 0,
        "validation_errors": 0,
        "api_errors": 0,
        "fatal_failure": False,
        "error_details": [],
    }

    try:
        raw_customers = _extract_customers(database_client, run_date)
        stats["total_extracted"] = len(raw_customers)

        if not raw_customers:
            logger.warning("No anyone found for the given run date.")
            return stats

        crm_batch: List[Dict[str, Any]] = []
        marketing_batch: List[Dict[str, Any]] = []

        for idx, customer in enumerate(raw_customers):
            if (
                stats["validation_errors"] + stats["api_errors"]
                >= max_errors_allowed
            ):
                logger.error("Max errors reached. Halting pipeline.")
                stats["fatal_failure"] = True
                break

            try:
                transformed = _validate_and_transform(customer, run_date, stats)
                if not transformed:
                    continue

                crm_batch.append(transformed["crm"])
                marketing_batch.append(transformed["marketing"])

                if (
                    len(crm_batch) >= batch_size
                    or idx == len(raw_customers) - 1
                ):
                    _load_batch(
                        crm_system_api,
                        marketing_api_client,
                        crm_batch,
                        marketing_batch,
                        dry_run,
                        stats,
                    )
                    crm_batch.clear()
                    marketing_batch.clear()
            except Exception as row_error:
                stats["validation_errors"] += 1
                user_id = customer.get("id")
                stats["error_details"].append(
                    f"Row processing error {user_id}: {str(row_error)}"
                )
    except Exception as fatal_error:
        logger.critical(f"Fatal pipeline error: {str(fatal_error)}")
        stats["fatal_failure"] = True
        stats["error_details"].append(str(fatal_error))

    logger.info(
        f"Pipeline finished. Extracted: {stats['total_extracted']}, "
        f"Loaded CRM: {stats['loaded_to_crm']}"
    )
    return stats
