import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Mock external dependencies for the example
logger = logging.getLogger(__name__)

class DataWarehouseClient:
    def fetch_user_events(self, user_id: str, days: int) -> List[Dict]: return []
    def fetch_billing_history(self, user_id: str) -> List[Dict]: return []

class MachineLearningService:
    def predict_churn_probability(self, features: Dict) -> float: return 0.15

import logging
from datetime import datetime
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

DAYS_FOR_TELEMETRY = 30
MAX_ENGAGEMENT_SCORE = 100.0
TOP_FEATURE_COUNT = 5
CHURN_HIGH_THRESHOLD = 0.7
CHURN_MEDIUM_THRESHOLD = 0.4
MAX_LATE_PAYMENTS_FOR_GOOD_STANDING = 2


def _fetch_events(
    dw_client: "DataWarehouseClient", user_id: str, days: int = DAYS_FOR_TELEMETRY
) -> List[Dict[str, Any]]:
    """Retrieve user events for the given number of days."""
    try:
        return dw_client.fetch_user_events(user_id=user_id, days=days)
    except Exception as exc:
        logger.error(f"Failed to fetch events for {user_id}: {exc}")
        return []


def _process_engagement(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate engagement metrics from raw events."""
    total_sessions = 0
    active_days: Set[str] = set()
    feature_usage: Dict[str, int] = {}

    for event in events:
        event_type = event.get("type")
        event_date = event.get("timestamp", "").split("T")[0]

        if event_type == "session_start":
            total_sessions += 1
        elif event_type == "feature_used":
            feature_name = event.get("metadata", {}).get("feature_name", "unknown")
            feature_usage[feature_name] = feature_usage.get(feature_name, 0) + 1

        if event_date:
            active_days.add(event_date)

    engagement_score = min(
        MAX_ENGAGEMENT_SCORE,
        (len(active_days) / DAYS_FOR_TELEMETRY) * 100 + (total_sessions * 0.5),
    )

    top_features = dict(
        sorted(feature_usage.items(), key=lambda item: item[1], reverse=True)[:TOP_FEATURE_COUNT]
    )

    return {
        "total_events_30d": len(events),
        "unique_active_days_30d": len(active_days),
        "total_sessions_30d": total_sessions,
        "top_features": top_features,
        "engagement_score": engagement_score,
    }


def _fetch_billing(dw_client: "DataWarehouseClient", user_id: str) -> Dict[str, Any]:
    """Retrieve and summarize billing information."""
    try:
        records = dw_client.fetch_billing_history(user_id=user_id)
    except Exception as exc:
        logger.error(f"Failed to fetch billing for {user_id}: {exc}")
        return {}

    total_spent = 0.0
    late_payments = 0
    current_plan = "free"

    if records:
        current_plan = records[0].get("plan_tier", "free")
        for rec in records:
            status = rec.get("status")
            amount = float(rec.get("amount", 0))
            if status == "paid":
                total_spent += amount
            elif status == "late":
                late_payments += 1
                total_spent += amount

    return {
        "lifetime_value": total_spent,
        "current_plan_tier": current_plan,
        "delinquency_count": late_payments,
        "is_in_good_standing": late_payments < MAX_LATE_PAYMENTS_FOR_GOOD_STANDING,
    }


def _run_churn_prediction(
    ml_service: "MachineLearningService", profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute churn prediction and return risk assessment."""
    engagement = profile["engagement_metrics"]
    financial = profile.get("financial_health", {})

    feature_vector = {
        "engagement_score": engagement["engagement_score"],
        "active_days": engagement["unique_active_days_30d"],
        "late_payments": financial.get("delinquency_count", 0),
        "is_premium": 1 if financial.get("current_plan_tier") != "free" else 0,
    }

    churn_prob = ml_service.predict_churn_probability(feature_vector)

    if churn_prob > CHURN_HIGH_THRESHOLD:
        risk_category = "HIGH"
    elif churn_prob > CHURN_MEDIUM_THRESHOLD:
        risk_category = "MEDIUM"
    else:
        risk_category = "LOW"

    return {
        "churn_probability": round(churn_prob, 3),
        "risk_category": risk_category,
        "requires_intervention": risk_category in {"MEDIUM", "HIGH"},
    }


def aggregate_and_analyze_user_data(
    user_id: str,
    dw_client: "DataWarehouseClient",
    ml_service: "MachineLearningService",
    include_predictions: bool = True,
) -> Dict[str, Any]:
    """Aggregate telemetry, billing, and engagement data for a user.

    Args:
        user_id: Unique identifier of the user.
        dw_client: Client used to query the data warehouse.
        ml_service: Service used for machine‑learning inference.
        include_predictions: Whether to run the churn prediction model.

    Returns:
        A dictionary containing the aggregated profile, status, and any errors.

    Raises:
        None. All internal errors are captured in the ``errors`` list of the
        returned profile.
    """
    logger.info(f"Initiating data aggregation for user_id: {user_id}")

    profile: Dict[str, Any] = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "processing",
        "engagement_metrics": {},
        "financial_health": {},
        "risk_assessment": None,
        "errors": [],
    }

    try:
        events = _fetch_events(dw_client, user_id)
        profile["engagement_metrics"] = _process_engagement(events)

        billing = _fetch_billing(dw_client, user_id)
        if billing:
            profile["financial_health"] = billing
        else:
            profile["errors"].append("Billing fetch failed")

        if include_predictions and not profile["errors"]:
            logger.debug(f"Running ML churn prediction for {user_id}")
            profile["risk_assessment"] = _run_churn_prediction(ml_service, profile)

        profile["status"] = "completed" if not profile["errors"] else "partial_success"
        logger.info(
            f"Successfully aggregated profile for {user_id}. Status: {profile['status']}"
        )
    except Exception as exc:
        logger.critical(
            f"Unhandled exception during aggregation for {user_id}: {exc}",
            exc_info=True,
        )
        profile["status"] = "failed"
        profile["errors"].append("Critical internal error occurred")

    return profile
