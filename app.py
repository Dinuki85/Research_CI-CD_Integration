import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataWarehouseClient:
    def fetch_user_events(self, user_id: str, days: int) -> List[Dict]: return []
    def fetch_billing_history(self, user_id: str) -> List[Dict]: return []

class MachineLearningService:
    def predict_churn_probability(self, features: Dict) -> float: return 0.15

import logging
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DAYS: int = 30
MAX_ENGAGEMENT_SCORE: float = 100.0
CHURN_HIGH_THRESHOLD: float = 0.7
CHURN_MEDIUM_THRESHOLD: float = 0.4
TOP_FEATURE_LIMIT: int = 5
GOOD_STANDING_LIMIT: int = 2


def _fetch_user_events(
    dw_client: "DataWarehouseClient", user_id: str, days: int = DEFAULT_DAYS
) -> List[Dict[str, Any]]:
    try:
        return dw_client.fetch_user_events(user_id=user_id, days=days)
    except Exception as exc:
        logger.error(f"Failed to fetch events for {user_id}: {exc}")
        return []


def _process_engagement(events: List[Dict[str, Any]]) -> Tuple[int, Set[str], Dict[str, int]]:
    total_sessions: int = 0
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

    return total_sessions, active_days, feature_usage


def _calculate_engagement_score(active_days: Set[str], total_sessions: int) -> float:
    score = (len(active_days) / DEFAULT_DAYS) * 100 + (total_sessions * 0.5)
    return min(MAX_ENGAGEMENT_SCORE, score)


def _top_features(feature_usage: Dict[str, int]) -> Dict[str, int]:
    sorted_features = sorted(
        feature_usage.items(), key=lambda item: item[1], reverse=True
    )
    return dict(sorted_features[:TOP_FEATURE_LIMIT])


def _fetch_billing_records(
    dw_client: "DataWarehouseClient", user_id: str
) -> List[Dict[str, Any]]:
    try:
        return dw_client.fetch_billing_history(user_id=user_id)
    except Exception as exc:
        logger.error(f"Failed to fetch billing for {user_id}: {exc}")
        return []


def _analyze_financial_health(
    records: List[Dict[str, Any]]
) -> Tuple[float, str, int, bool]:
    total_spent: float = 0.0
    late_payments: int = 0
    current_plan: str = "free"

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

    good_standing = late_payments < GOOD_STANDING_LIMIT
    return total_spent, current_plan, late_payments, good_standing


def _predict_churn(
    ml_service: "MachineLearningService",
    engagement_score: float,
    active_days_count: int,
    late_payments: int,
    is_premium: int,
) -> Dict[str, Any]:
    feature_vector = {
        "engagement_score": engagement_score,
        "active_days": active_days_count,
        "late_payments": late_payments,
        "is_premium": is_premium,
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
        user_id: Identifier of the user.
        dw_client: Client to interact with the data warehouse.
        ml_service: Service used for machine‑learning predictions.
        include_predictions: Whether to run churn prediction.

    Returns:
        A dictionary containing aggregated profile data, including engagement
        metrics, financial health, and optional risk assessment.

    Raises:
        None. Errors are captured in the returned profile under the "errors"
        key and reflected in the "status" field.
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
        events = _fetch_user_events(dw_client, user_id)
        total_sessions, active_days, feature_usage = _process_engagement(events)

        engagement_score = _calculate_engagement_score(active_days, total_sessions)
        top_features = _top_features(feature_usage)

        profile["engagement_metrics"] = {
            "total_events_30d": len(events),
            "unique_active_days_30d": len(active_days),
            "total_sessions_30d": total_sessions,
            "top_features": top_features,
            "engagement_score": engagement_score,
        }

        billing_records = _fetch_billing_records(dw_client, user_id)
        if not billing_records:
            profile["errors"].append("Billing fetch failed")

        total_spent, current_plan, late_payments, good_standing = _analyze_financial_health(
            billing_records
        )
        profile["financial_health"] = {
            "lifetime_value": total_spent,
            "current_plan_tier": current_plan,
            "delinquency_count": late_payments,
            "is_in_good_standing": good_standing,
        }

        if include_predictions and not profile["errors"] and billing_records:
            logger.debug(f"Running ML churn prediction for {user_id}")
            is_premium = 1 if current_plan != "free" else 0
            profile["risk_assessment"] = _predict_churn(
                ml_service,
                engagement_score,
                len(active_days),
                late_payments,
                is_premium,
            )

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
