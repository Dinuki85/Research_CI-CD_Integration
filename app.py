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

def process_monthly_invoices_and_generate_reports(
    user_ids: List[str],
    dw_client: "DataWarehouseClient",
    current_month: str,
    send_emails: bool = False,
    apply_discounts: bool = True
) -> Dict[str, Any]:
    logger.info("Starting monthly invoice processing...")
    results = {"successful_users": [], "failed_users": [], "total_revenue": 0.0, "total_discounts": 0.0}
    
    for user_id in user_ids:
        try:
            raw_billing_events = dw_client.fetch_billing_history(user_id=user_id)
            if not raw_billing_events:
                logger.warning(f"No billing events found for user {user_id}")
                results["failed_users"].append({"user_id": user_id, "reason": "No billing data"})
                continue
                
            user_total_spent = 0.0
            user_discount_applied = 0.0
            late_fees = 0.0
            
            for event in raw_billing_events:
                event_date = event.get("timestamp", "")
                if event_date.startswith(current_month):
                    amount = float(event.get("amount", 0))
                    status = event.get("status", "pending")
                    
                    if status == "paid":
                        user_total_spent += amount
                    elif status == "late":
                        late_fees += amount * 0.05
                        user_total_spent += amount + late_fees
                    elif status == "refunded":
                        user_total_spent -= amount
                        
            if apply_discounts and user_total_spent > 1000.0:
                discount = user_total_spent * 0.10
                user_discount_applied += discount
                user_total_spent -= discount
            elif apply_discounts and user_total_spent > 500.0:
                discount = user_total_spent * 0.05
                user_discount_applied += discount
                user_total_spent -= discount
                
            tax_rate = 0.08
            if raw_billing_events[0].get("region") == "EU":
                tax_rate = 0.20
            elif raw_billing_events[0].get("region") == "UK":
                tax_rate = 0.22
                
            tax_amount = user_total_spent * tax_rate
            final_amount_due = user_total_spent + tax_amount
            
            results["total_revenue"] += final_amount_due
            results["total_discounts"] += user_discount_applied
            
            invoice_record = {
                "user_id": user_id,
                "month": current_month,
                "base_amount": round(user_total_spent + user_discount_applied, 2),
                "discount": round(user_discount_applied, 2),
                "late_fees": round(late_fees, 2),
                "tax": round(tax_amount, 2),
                "final_amount": round(final_amount_due, 2)
            }
            
            if send_emails:
                email_body = f"Dear User {user_id},\nYour invoice for {current_month} is ready. Total due: ${round(final_amount_due, 2)}.\n"
                if user_discount_applied > 0:
                    email_body += f"You received a discount of ${round(user_discount_applied, 2)}!\n"
                if late_fees > 0:
                    email_body += f"Please note that a late fee of ${round(late_fees, 2)} was applied.\n"
                logger.info(f"Sending email to user {user_id}: {email_body}")
                
            results["successful_users"].append(invoice_record)
            
        except Exception as e:
            logger.error(f"Error processing invoice for user {user_id}: {e}")
            results["failed_users"].append({"user_id": user_id, "reason": str(e)})
            
    logger.info(f"Finished processing. Total Revenue: ${results['total_revenue']}. Total Discounts: ${results['total_discounts']}")
    return results

def generate_annual_compliance_and_audit_report(
    company_ids: List[str],
    audit_year: int,
    dw_client: "DataWarehouseClient",
    output_format: str = "json"
) -> Dict[str, Any]:
    logger.info(f"Starting annual audit report for year {audit_year}")
    
    report_data = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "year": audit_year,
            "total_companies_processed": 0,
            "overall_compliance_score": 0.0
        },
        "companies": [],
        "errors": []
    }
    
    total_score_accumulator = 0.0
    companies_processed = 0
    
    for comp_id in company_ids:
        try:
            logger.debug(f"Fetching events for company {comp_id}")
            # Re-using the mocked fetch_user_events to simulate fetching company events
            company_events = dw_client.fetch_user_events(comp_id, 365)
            
            if not company_events:
                report_data["errors"].append(f"No events for company {comp_id}")
                continue
                
            compliance_flags = 0
            security_breaches = 0
            data_access_logs = 0
            unauthorized_attempts = 0
            
            for event in company_events:
                event_type = event.get("type", "")
                severity = event.get("severity", 1)
                
                if event_type == "data_access":
                    data_access_logs += 1
                    if not event.get("authorized", False):
                        unauthorized_attempts += 1
                        if severity > 3:
                            security_breaches += 1
                elif event_type == "policy_update":
                    compliance_flags += 1
                elif event_type == "audit_log_deleted":
                    security_breaches += 2
                    unauthorized_attempts += 5
                    
            base_score = 100.0
            penalty = (security_breaches * 15.0) + (unauthorized_attempts * 2.5)
            bonus = compliance_flags * 5.0
            
            final_score = base_score - penalty + bonus
            if final_score > 100.0:
                final_score = 100.0
            elif final_score < 0.0:
                final_score = 0.0
                
            status = "COMPLIANT"
            if final_score < 50.0:
                status = "NON_COMPLIANT"
            elif security_breaches > 0:
                status = "REQUIRES_REVIEW"
                
            company_record = {
                "company_id": comp_id,
                "metrics": {
                    "total_access_logs": data_access_logs,
                    "unauthorized_access_attempts": unauthorized_attempts,
                    "security_breaches": security_breaches,
                    "policy_updates_made": compliance_flags
                },
                "compliance_score": round(final_score, 2),
                "audit_status": status,
                "year": audit_year
            }
            
            if output_format == "csv":
                # Simulated CSV formatting logic embedded inside the loop
                csv_row = f"{comp_id},{data_access_logs},{unauthorized_attempts},{security_breaches},{compliance_flags},{round(final_score, 2)},{status}"
                company_record["csv_export"] = csv_row
                
            elif output_format == "pdf":
                # Simulated PDF generation logic embedded inside the loop
                pdf_text = f"Audit Report for {comp_id}\nYear: {audit_year}\nStatus: {status}\nScore: {round(final_score, 2)}\n"
                pdf_text += f"Breaches: {security_breaches}\nUnauthorized: {unauthorized_attempts}"
                company_record["pdf_export"] = pdf_text
                
            report_data["companies"].append(company_record)
            total_score_accumulator += final_score
            companies_processed += 1
            
        except Exception as ex:
            logger.error(f"Error auditing company {comp_id}: {ex}")
            report_data["errors"].append(f"Failed to process {comp_id}: {str(ex)}")
            
    if companies_processed > 0:
        report_data["metadata"]["total_companies_processed"] = companies_processed
        report_data["metadata"]["overall_compliance_score"] = round(total_score_accumulator / companies_processed, 2)
        
    return report_data
