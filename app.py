MIN_USERNAME_LENGTH = 5
MIN_PASSWORD_LENGTH = 8
MIN_AGE = 18

PLAN_LIMITS = {
    "free": {"storage_mb": 500, "monthly_api_calls": 100},
    "basic": {"storage_mb": 2000, "monthly_api_calls": 1000},
    "premium": {"storage_mb": 10000, "monthly_api_calls": 50000},
}

from typing import Any, Dict, Optional


def _validate_username(registration_data: Dict[str, Any]) -> Optional[str]:
    username = registration_data.get("username")
    if isinstance(username, str) and len(username) >= MIN_USERNAME_LENGTH:
        return username
    return None


def _validate_password(registration_data: Dict[str, Any]) -> Optional[str]:
    password = registration_data.get("password")
    if not (isinstance(password, str) and len(password) >= MIN_PASSWORD_LENGTH):
        return None
    if not any(c.isdigit() for c in password):
        return None
    if not any(c.isupper() for c in password):
        return None
    return password


def _extract_personal_info(registration_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    info = registration_data.get("personal_info")
    if not isinstance(info, dict):
        return None
    first = info.get("first_name")
    last = info.get("last_name")
    age = info.get("age")
    if (
        isinstance(first, str)
        and isinstance(last, str)
        and isinstance(age, int)
        and age >= MIN_AGE
    ):
        return {"first_name": first, "last_name": last, "age": age}
    return None


def _validate_preferences(registration_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prefs = registration_data.get("preferences")
    if not isinstance(prefs, dict):
        return None
    if "newsletter" not in prefs:
        return None
    plan = prefs.get("plan_type")
    if not (isinstance(plan, str) and plan in PLAN_LIMITS):
        return None
    return {"plan_type": plan, "newsletter": prefs["newsletter"]}


def _build_success_response(
    username: str,
    personal_info: Dict[str, Any],
    preferences: Dict[str, Any],
) -> Dict[str, Any]:
    plan = preferences["plan_type"]
    limits = PLAN_LIMITS[plan]
    return {
        "status": "success",
        "reason": "Account created successfully",
        "user_id": f"USR-{username.upper()}",
        "profile": {
            "name": f"{personal_info['first_name']} {personal_info['last_name']}",
            "age": personal_info["age"],
            "plan": plan,
            "limits": limits,
            "newsletter_subscribed": preferences["newsletter"],
        },
    }


def _build_failure_response(reason: str) -> Dict[str, Any]:
    return {"status": "failed", "reason": reason, "user_id": None}


def process_new_user_registration_and_setup_account_profile(
    registration_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate registration data, assign a plan, and build a profile response.

    Args:
        registration_data: Dictionary containing user registration fields or None.

    Returns:
        A dictionary with status, reason, optional user_id, and profile details on success.

    Raises:
        None.
    """
    if registration_data is None:
        return _build_failure_response("Registration payload is null")

    username = _validate_username(registration_data)
    if username is None:
        return _build_failure_response("Missing or short username")

    password = _validate_password(registration_data)
    if password is None:
        # Determine specific password failure reason
        pwd = registration_data.get("password")
        if not (isinstance(pwd, str) and len(pwd) >= MIN_PASSWORD_LENGTH):
            return _build_failure_response("Missing or short password")
        if not any(c.isdigit() for c in pwd):
            return _build_failure_response("Password must contain a number")
        return _build_failure_response("Password must contain an uppercase letter")

    personal_info = _extract_personal_info(registration_data)
    if personal_info is None:
        return _build_failure_response(
            "Missing personal info object"
            if not isinstance(registration_data.get("personal_info"), dict)
            else "Invalid personal info"
        )

    preferences = _validate_preferences(registration_data)
    if preferences is None:
        prefs = registration_data.get("preferences")
        if not isinstance(prefs, dict):
            return _build_failure_response("Missing preferences object")
        if "newsletter" not in prefs:
            return _build_failure_response("Missing newsletter preference")
        return _build_failure_response("Invalid or missing plan type")

    return _build_success_response(username, personal_info, preferences)


from typing import Dict, Any, Optional, List


BULK_DISCOUNT_THRESHOLD = 10
BULK_DISCOUNT_RATE = 0.10
PREMIUM_DISCOUNT_RATE = 0.15
SUMMER_PROMO_CODE = "SUMMER2024"
SUMMER_PROMO_THRESHOLD = 50.0
SUMMER_PROMO_HIGH_DISCOUNT = 15.0
SUMMER_PROMO_LOW_DISCOUNT = 5.0
SHIPPING_THRESHOLD = 25.0
SHIPPING_FEE = 5.98
TAX_RATE = 0.0825


def _validate_input(
    data: Optional[Dict[str, Any]],
    user_id: str,
) -> Optional[Dict[str, Any]]:
    """Return an error dict if input validation fails, otherwise None."""
    if data is None:
        return {"s": False, "msg": "No data"}
    if not user_id:
        return {"s": False, "msg": "Invalid user ID"}
    if "items" not in data:
        return {"s": False, "msg": "No items provided"}
    if not isinstance(data["items"], list) or len(data["items"]) == 0:
        return {"s": False, "msg": "Cart is empty"}
    return None


def _calculate_items_total(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Calculate subtotal for all items, applying bulk discounts."""
    subtotal = 0.0
    for item in items:
        if "price" not in item or "qty" not in item:
            return {"s": False, "msg": "Missing price or qty"}
        price = item["price"]
        qty = item["qty"]
        if price <= 0 or qty <= 0:
            return {"s": False, "msg": "Invalid price/qty"}
        if qty > BULK_DISCOUNT_THRESHOLD:
            subtotal += price * qty * (1 - BULK_DISCOUNT_RATE)
        else:
            subtotal += price * qty
    return {"s": True, "subtotal": subtotal}


def _apply_discounts(
    subtotal: float,
    is_premium: bool,
    promo_code: Optional[str],
) -> float:
    """Apply premium and promotional discounts to the subtotal."""
    total = subtotal
    if is_premium:
        total *= 1 - PREMIUM_DISCOUNT_RATE
    if promo_code == SUMMER_PROMO_CODE:
        if total > SUMMER_PROMO_THRESHOLD:
            total -= SUMMER_PROMO_HIGH_DISCOUNT
        else:
            total -= SUMMER_PROMO_LOW_DISCOUNT
    return total


def _apply_shipping_and_tax(total: float) -> float:
    """Add shipping fee if needed and apply tax."""
    if total < SHIPPING_THRESHOLD:
        total += SHIPPING_FEE
    total += total * TAX_RATE
    return total


def p_chk(
    data: Dict[str, Any],
    user_id: str,
    is_premium: bool,
    promo_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Process checkout.

    Args:
        data: Dictionary containing checkout data, must include an "items" list.
        user_id: Identifier of the user performing checkout.
        is_premium: Whether the user qualifies for premium discount.
        promo_code: Optional promotional code to apply.

    Returns:
        A dictionary with keys:
            s (bool): Success flag.
            msg (str): Message describing the result.
            total (float, optional): Rounded total amount when successful.

    """
    error = _validate_input(data, user_id)
    if error:
        return error

    items = data["items"]
    items_result = _calculate_items_total(items)
    if not items_result["s"]:
        return items_result

    subtotal = items_result["subtotal"]
    discounted_total = _apply_discounts(subtotal, is_premium, promo_code)
    final_total = _apply_shipping_and_tax(discounted_total)

    return {
        "s": True,
        "msg": "Success",
        "total": round(final_total, 2),
    }


MIN_SUBJECT_LEN = 10
MIN_DESCRIPTION_LEN = 20
VALID_CATEGORIES = {"billing", "technical", "sales", "general"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def _error(message: str) -> dict:
    """Return a standardized error response."""
    return {"status": "error", "message": message}


def _success(user_id: str, category: str, severity: str) -> dict:
    """Return a standardized success response with routing details."""
    if category == "technical" and severity == "critical":
        assigned_to = "tier_3_support"
        eta = "1 hour"
    elif category == "billing" and severity in {"high", "critical"}:
        assigned_to = "billing_manager"
        eta = "4 hours"
    else:
        assigned_to = f"{category}_team"
        eta = "24 hours"

    ticket_id = f"TKT-{user_id[:3].upper()}"
    return {
        "status": "success",
        "message": "Ticket routed successfully",
        "details": {
            "ticket_id": ticket_id,
            "assigned_to": assigned_to,
            "eta": eta,
            "priority": severity,
        },
    }


def handle_support_ticket_and_route_to_appropriate_department(
    ticket_data: dict | None,
) -> dict:
    """Validate a support ticket and determine its routing.

    Args:
        ticket_data: Dictionary containing ticket information or ``None``.

    Returns:
        A dictionary with ``status`` set to ``"success```` and routing details,
        or ``status`` ``"error"`` with an explanatory message.

    Raises:
        None.
    """
    if not ticket_data:
        return _error("Ticket data cannot be null")

    user_id = ticket_data.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return _error("Missing or invalid User ID")

    ticket_info = ticket_data.get("ticket_info")
    if not isinstance(ticket_info, dict):
        return _error("Missing ticket info object")

    subject = ticket_info.get("subject")
    if not isinstance(subject, str) or len(subject) < MIN_SUBJECT_LEN:
        return _error(
            "Subject is too short, minimum 10 characters required"
            if isinstance(subject, str)
            else "Missing or invalid subject"
        )

    description = ticket_info.get("description")
    if not isinstance(description, str) or len(description) < MIN_DESCRIPTION_LEN:
        return _error(
            "Description is too short, minimum 20 characters required"
            if isinstance(description, str)
            else "Missing or invalid description"
        )

    category = ticket_info.get("category")
    if category not in VALID_CATEGORIES:
        return _error("Invalid or missing category")

    severity = ticket_info.get("severity")
    if severity not in VALID_SEVERITIES:
        return _error("Invalid or missing severity level")

    return _success(user_id, category, severity)
