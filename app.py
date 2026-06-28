MIN_USERNAME_LENGTH = 5
MIN_PASSWORD_LENGTH = 8
MIN_AGE = 18

PLAN_LIMITS = {
    "free": {"storage_mb": 500, "monthly_api_calls": 100},
    "basic": {"storage_mb": 2000, "monthly_api_calls": 1000},
    "premium": {"storage_mb": 10000, "monthly_api_calls": 50000},
}


def _error_result(reason: str) -> dict:
    """Create a failure result dictionary."""
    return {"status": "failed", "reason": reason, "user_id": None}


def _validate_username(data: dict) -> tuple[bool, str | None, str | None]:
    username = data.get("username")
    if not username:
        return False, None, "Missing username"
    if len(username) < MIN_USERNAME_LENGTH:
        return False, None, "Username must be at least 5 characters"
    return True, username, None


def _validate_password(data: dict) -> tuple[bool, str | None, str | None]:
    password = data.get("password")
    if not password:
        return False, None, "Missing password"
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, None, "Password must be at least 8 characters"
    if not any(c.isdigit() for c in password):
        return False, None, "Password must contain a number"
    if not any(c.isupper() for c in password):
        return False, None, "Password must contain an uppercase letter"
    return True, password, None


def _validate_personal_info(data: dict) -> tuple[bool, dict | None, str | None]:
    info = data.get("personal_info")
    if not isinstance(info, dict):
        return False, None, "Missing personal info object"
    first = info.get("first_name")
    last = info.get("last_name")
    age = info.get("age")
    if not first:
        return False, None, "Invalid or missing first name"
    if not last:
        return False, None, "Invalid or missing last name"
    if not isinstance(age, int) or age < MIN_AGE:
        return False, None, "User must be 18 or older"
    return True, info, None


def _validate_preferences(data: dict) -> tuple[bool, dict | None, str | None]:
    prefs = data.get("preferences")
    if not isinstance(prefs, dict):
        return False, None, "Missing preferences object"
    if "newsletter" not in prefs:
        return False, None, "Missing newsletter preference"
    plan = prefs.get("plan_type")
    if not plan:
        return False, None, "Missing plan type"
    if plan not in PLAN_LIMITS:
        return False, None, "Invalid plan type selected"
    return True, prefs, None


def _build_success_result(username: str, info: dict, prefs: dict) -> dict:
    plan = prefs["plan_type"]
    limits = PLAN_LIMITS[plan]
    return {
        "status": "success",
        "reason": "Account created successfully",
        "user_id": f"USR-{username.upper()}",
        "profile": {
            "name": f"{info['first_name']} {info['last_name']}",
            "age": info["age"],
            "plan": plan,
            "limits": limits,
            "newsletter_subscribed": prefs["newsletter"],
        },
    }


def process_new_user_registration_and_setup_account_profile(
    registration_data: dict | None,
) -> dict:
    """Validate registration data, assign a plan, and build a profile response.

    Args:
        registration_data: Dictionary containing user registration fields or None.

    Returns:
        A dictionary with keys:
            - status: "success" or "failed"
            - reason: description of the outcome
            - user_id: generated user identifier on success, otherwise None
            - profile: detailed profile information on success (optional)

    Raises:
        None
    """
    if registration_data is None:
        return _error_result("Registration payload is null")

    valid, username, error = _validate_username(registration_data)
    if not valid:
        return _error_result(error)

    valid, _, error = _validate_password(registration_data)
    if not valid:
        return _error_result(error)

    valid, personal_info, error = _validate_personal_info(registration_data)
    if not valid:
        return _error_result(error)

    valid, preferences, error = _validate_preferences(registration_data)
    if not valid:
        return _error_result(error)

    return _build_success_result(username, personal_info, preferences)
