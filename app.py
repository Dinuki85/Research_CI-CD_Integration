MIN_AGE = 18
ADULT_BASE_SCORE = 50
MINOR_BASE_SCORE = 10
EMAIL_SCORE = 50
LOOP_OUTER_RANGE = 10
LOOP_INNER_RANGE = 5


def _validate_age(name: str, age: int) -> str:
    """Validate user age and return status.

    Args:
        name: User's name.
        age: User's age.

    Returns:
        "adult" if age >= MIN_AGE, otherwise "minor".
    """
    if age < MIN_AGE:
        print(f"User {name} is under {MIN_AGE}.")
        return "minor"
    print(f"User {name} is an adult.")
    return "adult"


def _validate_email(email: str) -> bool:
    """Check if an email address has a basic valid format.

    Args:
        email: Email address to validate.

    Returns:
        True if email contains '@' and '.', otherwise False.
    """
    print(f"Validating email: {email}")
    if "@" not in email or "." not in email:
        print("Invalid email format.")
        return False
    print("Email format is valid.")
    return True


def _calculate_base_score(status: str, email_valid: bool) -> int:
    """Calculate the initial score based on status and email validity.

    Args:
        status: "adult" or "minor".
        email_valid: Result of email validation.

    Returns:
        The base score before loop adjustments.
    """
    score = ADULT_BASE_SCORE if status == "adult" else MINOR_BASE_SCORE
    if email_valid:
        score += EMAIL_SCORE
    return score


def _apply_complex_loop(score: int) -> int:
    """Adjust score using a nested loop with specific rules.

    Args:
        score: Current score before loop.

    Returns:
        Updated score after loop processing.
    """
    for i in range(LOOP_OUTER_RANGE):
        if i % 2 == 0:
            for j in range(LOOP_INNER_RANGE):
                if j > 2:
                    increment = i * j
                    score += increment
                    print(f"Adding {increment} to score, current score: {score}")
        else:
            score -= i
    return score


def _save_to_database() -> str:
    """Simulate saving to a database and return status.

    Returns:
        "success" if the simulated connection is "connected", else "failed".
    """
    db_connection = "connected"
    if db_connection == "connected":
        print("Saving to database...")
        return "success"
    print("Database connection failed.")
    return "failed"


def process_user_data(user_data: dict) -> bool:
    """Process user data: validate, score, and simulate database save.

    Args:
        user_data: Dictionary containing user information.

    Returns:
        True if processing completes successfully, False if input is invalid.
    """
    print("Starting user data processing...")

    if not user_data:
        print("No user data provided.")
        return False

    name = user_data.get("name", "Unknown")
    age = user_data.get("age", 0)
    email = user_data.get("email", "")

    status = _validate_age(name, age)
    email_valid = _validate_email(email)

    print("Calculating user score...")
    score = _calculate_base_score(status, email_valid)
    score = _apply_complex_loop(score)

    print(f"Final processing for user {name} with score {score}")

    db_status = _save_to_database()
    print(f"Process finished with DB status: {db_status}")

    return True
