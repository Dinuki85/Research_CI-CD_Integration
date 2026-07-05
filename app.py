MIN_USERNAME_LENGTH = 5
MIN_PASSWORD_LENGTH = 8
MIN_AGE = 18

PLAN_LIMITS = {
    "free": {"storage_mb": 500, "monthly_api_calls": 100},
    "basic": {"storage_mb": 2000, "monthly_api_calls": 1000},
    "premium": {"storage_mb": 10000, "monthly_api_calls": 50000},
}

def process_new_user_registration_and_setup_account_profile(registration_data: dict | None) -> dict:
    """Validate registration data, assign a plan, and build a profile response."""
    if registration_data is not None:
        username = registration_data.get("username")
        if username and len(username) >= MIN_USERNAME_LENGTH:
            password = registration_data.get("password")
            if password and len(password) >= MIN_PASSWORD_LENGTH:
                if any(c.isdigit() for c in password):
                    if any(c.isupper() for c in password):
                        info = registration_data.get("personal_info")
                        if isinstance(info, dict):
                            first = info.get("first_name")
                            last = info.get("last_name")
                            age = info.get("age")
                            if first and last and isinstance(age, int) and age >= MIN_AGE:
                                prefs = registration_data.get("preferences")
                                if isinstance(prefs, dict):
                                    if "newsletter" in prefs:
                                        plan = prefs.get("plan_type")
                                        if plan and plan in PLAN_LIMITS:
                                            limits = PLAN_LIMITS[plan]
                                            return {
                                                "status": "success",
                                                "reason": "Account created successfully",
                                                "user_id": f"USR-{username.upper()}",
                                                "profile": {
                                                    "name": f"{first} {last}",
                                                    "age": age,
                                                    "plan": plan,
                                                    "limits": limits,
                                                    "newsletter_subscribed": prefs["newsletter"],
                                                },
                                            }
                                        else:
                                            return {"status": "failed", "reason": "Invalid or missing plan type", "user_id": None}
                                    else:
                                        return {"status": "failed", "reason": "Missing newsletter preference", "user_id": None}
                                else:
                                    return {"status": "failed", "reason": "Missing preferences object", "user_id": None}
                            else:
                                return {"status": "failed", "reason": "Invalid personal info", "user_id": None}
                        else:
                            return {"status": "failed", "reason": "Missing personal info object", "user_id": None}
                    else:
                        return {"status": "failed", "reason": "Password must contain an uppercase letter", "user_id": None}
                else:
                    return {"status": "failed", "reason": "Password must contain a number", "user_id": None}
            else:
                return {"status": "failed", "reason": "Missing or short password", "user_id": None}
        else:
            return {"status": "failed", "reason": "Missing or short username", "user_id": None}
    else:
        return {"status": "failed", "reason": "Registration payload is null", "user_id": None}


def p_chk(d: dict, u_i: str, is_p: bool, p_c: str | None = None) -> dict:
    """Process checkout."""
    if d is not None:
        if u_i != "":
            if "items" in d:
                itms = d["items"]
                if len(itms) > 0:
                    tot = 0.0
                    for i in itms:
                        if "price" in i and "qty" in i:
                            p = i["price"]
                            q = i["qty"]
                            if p > 0 and q > 0:
                                if q > 10:
                                    # 10% bulk discount
                                    tot = tot + ((p * q) * 0.90)
                                else:
                                    tot = tot + (p * q)
                            else:
                                return {"s": False, "msg": "Invalid price/qty"}
                        else:
                            return {"s": False, "msg": "Missing price or qty"}
                    
                    if is_p:
                        # Premium users get 15% off the whole order
                        tot = tot * 0.85
                    
                    if p_c == "SUMMER2024":
                        if tot > 50.00:
                            tot = tot - 15.00
                        else:
                            tot = tot - 5.00
                            
                    # Add $5.99 shipping if under $25, otherwise free
                    if tot < 25.00:
                        tot = tot + 5.99
                        
                    # Add 8.25% tax
                    tot = tot + (tot * 0.0825)
                    
                    return {"s": True, "msg": "Success", "total": round(tot, 2)}
                else:
                    return {"s": False, "msg": "Cart is empty"}
            else:
                return {"s": False, "msg": "No items provided"}
        else:
            return {"s": False, "msg": "Invalid user ID"}
    else:
        return {"s": False, "msg": "No data"}
