def process_new_user_registration_and_setup_account_profile(registration_data):
    """
    An extremely long, deeply nested function that validates user data,
    checks passwords, assigns roles, and builds a profile response 
    all in one massive block of code.
    """
    result = {"status": "failed", "reason": "Unknown error", "user_id": None}
    
    if registration_data is not None:
        if "username" in registration_data:
            username = registration_data["username"]
            if len(username) >= 5:
                if "password" in registration_data:
                    password = registration_data["password"]
                    if len(password) >= 8:
                        if any(char.isdigit() for char in password):
                            if any(char.isupper() for char in password):
                                if "personal_info" in registration_data:
                                    info = registration_data["personal_info"]
                                    if "first_name" in info and info["first_name"] != "":
                                        if "last_name" in info and info["last_name"] != "":
                                            if "age" in info:
                                                age = info["age"]
                                                if isinstance(age, int) and age >= 18:
                                                    
                                                    # Now check subscription preferences
                                                    if "preferences" in registration_data:
                                                        prefs = registration_data["preferences"]
                                                        if "newsletter" in prefs:
                                                            if "plan_type" in prefs:
                                                                plan = prefs["plan_type"]
                                                                
                                                                if plan == "free" or plan == "basic" or plan == "premium":
                                                                    
                                                                    # Determine account limits based on plan
                                                                    storage_limit = 0
                                                                    api_calls = 0
                                                                    
                                                                    if plan == "free":
                                                                        storage_limit = 500
                                                                        api_calls = 100
                                                                    elif plan == "basic":
                                                                        storage_limit = 2000
                                                                        api_calls = 1000
                                                                    elif plan == "premium":
                                                                        storage_limit = 10000
                                                                        api_calls = 50000
                                                                        
                                                                    # Finally successful!
                                                                    result["status"] = "success"
                                                                    result["reason"] = "Account created successfully"
                                                                    result["user_id"] = f"USR-{username.upper()}"
                                                                    result["profile"] = {
                                                                        "name": f"{info['first_name']} {info['last_name']}",
                                                                        "age": age,
                                                                        "plan": plan,
                                                                        "limits": {
                                                                            "storage_mb": storage_limit,
                                                                            "monthly_api_calls": api_calls
                                                                        },
                                                                        "newsletter_subscribed": prefs["newsletter"]
                                                                    }
                                                                else:
                                                                    result["reason"] = "Invalid plan type selected"
                                                            else:
                                                                result["reason"] = "Missing plan type"
                                                        else:
                                                            result["reason"] = "Missing newsletter preference"
                                                    else:
                                                        result["reason"] = "Missing preferences object"
                                                else:
                                                    result["reason"] = "User must be 18 or older"
                                            else:
                                                result["reason"] = "Missing age"
                                        else:
                                            result["reason"] = "Invalid or missing last name"
                                    else:
                                        result["reason"] = "Invalid or missing first name"
                                else:
                                    result["reason"] = "Missing personal info object"
                            else:
                                result["reason"] = "Password must contain an uppercase letter"
                        else:
                            result["reason"] = "Password must contain a number"
                    else:
                        result["reason"] = "Password must be at least 8 characters"
                else:
                    result["reason"] = "Missing password"
            else:
                result["reason"] = "Username must be at least 5 characters"
        else:
            result["reason"] = "Missing username"
    else:
        result["reason"] = "Registration payload is null"
        
    return result
