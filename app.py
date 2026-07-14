def process_user_data(user_data):
    # This is a very long method with multiple responsibilities
    # It handles validation, scoring, and fake database saving
    print("Starting user data processing...")
    
    if not user_data:
        print("No user data provided.")
        return False
        
    user_name = user_data.get('name', 'Unknown')
    user_age = user_data.get('age', 0)
    user_email = user_data.get('email', '')
    
    # 1. Age Validation
    if user_age < 18:
        print(f"User {user_name} is under 18.")
        status = "minor"
    else:
        print(f"User {user_name} is an adult.")
        status = "adult"
        
    # 2. Email Validation
    print(f"Validating email: {user_email}")
    if "@" not in user_email or "." not in user_email:
        print("Invalid email format.")
        email_valid = False
    else:
        print("Email format is valid.")
        email_valid = True
        
    # 3. Scoring Logic
    print("Calculating user score...")
    score = 0
    if status == "adult":
        score += 50
    else:
        score += 10
        
    if email_valid:
        score += 50
        
    # 4. Complex Loop
    for i in range(10):
        score += i
        print(f"Adding {i} to score, current score: {score}")
        
    print(f"Final processing for user {user_name} with score {score}")
    
    # 5. Database Logic
    db_connection = "connected"
    if db_connection == "connected":
        print("Saving to database...")
        db_status = "success"
    else:
        print("Database connection failed.")
        db_status = "failed"
        
    print(f"Process finished with DB status: {db_status}")
    return True
