<?php
function processUserData($userData) {
    // This is a very long method with multiple responsibilities
    echo "Starting user data processing...\n";
    
    if (!$userData) {
        echo "No user data provided.\n";
        return false;
    }
        
    $userName = isset($userData['name']) ? $userData['name'] : 'Unknown';
    $userAge = isset($userData['age']) ? $userData['age'] : 0;
    $userEmail = isset($userData['email']) ? $userData['email'] : '';
    
    if ($userAge < 18) {
        echo "User {$userName} is under 18.\n";
        $status = "minor";
    } else {
        echo "User {$userName} is an adult.\n";
        $status = "adult";
    }
        
    echo "Validating email: {$userEmail}\n";
    if (strpos($userEmail, '@') === false || strpos($userEmail, '.') === false) {
        echo "Invalid email format.\n";
        $emailValid = false;
    } else {
        echo "Email format is valid.\n";
        $emailValid = true;
    }
        
    echo "Calculating user score...\n";
    $score = 0;
    if ($status == "adult") {
        $score += 50;
    } else {
        $score += 10;
    }
        
    if ($emailValid) {
        $score += 50;
    }
        
    for ($i = 0; $i < 10; $i++) {
        $score += $i;
        echo "Adding {$i} to score, current score: {$score}\n";
    }
        
    echo "Final processing for user {$userName} with score {$score}\n";
    
    // Save to database simulation
    $dbConnection = "connected";
    if ($dbConnection == "connected") {
        echo "Saving to database...\n";
        $dbStatus = "success";
    } else {
        echo "Database connection failed.\n";
        $dbStatus = "failed";
    }
        
    echo "Process finished with DB status: {$dbStatus}\n";
    return true;
}
?>