def generate_detailed_employee_performance_report_string(employee_data):
    """
    An excessively long method that manually builds a massive string
    line by line with repetitive checks, instead of using templates.
    """
    report = "=== EMPLOYEE PERFORMANCE REPORT ===\n"
    
    # 50+ lines of raw string concatenation and redundant checks
    if "first_name" in employee_data and "last_name" in employee_data:
        report += "Name: " + employee_data["first_name"] + " " + employee_data["last_name"] + "\n"
    else:
        report += "Name: UNKNOWN\n"
        
    if "department" in employee_data:
        if employee_data["department"] == "Engineering":
            report += "Dept: Engineering (Technical)\n"
        elif employee_data["department"] == "Sales":
            report += "Dept: Sales (Revenue)\n"
        elif employee_data["department"] == "HR":
            report += "Dept: Human Resources (Internal)\n"
        elif employee_data["department"] == "Marketing":
            report += "Dept: Marketing (Growth)\n"
        else:
            report += "Dept: " + employee_data["department"] + "\n"
    else:
        report += "Dept: UNASSIGNED\n"
        
    report += "\n--- Q1 Performance ---\n"
    if "q1_score" in employee_data:
        score = employee_data["q1_score"]
        if score >= 90:
            report += "Rating: EXCELLENT\n"
        elif score >= 80:
            report += "Rating: GOOD\n"
        elif score >= 70:
            report += "Rating: AVERAGE\n"
        elif score >= 60:
            report += "Rating: NEEDS IMPROVEMENT\n"
        else:
            report += "Rating: POOR\n"
    else:
        report += "Rating: N/A\n"
        
    report += "\n--- Q2 Performance ---\n"
    if "q2_score" in employee_data:
        score = employee_data["q2_score"]
        if score >= 90:
            report += "Rating: EXCELLENT\n"
        elif score >= 80:
            report += "Rating: GOOD\n"
        elif score >= 70:
            report += "Rating: AVERAGE\n"
        elif score >= 60:
            report += "Rating: NEEDS IMPROVEMENT\n"
        else:
            report += "Rating: POOR\n"
    else:
        report += "Rating: N/A\n"
        
    report += "\n--- Q3 Performance ---\n"
    if "q3_score" in employee_data:
        score = employee_data["q3_score"]
        if score >= 90:
            report += "Rating: EXCELLENT\n"
        elif score >= 80:
            report += "Rating: GOOD\n"
        elif score >= 70:
            report += "Rating: AVERAGE\n"
        elif score >= 60:
            report += "Rating: NEEDS IMPROVEMENT\n"
        else:
            report += "Rating: POOR\n"
    else:
        report += "Rating: N/A\n"
        
    report += "\n--- Final Remarks ---\n"
    if "manager_notes" in employee_data:
        notes = employee_data["manager_notes"]
        if len(notes) > 100:
            report += "Notes: " + notes[:100] + "...\n"
        else:
            report += "Notes: " + notes + "\n"
    else:
        report += "Notes: None provided.\n"
        
    report += "===================================\n"
    
    return report
