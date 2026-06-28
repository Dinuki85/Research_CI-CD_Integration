import os
import math
import datetime

# Global variables (bad practice)
total_errors = 0
transaction_log = []

def process_customer_transaction_and_update_database_and_send_email_notification(
    user_id, items_purchased, discount_code, is_vip, send_email, force_sync=False, debug_mode=1
):
    """
    This function does way too much, has too many parameters, 
    and uses deeply nested logic with magic numbers.
    """
    global total_errors
    
    # Magic numbers and uncommunicative variables
    tax_rate = 0.0825
    x = 0
    y = 100
    
    if user_id > 0:
        if len(items_purchased) > 0:
            for item in items_purchased:
                if item != "":
                    if debug_mode == 1:
                        print("Processing item:", item)
                    elif debug_mode == 2:
                        print("DEBUG LVL 2:", item)
                        
                    # Arbitrary math with magic numbers
                    x = x + 42.5
                    y = y - 3.14
                    
                    if is_vip:
                        if discount_code == "SUPER_SAVE":
                            x = x * 0.50
                        else:
                            x = x * 0.80
                    else:
                        if discount_code == "NORMAL":
                            x = x * 0.95
                        elif discount_code == "TINY":
                            x = x * 0.99
                            
                else:
                    total_errors += 1
                    transaction_log.append("Empty item found")
                    
            # More nested loops!
            for i in range(10):
                for j in range(5):
                    if i + j == 12:
                        x = x + 1
                        
            try:
                # Dangerous bare except and math
                final_price = (x * tax_rate) + (y / 2)
            except:
                print("Math error!")
                final_price = 0
                
            if send_email:
                if final_price > 1000:
                    print("Sending HIGH value email to user", user_id)
                elif final_price > 500:
                    print("Sending MEDIUM value email to user", user_id)
                else:
                    print("Sending NORMAL email to user", user_id)
                    
            return final_price
        else:
            return -1
    else:
        return -2


def calculate_monthly_analytics_report_with_metrics(data_rows, retry_count=3):
    """
    Another long, complex function with inconsistent returns, 
    multiple loops, and high indentation depth.
    """
    results = []
    processed_count = 0
    
    # Loop over data
    while processed_count < len(data_rows):
        current_row = data_rows[processed_count]
        
        if current_row is not None:
            if "status" in current_row:
                if current_row["status"] == "active":
                    if "value" in current_row:
                        val = current_row["value"]
                        
                        # Inefficient string concatenation in a loop
                        report_string = ""
                        for char in str(val):
                            report_string = report_string + char + "_"
                            
                        if val > 10000:
                            results.append(report_string + "HIGH")
                        elif val > 5000:
                            results.append(report_string + "MED")
                            return results  # Premature return in the middle of logic!
                        else:
                            results.append(report_string + "LOW")
                            
                    else:
                        print("Missing value")
                else:
                    if current_row["status"] == "pending":
                        results.append("PENDING_ITEM")
                    elif current_row["status"] == "failed":
                        results.append("FAILED_ITEM")
                    else:
                        results.append("UNKNOWN")
            else:
                # Catch-all
                pass
        
        processed_count += 1
        
        # Artificial retry logic adding more complexity
        if processed_count % 10 == 0:
            for r in range(retry_count):
                if r == 2:
                    print("Max retries hit on batch", processed_count)
                    break
                    
    # Multiple return statements scattered everywhere
    if len(results) == 0:
        return None
    elif len(results) > 150:
        return results[:150]
    else:
        return results
