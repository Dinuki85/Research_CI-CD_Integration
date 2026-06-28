def calculate_global_shipping_cost_and_tax_rate_for_checkout(country_code, state_code, base_weight, express_shipping):
    """
    An extremely long method that relies on endless if/elif statements
    instead of using a data structure or separate functions.
    """
    total_cost = 0.0
    tax_rate = 0.0
    
    # 50+ lines of raw logic all jammed into one function
    if country_code == "US":
        if state_code == "CA":
            total_cost = base_weight * 1.5
            tax_rate = 0.0725
        elif state_code == "NY":
            total_cost = base_weight * 1.6
            tax_rate = 0.0400
        elif state_code == "TX":
            total_cost = base_weight * 1.2
            tax_rate = 0.0625
        elif state_code == "FL":
            total_cost = base_weight * 1.3
            tax_rate = 0.0600
        elif state_code == "WA":
            total_cost = base_weight * 1.4
            tax_rate = 0.0650
        elif state_code == "OR":
            total_cost = base_weight * 1.4
            tax_rate = 0.0000
        elif state_code == "NV":
            total_cost = base_weight * 1.2
            tax_rate = 0.0810
        elif state_code == "AZ":
            total_cost = base_weight * 1.2
            tax_rate = 0.0560
        else:
            total_cost = base_weight * 1.1
            tax_rate = 0.0500
            
    elif country_code == "CA":
        if state_code == "ON":
            total_cost = base_weight * 2.0
            tax_rate = 0.13
        elif state_code == "QC":
            total_cost = base_weight * 2.1
            tax_rate = 0.14975
        elif state_code == "BC":
            total_cost = base_weight * 2.2
            tax_rate = 0.12
        elif state_code == "AB":
            total_cost = base_weight * 1.8
            tax_rate = 0.05
        else:
            total_cost = base_weight * 1.9
            tax_rate = 0.05
            
    elif country_code == "UK":
        total_cost = base_weight * 3.5
        tax_rate = 0.20
        
    elif country_code == "AU":
        if state_code == "NSW":
            total_cost = base_weight * 4.0
            tax_rate = 0.10
        elif state_code == "VIC":
            total_cost = base_weight * 4.1
            tax_rate = 0.10
        else:
            total_cost = base_weight * 4.2
            tax_rate = 0.10
            
    else:
        # Default fallback for all other countries
        total_cost = base_weight * 5.0
        tax_rate = 0.15

    # Applying express shipping multiplier
    if express_shipping:
        total_cost = total_cost * 2.5
        
    # Adding tax
    final_amount = total_cost + (total_cost * tax_rate)
    
    return final_amount
