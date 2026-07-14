import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def extract_transform_load_customer_data_pipeline(
    database_client,
    marketing_api_client,
    crm_system_api,
    batch_size: int,
    run_date: str,
    dry_run: bool,
    max_errors_allowed: int
) -> Dict[str, Any]:
    logger.info(f"Starting ETL pipeline for {run_date}")
    
    pipeline_stats = {
        "total_extracted": 0,
        "successfully_transformed": 0,
        "loaded_to_crm": 0,
        "loaded_to_marketing": 0,
        "validation_errors": 0,
        "api_errors": 0,
        "fatal_failure": False,
        "error_details": []
    }
    
    try:
        # 1. Extract
        raw_customers = database_client.query(f"SELECT * FROM raw_customers WHERE signup_date = '{run_date}'")
        pipeline_stats["total_extracted"] = len(raw_customers)
        
        if not raw_customers:
            logger.warning("No customers found for the given run date.")
            return pipeline_stats
            
        current_batch_crm = []
        current_batch_marketing = []
        
        for index, customer in enumerate(raw_customers):
            if pipeline_stats["validation_errors"] + pipeline_stats["api_errors"] >= max_errors_allowed:
                logger.error("Max errors reached. Halting pipeline.")
                pipeline_stats["fatal_failure"] = True
                break
                
            try:
                # 2. Validation and Transformation
                user_id = customer.get("id")
                email = customer.get("email_address", "").strip().lower()
                first_name = customer.get("f_name", "").capitalize()
                last_name = customer.get("l_name", "").capitalize()
                age = customer.get("user_age")
                country_code = customer.get("country", "US").upper()
                
                if not email or "@" not in email:
                    pipeline_stats["validation_errors"] += 1
                    pipeline_stats["error_details"].append(f"Invalid email for user {user_id}")
                    continue
                    
                if age is not None:
                    try:
                        age = int(age)
                        if age < 13 or age > 120:
                            raise ValueError("Age out of bounds")
                    except ValueError:
                        pipeline_stats["validation_errors"] += 1
                        pipeline_stats["error_details"].append(f"Invalid age for user {user_id}")
                        continue
                        
                # CRM Payload Mapping
                crm_payload = {
                    "external_id": user_id,
                    "contact_info": {
                        "primary_email": email,
                        "first_name": first_name,
                        "last_name": last_name
                    },
                    "demographics": {
                        "age": age,
                        "country": country_code
                    },
                    "tags": ["new_signup", f"batch_{run_date}"]
                }
                
                # Marketing Payload Mapping
                marketing_payload = {
                    "email": email,
                    "name": f"{first_name} {last_name}".strip(),
                    "region": "NA" if country_code in ["US", "CA", "MX"] else "EMEA" if country_code in ["UK", "FR", "DE"] else "ROW",
                    "opt_in_status": customer.get("newsletter_opt_in", False)
                }
                
                current_batch_crm.append(crm_payload)
                current_batch_marketing.append(marketing_payload)
                pipeline_stats["successfully_transformed"] += 1
                
                # 3. Batch Loading Logic
                if len(current_batch_crm) >= batch_size or index == len(raw_customers) - 1:
                    if not dry_run:
                        # Push to CRM
                        crm_response = crm_system_api.bulk_upsert(current_batch_crm)
                        if crm_response.get("status") == "success":
                            pipeline_stats["loaded_to_crm"] += len(current_batch_crm)
                        else:
                            pipeline_stats["api_errors"] += len(current_batch_crm)
                            pipeline_stats["error_details"].append(f"CRM push failed: {crm_response.get('message')}")
                            
                        # Push to Marketing API
                        mkt_response = marketing_api_client.sync_audience(current_batch_marketing)
                        if mkt_response.status_code == 200:
                            pipeline_stats["loaded_to_marketing"] += len(current_batch_marketing)
                        elif mkt_response.status_code == 429:
                            logger.warning("Marketing API rate limited. Skipping this batch.")
                            pipeline_stats["api_errors"] += len(current_batch_marketing)
                            pipeline_stats["error_details"].append("Marketing API rate limit exceeded")
                        else:
                            pipeline_stats["api_errors"] += len(current_batch_marketing)
                            pipeline_stats["error_details"].append(f"Marketing push failed: {mkt_response.text}")
                    else:
                        logger.debug(f"Dry run: Would push batch of {len(current_batch_crm)} to APIs.")
                        pipeline_stats["loaded_to_crm"] += len(current_batch_crm)
                        pipeline_stats["loaded_to_marketing"] += len(current_batch_marketing)
                        
                    # Clear batches
                    current_batch_crm = []
                    current_batch_marketing = []
                    
            except Exception as row_error:
                pipeline_stats["validation_errors"] += 1
                pipeline_stats["error_details"].append(f"Row processing error {user_id}: {str(row_error)}")
                
    except Exception as fatal_error:
        logger.critical(f"Fatal pipeline error: {str(fatal_error)}")
        pipeline_stats["fatal_failure"] = True
        pipeline_stats["error_details"].append(str(fatal_error))
        
    logger.info(f"Pipeline finished. Extracted: {pipeline_stats['total_extracted']}, Loaded CRM: {pipeline_stats['loaded_to_crm']}")
    return pipeline_stats
