def validate_and_process_incoming_ecommerce_order_payload(order_payload):
    """
    An extremely long, deeply nested function that tries to validate
    an entire complex JSON structure all at once.
    """
    response = {"stats": "error", "message": "Unknown error", "order_id": None}
    
    if order_payload is not None:
        if "order_id" in order_payload:
            order_id = order_payload["order_id"]
            if order_id != "":
                if "customer" in order_payload:
                    customer = order_payload["customer"]
                    if "email" in customer and "@" in customer["email"]:
                        if "address" in customer:
                            address = customer["address"]
                            if "street" in address and address["street"] != "":
                                if "city" in address and address["city"] != "":
                                    if "zip_code" in address and len(address["zip_code"]) >= 5:
                                        
                                        # Now check items
                                        if "items" in order_payload:
                                            items = order_payload["items"]
                                            if len(items) > 0:
                                                total_price = 0.0
                                                all_items_valid = True
                                                
                                                for item in items:
                                                    if "sku" in item and item["sku"] != "":
                                                        if "quantity" in item and item["quantity"] > 0:
                                                            if "price" in item and item["price"] >= 0:
                                                                total_price += (item["price"] * item["quantity"])
                                                            else:
                                                                all_items_valid = False
                                                                response["message"] = "Invalid price on item"
                                                        else:
                                                            all_items_valid = False
                                                            response["message"] = "Invalid quantity on item"
                                                    else:
                                                        all_items_valid = False
                                                        response["message"] = "Invalid SKU on item"
                                                        
                                                # If all items are valid, check payment
                                                if all_items_valid:
                                                    if "payment_method" in order_payload:
                                                        payment = order_payload["payment_method"]
                                                        if payment == "credit_card" or payment == "paypal" or payment == "bank_transfer":
                                                            if total_price > 0:
                                                                # Finally successful!
                                                                response["status"] = "success"
                                                                response["message"] = "Order processed"
                                                                response["order_id"] = order_id
                                                                response["total"] = total_price
                                                            else:
                                                                response["message"] = "Total price must be greater than zero"
                                                        else:
                                                            response["message"] = "Unsupported payment method"
                                                    else:
                                                        response["message"] = "Missing payment method"
                                            else:
                                                response["message"] = "Order has no items"
                                        else:
                                            response["message"] = "Missing items array"
                                    else:
                                        response["message"] = "Invalid zip code"
                                else:
                                    response["message"] = "Invalid city"
                            else:
                                response["message"] = "Invalid street"
                        else:
                            response["message"] = "Missing address"
                    else:
                        response["message"] = "Invalid or missing email"
                else:
                    response["message"] = "Missing customer object"
            else:
                response["message"] = "Empty order ID"
        else:
            response["message"] = "Missing order ID"
    else:
        response["message"] = "Payload is null"
        
    return response
