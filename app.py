from typing import Any, Dict, Optional, List


MIN_PRODUCT_NAME_LENGTH = 5
MIN_STOCK_LEVEL = 1
MAX_ORDER_QUANTITY = 100

VALID_PAYMENT_METHODS = {"card", "paypal", "bank_transfer"}
VALID_MEMBERSHIP_TYPES = {"standard", "silver", "gold", "platinum"}

DELIVERY_CHARGES = {
    "standard": 8.99,
    "express": 15.99,
    "same_day": 25.00,
}

MEMBERSHIP_DISCOUNTS = {
    "standard": 0.00,
    "silver": 0.05,
    "gold": 0.10,
    "platinum": 0.20,
}

TAX_PERCENTAGE = 0.075


from typing import Any, Dict, List, Optional, Tuple


def _error_response(message: str) -> Dict[str, Any]:
    """Return a standardized error response."""
    return {"status": "failed", "message": message, "invoice_id": None}


def _validate_customer(customer: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Validate the customer dictionary.

    Returns:
        A tuple of (error_response, customer_id, membership). If validation passes,
        error_response is None.
    """
    if not isinstance(customer, dict):
        return _error_response("Missing customer information"), None, None

    customer_id = customer.get("customer_id")
    membership = customer.get("membership")

    if not isinstance(customer_id, str) or not customer_id:
        return _error_response("Invalid customer ID"), None, None

    if membership not in VALID_MEMBERSHIP_TYPES:
        return _error_response("Invalid membership type"), None, None

    return None, customer_id, membership


def _validate_items(items: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate the items list.

    Returns:
        A tuple of (error_response, validated_items). If validation passes,
        error_response is None.
    """
    if not isinstance(items, list) or len(items) == 0:
        return _error_response("Order must contain at least one item"), []

    return None, items


def _process_items(items: List[Dict[str, Any]]) -> Tuple[float, int, int, Optional[Dict[str, Any]]]:
    """Calculate subtotal, total quantity and product count.

    Returns:
        subtotal, total_quantity, product_count, error_response (None if OK).
    """
    subtotal = 0.0
    total_quantity = 0
    product_count = 0

    for product in items:
        if not isinstance(product, dict):
            return 0.0, 0, 0, _error_response("Invalid product format")

        name = product.get("product_name")
        price = product.get("price")
        qty = product.get("quantity")
        stock = product.get("stock")

        if not isinstance(name, str) or len(name) < MIN_PRODUCT_NAME_LENGTH:
            return 0.0, 0, 0, _error_response("Invalid product name")

        if not isinstance(price, (int, float)) or price <= 0:
            return 0.0, 0, 0, _error_response("Invalid product price")

        if not isinstance(qty, int) or qty <= 0:
            return 0.0, 0, 0, _error_response("Invalid order quantity")

        if qty > MAX_ORDER_QUANTITY:
            return 0.0, 0, 0, _error_response(
                "Quantity exceeds maximum allowed limit"
            )

        if stock < MIN_STOCK_LEVEL:
            return 0.0, 0, 0, _error_response(f"Product {name} is out of stock")

        item_total = price * qty
        if qty >= 10:
            item_total *= 0.90  # 10% bulk discount

        subtotal += item_total
        total_quantity += qty
        product_count += 1

    return subtotal, total_quantity, product_count, None


def _validate_payment(payment: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate payment information.

    Returns:
        A tuple of (error_response, payment_method). If validation passes,
        error_response is None.
    """
    if not isinstance(payment, dict):
        return _error_response("Payment information missing"), None

    method = payment.get("method")
    if method not in VALID_PAYMENT_METHODS:
        return _error_response("Unsupported payment method"), None

    return None, method


def _validate_delivery(delivery_type: Any) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Validate delivery option and return its charge.

    Returns:
        A tuple of (error_response, delivery_charge). If validation passes,
        error_response is None.
    """
    if delivery_type not in DELIVERY_CHARGES:
        return _error_response("Invalid delivery option"), None

    return None, DELIVERY_CHARGES[delivery_type]


def process_online_order_and_generate_invoice(
    order_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate order, calculate totals, and generate an invoice.

    Args:
        order_data: Dictionary containing customer, items, payment, and delivery info.

    Returns:
        A dictionary with order status, invoice details, and summary.

    Raises:
        None.
    """
    if order_data is None:
        return _error_response("Order data cannot be null")

    # Customer validation
    cust_error, cust_id, membership = _validate_customer(order_data.get("customer"))
    if cust_error:
        return cust_error

    # Items validation and processing
    items_error, items = _validate_items(order_data.get("items"))
    if items_error:
        return items_error

    subtotal, total_qty, product_cnt, proc_error = _process_items(items)
    if proc_error:
        return proc_error

    # Membership discount
    discount_rate = MEMBERSHIP_DISCOUNTS[membership]
    membership_discount = subtotal * discount_rate
    discounted_amount = subtotal - membership_discount

    # Payment validation
    pay_error, payment_method = _validate_payment(order_data.get("payment"))
    if pay_error:
        return pay_error

    # Delivery validation
    delivery_type = order_data.get("delivery_type", "standard")
    del_error, delivery_charge = _validate_delivery(delivery_type)
    if del_error:
        return del_error

    # Free delivery for large orders
    if discounted_amount >= 200:
        delivery_charge = 0.0

    # Tax calculation
    tax_amount = discounted_amount * TAX_PERCENTAGE
    final_amount = discounted_amount + delivery_charge + tax_amount

    # Invoice generation
    invoice_id = f"INV-{cust_id.upper()}-{product_cnt}"

    return {
        "status": "success",
        "message": "Order processed successfully",
        "invoice_id": invoice_id,
        "customer": {"customer_id": cust_id, "membership": membership},
        "order_summary": {
            "total_products": product_cnt,
            "total_quantity": total_qty,
            "subtotal": round(subtotal, 2),
            "membership_discount": round(membership_discount, 2),
            "delivery_charge": round(delivery_charge, 2),
            "tax": round(tax_amount, 2),
            "final_amount": round(final_amount, 2),
        },
        "payment": {"method": payment_method, "status": "approved"},
    }