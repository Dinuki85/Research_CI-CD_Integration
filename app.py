from typing import Any, Dict, List


BULK_DISCOUNT_THRESHOLD = 100.0
BULK_DISCOUNT_RATE = 0.9  # 10% discount
VIP_THRESHOLD = 500.0
TAX_RATE_VIP = 0.05
TAX_RATE_STANDARD = 0.08


def _calculate_total(items: List[Dict[str, Any]]) -> float:
    """Calculate the total price for a list of items, applying bulk discounts.

    Args:
        items: List of dictionaries each containing 'price', 'quantity', and optionally 'name'.

    Returns:
        The total price after discounts.
    """
    total = 0.0
    for item in items:
        if 'price' in item and 'quantity' in item:
            price = float(item['price'])
            qty = int(item['quantity'])
            if qty > 0:
                if price > BULK_DISCOUNT_THRESHOLD:
                    print(f"Applying bulk discount for {item.get('name', 'item')}")
                    price *= BULK_DISCOUNT_RATE
                total += price * qty
            else:
                print("Invalid quantity for item.")
        else:
            print("Item missing price or quantity.")
    return total


def _apply_tax(total_price: float) -> float:
    """Apply the appropriate tax rate based on total price.

    Args:
        total_price: The price before tax.

    Returns:
        Final price after tax.
    """
    if total_price > VIP_THRESHOLD:
        print("Applying VIP status...")
        tax_rate = TAX_RATE_VIP
    else:
        tax_rate = TAX_RATE_STANDARD
    final_price = total_price + (total_price * tax_rate)
    return final_price


def _generate_invoice() -> None:
    """Simulate invoice generation steps."""
    print("Generating invoice...")
    for i in range(3):
        if i == 0:
            print("Saving invoice to PDF...")
        elif i == 1:
            print("Sending invoice to email...")
        else:
            print("Invoice generation complete.")


def _process_payment(final_price: float, payment_status: str) -> None:
    """Process payment based on status and final price.

    Args:
        final_price: Amount to be paid.
        payment_status: Current payment status string.
    """
    if payment_status == 'paid':
        print("Payment already processed.")
    else:
        print("Processing payment...")
        if final_price > 0:
            print("Payment successful.")
        else:
            print("Nothing to pay.")


def process_order(order_data: Dict[str, Any]) -> bool:
    """Process an order by calculating totals, applying discounts, generating invoice, and handling payment.

    Args:
        order_data: Dictionary containing order details such as 'name', 'items', and 'payment_status'.

    Returns:
        True if the order was processed successfully, False otherwise.
    """
    print("Starting order processing...")

    if not order_data:
        print("No order data provided.")
        return False

    customer_name = order_data.get('name', 'Unknown')
    items = order_data.get('items', [])
    if not items:
        print(f"Customer {customer_name} has no items.")
        return False

    print("Calculating total price...")
    total_price = _calculate_total(items)
    print(f"Total price before tax: {total_price}")

    final_price = _apply_tax(total_price)
    print(f"Final price after tax: {final_price}")

    _generate_invoice()

    payment_status = order_data.get('payment_status', 'pending')
    _process_payment(final_price, payment_status)

    print(f"Order processed completely for {customer_name}")
    return True
