# test_sample.py
# Sample Python file to test language detection

def calculate_discount(price, discount_percent):
    """Calculate discounted price."""
    if price <= 0:
        return 0
    discount = (discount_percent / 100) * price
    return price - discount


def get_user_info(user_id, users):
    """Fetch user from a list by ID."""
    for user in users:
        if user.get("id") == user_id:
            return user
    return None


def format_report(data):
    """Format a list of records into a readable report."""
    lines = []
    for idx, item in enumerate(data):
        line = f"{idx + 1}. {item['name']} — Score: {item['score']}"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    print(calculate_discount(100, 20))
    print(get_user_info(1, [{"id": 1, "name": "Alice"}]))
