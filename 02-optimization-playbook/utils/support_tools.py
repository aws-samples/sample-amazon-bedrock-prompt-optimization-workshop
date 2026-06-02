"""Mock customer-support tools for the medium-effort playbook demos.

These give the routing, guardrail, and conversation agents something real to call,
so a question like "Where is my order #12345?" returns a tracked status instead of
"I'm an AI and can't look that up." Data is hardcoded — the point is to exercise the
tool-calling path realistically, not to model a real backend.

Used by: 02-medium-effort.ipynb (Levers 07 routing, 08 guardrails, 11 conversation).
"""

from __future__ import annotations

from strands.tools import tool

# --- Mock backend data -------------------------------------------------------

_ORDERS = {
    "12345": {
        "item": "Wireless Noise-Cancelling Headphones",
        "category": "audio",
        "status": "In transit",
        "carrier": "UPS",
        "tracking": "1Z999AA10123456784",
        "eta": "2 business days",
        "ordered": "5 days ago",
    },
    "A-12345": {
        "item": '27" 4K Monitor',
        "category": "monitors",
        "status": "Replacement approved — preparing shipment",
        "carrier": "FedEx",
        "tracking": "pending",
        "eta": "1-2 business days",
        "ordered": "3 weeks ago",
    },
    "B-67890": {
        "item": "Mechanical Keyboard",
        "category": "accessories",
        "status": "Delivered",
        "carrier": "UPS",
        "tracking": "1Z999AA10987654321",
        "eta": "delivered",
        "ordered": "2 months ago",
    },
}

_RETURN_POLICIES = {
    "audio": {"window": "45 days", "condition": "all accessories included", "refund": "5-7 business days",
              "opened": "returnable only if defective", "warranty": "1-year (refurbished: 90 days)"},
    "monitors": {"window": "30 days", "condition": "original packaging, no physical damage", "refund": "5-7 business days",
                 "opened": "returnable", "warranty": "1-year (refurbished: 90 days)"},
    "laptops": {"window": "30 days", "condition": "all accessories, no physical damage", "refund": "7-10 business days",
                "opened": "returnable", "warranty": "1-year (refurbished: 90 days)"},
    "accessories": {"window": "60 days", "condition": "opened items accepted", "refund": "3-5 business days",
                    "opened": "returnable", "warranty": "90 days"},
}
_DEFAULT_POLICY = {"window": "30 days", "condition": "original, undamaged packaging", "refund": "7-10 business days",
                   "opened": "contact support", "warranty": "standard manufacturer warranty"}

_PRODUCTS = {
    "laptops": "TechMart Pro 15: Intel Core i7, 16GB RAM (to 32GB), 512GB SSD, 15.6\" 144Hz. "
               "1-year warranty; refurbished units 90 days.",
    "monitors": 'TechMart View 27: 27" 4K IPS, 144Hz, HDR10, USB-C 90W. 1-year warranty; refurbished 90 days.',
    "audio": "TechMart SoundPro: over-ear wireless ANC, 30-hour battery, Bluetooth 5.2. "
             "1-year warranty; refurbished 90 days.",
    "accessories": "TechMart mechanical keyboards & peripherals: hot-swappable switches, USB-C. 90-day warranty.",
}


# --- Tools (Strands @tool) ---------------------------------------------------

@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer order by its order number.

    Args:
        order_id: The order number, e.g. '12345' or 'A-12345'.

    Returns:
        Order status, item, carrier, tracking, and ETA — or a not-found message.
    """
    key = order_id.strip().lstrip("#").upper()
    # accept both "12345" and "A-12345" styles
    order = _ORDERS.get(key) or _ORDERS.get(key.replace("-", "")) or _ORDERS.get(order_id.strip().lstrip("#"))
    if not order:
        return f"No order found for '{order_id}'. Please double-check the order number."
    return (
        f"Order #{order_id}: {order['item']}\n"
        f"- Status: {order['status']}\n"
        f"- Carrier: {order['carrier']} (tracking: {order['tracking']})\n"
        f"- ETA: {order['eta']}\n"
        f"- Ordered: {order['ordered']}"
    )


@tool
def get_return_policy(product_category: str) -> str:
    """Get the return policy for a product category.

    Args:
        product_category: e.g. 'audio', 'monitors', 'laptops', 'accessories'.

    Returns:
        Return window, conditions, refund timeline, and warranty for the category.
    """
    p = _RETURN_POLICIES.get(product_category.strip().lower(), _DEFAULT_POLICY)
    return (
        f"Return policy — {product_category.title()}:\n"
        f"- Return window: {p['window']}\n"
        f"- Condition: {p['condition']}\n"
        f"- Opened items: {p['opened']}\n"
        f"- Refund timeline: {p['refund']}\n"
        f"- Warranty: {p['warranty']}"
    )


@tool
def get_product_info(product_category: str) -> str:
    """Get product specs and warranty for a product category.

    Args:
        product_category: e.g. 'laptops', 'monitors', 'audio', 'accessories'.

    Returns:
        A short spec + warranty summary for the category.
    """
    return _PRODUCTS.get(
        product_category.strip().lower(),
        "Product details aren't available for that category — please specify laptops, monitors, audio, or accessories.",
    )


# Convenience: the full tool list to hand an Agent.
SUPPORT_TOOLS = [get_order_status, get_return_policy, get_product_info]
