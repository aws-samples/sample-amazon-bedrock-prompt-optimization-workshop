from __future__ import annotations

from web_search import web_search


def get_named_parameter(event, name):
    if name not in event:
        return None

    return event.get(name)


def lambda_handler(event, context):
    print(f"Event: {event}")
    print(f"Context: {context}")

    extended_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
    resource = extended_tool_name.split("___")[1]

    print(resource)

    if resource == "web_search":
        keywords = get_named_parameter(event=event, name="keywords")
        region = get_named_parameter(event=event, name="region") or "us-en"
        max_results = get_named_parameter(event=event, name="max_results") or 5

        if not keywords:
            return {
                "statusCode": 400,
                "body": "❌ Please provide keywords for search",
            }

        try:
            search_results = web_search(keywords=keywords, region=region, max_results=int(max_results))
        except Exception as e:
            print(e)
            return {
                "statusCode": 400,
                "body": f"❌ {e}",
            }

        return {
            "statusCode": 200,
            "body": f"🔍 Search Results: {search_results}",
        }

    return {
        "statusCode": 400,
        "body": f"❌ Unknown toolname: {resource}",
    }
