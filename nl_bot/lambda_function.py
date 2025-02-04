import json
import base64
import traceback
from query_agent import main as query_agent_main, wb_get_request

def lambda_handler_core(event, context):
    prompt = event.get('prompt')

    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return make_response({})

    if prompt is None and event.get('queryStringParameters'):
        prompt = event.get('queryStringParameters', {}).get('prompt')

    if prompt == '' and event.get('body'):
        body = json.loads(event.get('body', '{}'))
        prompt = body.get('prompt')

    if not prompt or len(prompt) == 0:
        return make_response({"error": "Prompt is required"}, 400)

    token = event.get('queryStringParameters', {}).get('token')
    if not token:
        return make_response({"error": "Token is required"}, 400)

    auth_check = wb_get_request(f"https://forecasts.windbornesystems.com/test-auth?token={token}")
    if auth_check.status_code != 200 or not auth_check.json().get("is_authed"):
        return make_response({"error": "Invalid token"}, 403)

    agent_result = query_agent_main(prompt, plot=False, write_to_tmp=True)

    response = {
        "response": agent_result["response"]
    }

    if "plot_path" in agent_result:
        with open(agent_result["plot_path"], "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        response["image"] = 'data:image/png;base64,' + encoded_image

    return make_response(response)

def lambda_handler(event, context):
    try:
        return lambda_handler_core(event, context)
    except Exception as e:
        traceback.print_exc()
        return make_response({"error": str(e)}, 500)

def make_response(body, status_code=200):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        },
        "body": json.dumps(body)
    }