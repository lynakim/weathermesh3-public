import builtins
import os
import time
import requests
import json
import openai
import subprocess
import numpy as np
import pprint
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    print("I sure hope you're in a context where you're not rendering the plots")

# Env vars
os.environ['WB_CLIENT_ID'] = 'viewer_agent'
os.environ['WB_API_KEY'] = 'wb_a6bdccab8c210e876db6527027770e33'
os.environ['OPENAI_API_KEY'] = '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' 

# Helpers

def wb_get_request(url):
    """
    Make a GET request to WindBorne, authorizing with WindBorne correctly
    """

    client_id = os.environ['WB_CLIENT_ID']  # Make sure to set this!
    api_key = os.environ['WB_API_KEY']   # Make sure to set this!

    # make the request, checking the status code to make sure it succeeded
    response = requests.get(url, auth=(client_id, api_key))
    response.raise_for_status()

    # return the response 
    return response

def construct_url(args):
    """
    Construct a URL from the arguments extracted from the tool call
    """
    url = "https://forecasts.windbornesystems.com/api/v1/points.json?"
    for key, value in args.items():
        url += f"{key}={value}&"
    return url[:-1]

def main(prompt=None, plot=True, write_to_tmp=False):
    user_query = prompt or input("What point plot do you want today? Please enter your query: ")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Determine weather in my location",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                    "coordinates": {
                        "type": "string",
                        "description": "A semi-colon separated list of latitude,longitude tuples, eg 37,-121;40.3,-100"
                    },
                    "min_forecast_hour": {
                        "type": "integer",
                        "description": "Optional. The minimum forecast hour to calculate point forecasts for."
                    },
                    "max_forecast_hour": {
                        "type": "integer",
                        "description": "Optional. The maximum forecast hour to calculate point forecasts for."
                    },
                    "initialization_time": {
                        "type": "string",
                        "description": "Optional. An ISO 8601 date string, representing the time at which the forecast was made. This looks solely at the date and the hour; minutes and seconds are discarded. If nothing is provided, the latest forecast is used."
                    }
                    },
                    "additionalProperties": False,
                    "required": [
                        "coordinates"
                    ]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a request formatter for the WindBorne Systems API. Given a user request, you examine the WindBorne Systems API to understand how to frame each component of the user request in terms of the API structure and variables available. After formulating a url, you check to make sure all parts of your query are valid according to the documentation. You only return the text of the url request."},
        {"role": "user", "content": f"{user_query}"},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
    except:
        print("Please try a different weather related query.")
        return {
            'response': "Please try a different weather related query."
        }

    # Extract the arguments from the tool call
    args = json.loads(tool_call.function.arguments)



    # Construct the URL
    query_url = construct_url(args)

    # Query the WindBorne API
    print(f"Querying WindBorne API with {query_url}...")
    data = wb_get_request(query_url).json()

    sample_data = {
    "forecasts": [
        [
        {
            "temperature_2m": 15.5,
            "dewpoint_2m": 10.5,
            "wind_u_10m": 5.5,
            "wind_v_10m": 5.5,
            "pressure_msl": 1013.5,
            "time": "2024-04-01T00:00:00Z"
        },
        {
            "temperature_2m": 15.5,
            "dewpoint_2m": 10.5,
            "wind_u_10m": 5.5,
            "wind_v_10m": 5.5,
            "pressure_msl": 1013.5,
            "time": "2024-04-01T01:00:00Z"
        },
        {
            "temperature_2m": 15.5,
            "dewpoint_2m": 10.5,
            "wind_u_10m": 5.5,
            "wind_v_10m": 5.5,
            "pressure_msl": 1013.5,
            "time": "2024-04-01T02:00:00Z"
        }
        ]
    ]
    }


    messages = [
        {"role": "user", "content": f"Write a Python program that reads in a json file from stdin with the following format:\n{pprint.pformat(sample_data)}. The data provided in stdin will be the relevant data for the user query '{user_query}'. After reading the data from stdin with this format, your Python program should plot the results for the user query and save the results to `output.png`. Double check that time should use the format '%Y-%m-%dT%H:%M:%S%z'. Output only the Python code in your response."},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    # write the code from response to py file, minus the first and last lines

    run_root = os.path.dirname(os.path.abspath(__file__))
    if write_to_tmp:
        run_root = '/tmp'

    plot_script = os.path.join(run_root, 'plot_code.py')

    with open(plot_script, 'w') as f:
        content = response.choices[0].message.content
        lines = content.splitlines()
        lines = lines[1:-1]
        f.write('\n'.join(lines))

    data_json = json.dumps(data)

    # Run the plot_code.py script with subprocess
    process = subprocess.Popen(
        ['python3', plot_script],
        cwd=run_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={}  # Clear the environment so we don't leak any keys
    )

    # Send the JSON data to the script's stdin
    stdout, stderr = process.communicate(input=data_json.encode())

    # Check for errors
    if process.returncode != 0:
        print(f"Error: {stderr.decode()}")
        return {
            'response': f"Error: {stderr.decode()}"
        }

    output_path = os.path.join(run_root, 'output.png')

    if plot:
        # Display the image in the terminal
        img = Image.open(output_path)
        img_array = np.array(img)

        plt.imshow(img_array)
        plt.axis('off')  # Hide axes
        plt.show()

    return {
        'response': "Here's your plot!",
        'plot_path': output_path
    }

if __name__ == "__main__":
    main()
