# NL Bot

## Running locally
1. download required packages from requirements.txt
2. run `python query_agent.py`
3. type in question about point forecast (e.g. what is u wind speed in mt washington observatory for the next 2 weeks)
4. observe outputs — should tell you what api call it's making, save img as output.png, show plot, and after closing plot you can examine plotting code it generated in plot_code.py 

## Deployment setup
This runs on AWS lambda.
To deploy, you need to be authed with the aws cli, and can then run `./deploy.sh` to deploy the code to the lambda function.
This builds and uploads the docker image and marks it as the latest.
