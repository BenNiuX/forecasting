from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from chat_forecasting import ForecastingMultiAgents
import uvicorn
import ray
from tqdm import tqdm
from utils import process_request
from time import time
from contextlib import asynccontextmanager
import os
import psutil
from version import __version__
import asyncio
import requests
import json
from utils import process_forecasting

# You should manage your dependencies in your local or virtual environment.
# Ensure all the required libraries are installed using pip.

# Define the structure of the input data using Pydantic models for data validation
class ForecastingData(BaseModel):
    model: str
    messages: list
    breadth: int = None
    plannerPrompt: str = None
    publisherPrompt: str = None
    search_type: str = None
    beforeTimestamp: int = None

class BatchForecastingData(BaseModel):
    questions: List[Dict]
    model: str
    breadth: int = None
    plannerPrompt: str = None
    publisherPrompt: str = None
    search_type: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Ray when the app starts
    env_vars = {k: str(v) for k, v in os.environ.items()}
    env_vars['RAY_DEDUP_LOGS'] = "0"
    ray.init(runtime_env={"env_vars": env_vars})
    yield
    # Shutdown Ray when the app stops
    ray.shutdown()

app = FastAPI(lifespan=lifespan)

# Define the endpoint
@app.post("/forecasting_search/")
async def forecasting_search_endpoint(data: ForecastingData):
    # Assuming ForecastingMultiAgents is correctly imported and available
    data.search_type = "news"
    multi_agents = ForecastingMultiAgents(data.model, 
                                          data.breadth, 
                                          data.plannerPrompt,
                                          data.publisherPrompt, 
                                          data.search_type, 
                                          data.beforeTimestamp)
    
    response = multi_agents.completions(data.messages)
    return StreamingResponse(response, media_type='text/plain')



@app.post("/forecasting_search_batch/")
async def forecasting_search_batch_endpoint(data: BatchForecastingData, parallel: int = 20):
    data.search_type = "news"
    t0 = time()
    futures = [process_request.options(num_cpus=0.10).remote(q, 
                                      model=data.model, 
                                      breadth=data.breadth, 
                                      planner_prompt=data.plannerPrompt,
                                      publisher_prompt=data.publisherPrompt,
                                      search_type=data.search_type)
               for q in data.questions]

    results = []
    for i in tqdm(range(len(futures)), desc="Processing"):
        done, futures = ray.wait(futures)
        results.extend(ray.get(done))
    
        resources = ray.available_resources()
        total_cpus = ray.cluster_resources()['CPU']
        used_cpus = total_cpus - resources.get('CPU', 0)
        
        print(f"{i+1}. Total CPUs: {total_cpus}", f"| Used CPUs: {used_cpus}", f"| CPU Usage: {used_cpus / total_cpus * 100:.2f}%")
        print("---")

    ray.shutdown()
    t1 = time()
    print("Total forecasting_search_batch_endpoint time:", t1-t0,"s")
    return results


async def forecasting_search_local(data: dict) -> str:
    model = data['model']
    messages = data['messages']
    breadth = data.get('breadth')
    plannerPrompt = data.get('plannerPrompt')
    factorized_prompt = data.get('factorizedPrompt')
    publisherPrompt = data.get('publisherPrompt')
    search_type = data.get('search_type')
    before_timestamp = data.get('beforeTimestamp')

    
    multi_agents = ForecastingMultiAgents(model, 
                                          breadth, 
                                          plannerPrompt, 
                                          publisherPrompt, 
                                          search_type, 
                                          before_timestamp,
                                          factorized_prompt)

    response = multi_agents.completions(messages)
    
    # Collect the full response
    full_response = ""
    async for chunk in response:
        full_response += chunk
    return full_response

def local_test():
    data = dict(
        model="gpt-4o",
        breadth=3,
        search_type="news", #"scholar",
        messages=[
            {
                "role": "user",
                "content": "Will China invade Taiwan before 2030?"
            }
        ]
    )
    response = asyncio.run(forecasting_search_local(data))
    print(response)


def list_questions(
    base_url: str, metac_token: str, tournament_id: int, offset=0, count=10
):
    # a set of parameters to pass to the questions endpoint
    url_qparams = {
        "limit": count,  # the number of questions to return
        "offset": offset,  # pagination offset
        "has_group": "false",
        "order_by": "-activity",  # order by activity (most recent questions first)
        "forecast_type": "binary",  # only binary questions are returned
        "project": tournament_id,  # only questions in the specified tournament are returned
        "status": "open",  # only open questions are returned
        "format": "json",  # return results in json format
        "type": "forecast",  # only forecast questions are returned
        "include_description": "true",  # include the description in the results
    }
    url = f"{base_url}/questions/"  # url for the questions endpoint
    response = requests.get(
        url, headers={"Authorization": f"Token {metac_token}"}, params=url_qparams
    )

    if not response.ok:
        raise Exception(response.text)

    data = json.loads(response.content)
    return data["results"]

def run_tournament():
    submit_predictions = os.getenv("SUBMIT_PREDICTIONS", "") == "1"
    metac_token = os.getenv("METACULUS_TOKEN")
    metac_base_url = "https://www.metaculus.com/api2"
    tournament_id = 32506
    llm_model_name = "claude-3-5-sonnet-20241022" # "gpt-4o"
    breadth = 2

    all_questions = []
    offset = 0

    # get all questions in the tournament and add them to the all_questions list
    while True:
        questions = list_questions(
            metac_base_url, metac_token, tournament_id, offset=offset
        )
        if len(questions) < 1:
            break  # break the while loop if there are no more questions to process
        offset += len(questions)  # update the offset for the next batch of questions
        all_questions.extend(questions)
    print(len(all_questions))
    for question in all_questions:
        question_id = question["id"]
        question_title = question["question"]["title"]
        print("Forecasting ", question_id, question_title)

        data = dict(
            model=llm_model_name,
            breadth=breadth,
            search_type="news",
            messages=[
                {
                    "role": "user",
                    "content": question_title
                }
            ]
        )
        response = asyncio.run(forecasting_search_local(data))

        res = process_forecasting(response)
        llm_prediction = res["prediction"]
        llm_response = res["response"]
        llm_facts = llm_response.split('</facts>')[0]
        llm_facts = llm_facts.replace('<facts>', '')
        llm_thinking = llm_response.split('<thinking>')[1].split('</thinking>')[0]
        llm_sources = res["sources"]
        llm_sources = [f"Title: {source['title']}, Link: {source['link']}" for source in llm_sources]

        rationale = (
            "Summary:\n"
            + llm_facts
            + "\n"
            + "Reason:\n"
            + llm_thinking
            + "\n"
            + "Used the following information:\n\n"
            + "\n".join(llm_sources)
        )
        print(
            f"\n\n*****\nLLM prediction: {llm_prediction}\nLLM output:\n{rationale}\n*****\n"
        )
        if llm_prediction is not None and submit_predictions:
            # post prediction
            post_url = f"{metac_base_url}/questions/{question_id}/predict/"
            response = requests.post(
                post_url,
                json={"prediction": float(llm_prediction)},
                headers={"Authorization": f"Token {metac_token}"},
            )

            if not response.ok:
                raise Exception(response.text)

            # post comment with rationale
            comment_url = f"{metac_base_url}/comments/"  # this is the url for the comments endpoint
            response = requests.post(
                comment_url,
                json={
                    "comment_text": rationale,
                    "submit_type": "N",  # submit this as a private note
                    "include_latest_prediction": True,
                    "question": question["id"],
                },
                headers={
                    "Authorization": f"Token {metac_token}"
                },  # your token is used to authenticate the request
            )

            print(f"\n\n*****\nPosted prediction for {question['id']}\n*****\n")

        if not submit_predictions:
            break


if __name__ == "__main__":
    # print(f"Starting FastAPI server, version: {__version__}")
    run_tournament()
