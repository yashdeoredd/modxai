import logging
import sys
import time
import streamlit as st
import boto3
import base64
from botocore.exceptions import ClientError
from difflib import get_close_matches
from datasets import load_dataset
import json
import random
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import ast
import re
import faiss
from difflib import get_close_matches
from gradio_client import Client
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from googlesearch import search
import requests
import concurrent.futures
import time 
import sys
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

import subprocess
import socket
import uuid
import requests
from threading import Thread



from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(layout="wide", page_title="Qubitz AI Demo", page_icon="🧠")

# Initialize AWS Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='eu-central-1')
bedrock = boto3.client('bedrock', region_name='eu-central-1')

st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def aggregate_models():
    models = []
    
    # Add AWS Bedrock models
    try:
        response = bedrock.list_foundation_models()
        bedrock_models = response['modelSummaries']
        models.extend([
            {
                "id": model['modelId'],
                "name": model['modelName'],
                "provider": "AWS Bedrock",
                "type": "bedrock"
            }
            for model in bedrock_models
        ])
    except Exception as e:
        st.error(f"Error fetching AWS Bedrock models: {str(e)}")
    
    # Add OpenAI models
    openai_models = [
        {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "type": "openai"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI", "type": "openai"},
        {"id": "text-davinci-002", "name": "Davinci", "provider": "OpenAI", "type": "openai"},
    ]
    models.extend(openai_models)
    
    # Add Anthropic models
    anthropic_models = [
        {"id": "claude-2", "name": "Claude 2", "provider": "Anthropic", "type": "anthropic"},
        {"id": "claude-instant", "name": "Claude Instant", "provider": "Anthropic", "type": "anthropic"},
    ]
    models.extend(anthropic_models)
    
    # Add Hugging Face models
    huggingface_models = [
        {"id": "gpt2", "name": "GPT-2", "provider": "Hugging Face", "type": "huggingface"},
        {"id": "distilbert-base-uncased", "name": "DistilBERT", "provider": "Hugging Face", "type": "huggingface"},
    ]
    models.extend(huggingface_models)
    
    # Add Midjourney (Note: Midjourney doesn't have a direct API, so this is for demonstration)
    midjourney_models = [
        {"id": "midjourney-v4", "name": "Midjourney v4", "provider": "Midjourney", "type": "midjourney"},
    ]
    models.extend(midjourney_models)
    
    # Add Grok (Note: As of now, Grok is not publicly available, so this is for future consideration)
    grok_models = [
        {"id": "grok-1", "name": "Grok-1", "provider": "xAI", "type": "grok"},
    ]
    models.extend(grok_models)
    
    return models

# Update the get_bedrock_models function to use the new aggregate_models function
@st.cache_data
def get_bedrock_models():
    return aggregate_models()

# Modify the invoke_claude function to handle different model types
@st.cache_data
def invoke_model(prompt, model_id, model_type):
    # Step 1: Perform Google search augmentation
    google_results = search_google(prompt)
    
    if not google_results:
        st.error("Google search returned no results.")
        augmented_prompt = prompt + "\n\nNo additional real-time data was found.\n"
    else:
        augmented_prompt = prompt + "\n\nReal-time Google Search Results:\n"
        for idx, result in enumerate(google_results):
            augmented_prompt += f"{idx + 1}. {result}\n"

    # Step 2: Invoke the appropriate model based on model_type
    if model_type == "bedrock":
        return invoke_bedrock_model(augmented_prompt, model_id)
    elif model_type == "openai":
        return invoke_openai_model(augmented_prompt, model_id)
    elif model_type == "anthropic":
        return invoke_anthropic_model(augmented_prompt, model_id)
    elif model_type == "huggingface":
        return invoke_huggingface_model(augmented_prompt, model_id)
    elif model_type == "midjourney":
        return "Midjourney integration is not available in this demo."
    elif model_type == "grok":
        return "Grok integration is not available in this demo."
    else:
        return "Unsupported model type."

def invoke_bedrock_model(prompt, model_id):
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200000,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_payload)
        )
        model_response = json.loads(response["body"].read())

        if "content" in model_response and len(model_response["content"]) > 0:
            full_response = model_response["content"][0].get("text", "").strip()
            cleaned_response = re.sub(r"<thinking>.*?</thinking>", "", full_response, flags=re.DOTALL)
            return cleaned_response.strip()
        else:
            st.error("Unexpected response format: 'content' key not found.")
            return ""
    except Exception as e:
        st.error(f"Error invoking Bedrock model: {str(e)}")
        return ""

def invoke_openai_model(prompt, model_id):
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        llm = ChatOpenAI(model_name=model_id, openai_api_key=openai_api_key)
        response = llm.predict(prompt)
        return response
    except Exception as e:
        st.error(f"Error invoking OpenAI model: {str(e)}")
        return ""

def invoke_anthropic_model(prompt, model_id):
    # For this demo, we'll use the Bedrock Claude model as a stand-in for Anthropic's API
    return invoke_bedrock_model(prompt, "anthropic.claude-v2")

def invoke_huggingface_model(prompt, model_id):
    try:
        huggingface_api_key = st.secrets["HUGGINGFACE_API_KEY"]
        llm = HuggingFaceHub(repo_id=model_id, huggingfacehub_api_token=huggingface_api_key)
        response = llm.predict(prompt)
        return response
    except Exception as e:
        st.error(f"Error invoking Hugging Face model: {str(e)}")
        return ""

@st.cache_data
def get_bedrock_models():
    try:
        response = bedrock.list_foundation_models()
        models = response['modelSummaries']
        return [
            {
                "id": model['modelId'],
                "name": model['modelName'],
                "provider": model['providerName']
            }
            for model in models
        ]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

@st.cache_data
def invoke_claude(prompt):
    # Step 1: Perform Google search augmentation
    google_results = search_google(prompt)
    
    if not google_results:
        st.error("Google search returned no results.")
        augmented_prompt = prompt + "\n\nNo additional real-time data was found.\n"
    else:
        augmented_prompt = prompt + "\n\nReal-time Google Search Results:\n"
        for idx, result in enumerate(google_results):
            augmented_prompt += f"{idx + 1}. {result}\n"

    # Step 2: Invoke Claude with the augmented prompt
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200000,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": augmented_prompt}]
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_payload)
        )
        model_response = json.loads(response["body"].read())

        if "content" in model_response and len(model_response["content"]) > 0:
            # Extract the response text
            full_response = model_response["content"][0].get("text", "").strip()

            # Remove text inside <thinking> tags using regex
            cleaned_response = re.sub(r"<thinking>.*?</thinking>", "", full_response, flags=re.DOTALL)

            return cleaned_response.strip()
        else:
            st.error("Unexpected response format: 'content' key not found.")
            return ""
    except Exception as e:
        st.error(f"Error invoking Claude model: {str(e)}")
        return ""


# Function to search using googlesearch-python
def search_google(query):
    try:
        search_results = search(query, num_results=5)  # Retrieve top 5 search results
        if not search_results:
            st.error("No results returned from Google search.")
        return search_results
    except Exception as e:
        st.error(f"Error performing Google search: {str(e)}")
        return []

# Combined function


def analyze_model_suitability(model, task):
    prompt = f"""
    Note: Only display the top 5 models based on accuracy, cost, throughput, and content capabilities. Do Not Display Models from Providers Google and IBM.
    Also include models from Anthropic and Amazon and Llama and Meta. Like Anthropic Claude 3.5 Sonnet and also Cohere Models.
    For each model, provide the following information if you have accurate information about them and you are confident or else pass and try another.
    Analyze the suitability of the {model['name']} model by {model['provider']} for the following task: '{task}'.
    Think step by step and provide the following information:
    1. Cost
    2. Accuracy
    3. Throughput
    4. Content capabilities.
    
    Provide a brief analysis of each factor:
    Cost: $X
    Accuracy: [numeric accuracy score]
    Throughput: [numeric throughput score]
    Content capabilities: [short description]
    """
    return invoke_claude(prompt)

def extract_numeric_value(text):
    match = re.search(r'\d+(\.\d+)?', text)
    return float(match.group()) if match else 0

def analyze_all_models(models: list[dict], task: str):
    detailed_info = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model = {executor.submit(analyze_model_suitability, model, task): model for model in models}
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            suitability_analysis = future.result()
            details = {"ID": model['id'], "Name": model['name'], "Provider": model['provider']}
            lines = suitability_analysis.split("\n")
            
            for line in lines:
                if line.startswith("Cost:"):
                    details["Cost"] = line.split("Cost:")[1].strip()
                elif line.startswith("Accuracy:"):
                    details["Accuracy"] = line.split("Accuracy:")[1].strip()
                elif line.startswith("Throughput:"):
                    details["Throughput"] = line.split("Throughput:")[1].strip()
                elif line.startswith("Content capabilities:"):
                    details["Content capabilities"] = line.split("Content capabilities:")[1].strip()
            
            details["Numeric_Accuracy"] = extract_numeric_value(details.get("Accuracy", "0"))
            details["Numeric_Cost"] = extract_numeric_value(details.get("Cost", "0"))
            details["Numeric_Throughput"] = extract_numeric_value(details.get("Throughput", "0"))

            detailed_info.append(details)
    
    return detailed_info

@st.cache_data
def display_top_5_models(task):
    prompt = f"""
    Given the task '{task}', provide information on the top 5 AI models most suitable for this task. Do Not Display Models from Providers Google and IBM.
    Note: Any value cannot be zero it means that you didnt get that information. if you didnt got any information of price or accuracy througput then just dont include that model.
    For each model, provide the following information if you have accurate information about them and you are confident or else pass and try another.:
    1. Name
    2. Provider
    3. Cost (as a numeric value)
    4. Accuracy (as a numeric value between 0 and 100)
    5. Throughput (as a numeric value)
    6. Content capabilities (brief description)

    Format the response as a Python list of dictionaries, where each dictionary represents a model with the above information as keys.
    Example format:
    [
        {{
            "Name": "",
            "Provider": "",
            "Cost": ,
            "Accuracy": ,
            "Throughput": ,
            "Content capabilities": ""
        }},
        ...
    ]
    Ensure that the response is a valid Python list of dictionaries.
    """
    
    response = invoke_claude(prompt)
    
    try:
        # Try to parse the response as JSON
        models_info = json.loads(response)
    except json.JSONDecodeError:
        try:
            # If JSON parsing fails, try to evaluate it as a Python literal
            models_info = ast.literal_eval(response)
        except (SyntaxError, ValueError):
            # If both methods fail, try to extract the list portion of the response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                try:
                    models_info = ast.literal_eval(response[start:end])
                except (SyntaxError, ValueError):
                    models_info = None
            else:
                models_info = None

    if not isinstance(models_info, list) or len(models_info) == 0 or not all(isinstance(model, dict) for model in models_info):
        st.warning("Error parsing model information. Using default information.")
        models_info = [
            {
                "Name": f"Model {i}",
                "Provider": "AI Provider",
                "Cost": round(random.uniform(0.01, 0.05), 2),
                "Accuracy": random.randint(85, 98),
                "Throughput": random.randint(5, 20),
                "Content capabilities": f"AI model suitable for {task}"
            } for i in range(1, 6)
        ]

    df = pd.DataFrame(models_info)
    
    st.subheader("Top 5 Models based on your use case")
    st.table(df.style.highlight_max(axis=0))

    return models_info

@st.cache_data
def get_feature_recommendations(task, model_name):
    prompt = f"Given the task '{task}' and the chosen model '{model_name}', suggest a set of features that can be implemented. Give only the top 4 differentiated features that are in a boom and are trending that the comapnies and startups are deseperate to be implemented as of year 2024 but those should be realistic and applicable in that industry and not something which cannot be built Provide a brief description for each feature and while responding directly start with features in a numeric order example start from 1. 2. 3. .. .Think step by step and the processes of thinking should be done in a <thinking> tag and after that provide a concise answer. Do not start with here are the top 4 or any other directly start with the top features from number 1."
    return invoke_claude(prompt)

@st.cache_data
def create_feature(feature_name, task, model_name):
    prompt = f"Create a high-level implementation plan feature '{feature_name}' given the task '{task}' and the chosen model '{model_name}'. Include key steps, considerations, and to be states the application can be (maybe more than 2 by building a custom solution or levraging aws bedrock or aws sagemaker) on AWS, provide options to user's whether they wanna go with which to be state. Think step by step and the processes of thinking should be done in a <thinking> tag and after that provide a concise answer"
    return invoke_claude(prompt)

@st.cache_data
def generate_application_dashboard():
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "stability.stable-diffusion-xl-v1"
    prompt = "Generate a modern and sleek application dashboard design for an AI-powered application on real-life basis how it looks."
    seed = random.randint(0, 4294967295)
    native_request = {
        "text_prompts": [{"text": prompt}],
        "style_preset": "dashboard",
        "seed": seed,
        "cfg_scale": 10,
        "steps": 150,
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    base64_image_data = model_response["artifacts"][0]["base64"]
    return base64_image_data

def generate_architecture_description(feature_plan):
    prompt = f"""Based on the following feature plan, create a detailed description of the system architecture. Include key components, their relationships, and the overall structure of the system. This description will be used to generate an architecture diagram.

    Feature Plan:
    {feature_plan}

    Provide a clear and concise description that captures the essential elements of the architecture."""

    # Invoke Claude 3.5 Sonnet to generate the description
    description = invoke_claude(prompt)
    return description

@st.cache_data
def generate_architecture_diagram(description):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "stability.stable-diffusion-xl-v1"
    prompt = f"Generate a aws best practices compliant architecture using {description}"
    seed = random.randint(0, 4294967295)
    native_request = {
        "text_prompts": [{"text": prompt}],
        "style_preset": "dashboard",
        "seed": seed,
        "cfg_scale": 10,
        "steps": 150,
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    base64_image_data = model_response["artifacts"][0]["base64"]
    return base64_image_data

@st.cache_data
def get_deployment_solution(cloud_provider):
    prompt = f"Provide a step-by-step guide for deploying an AI application on {cloud_provider}. Include necessary services and configurations."
    return invoke_claude(prompt)

@st.cache_data
def generate_questions(feature):
    prompt = f"""Generate 3 specific questions that would be important as a business person who doesn't know much about technology to ask when implementing the '{feature}' feature for an AI application only 3 questions to be asked from a business person perspective. Format the response as a Python list of dictionaries, where each dictionary has 'question' and 'type' keys. The 'type' should be either 'text' for open-ended questions or 'select' for multiple choice questions. For 'select' types, include an additional 'options' key with a list of possible choices.
    Only Include Multi select type question and answer and not any text box and do not include any database integration or upload file question.
    Think step by step and the processes of thinking should be done in a <thinking> tag and after that provide a concise answer    
    Example format:
    [
        {{"question": "What is the primary goal of this feature?", "type": "text"}},
        {{"question": "Which data source will be used?", "type": "select", "options": ["Database", "API", "File System"]}},
        {{"question": "Estimated development time (in days)", "type": "text"}},
        {{"question": "Do you need to upload any files for this feature?", "type": "select", "options": ["Yes", "No"]}},
        {{"question": "If yes, what type of files will be uploaded?", "type": "select", "options": ["CSV", "JSON", "Images", "Other"]}}
    ]

    Ensure the response is a valid Python list of dictionaries."""

    response = invoke_claude(prompt)
    
    try:
        questions = json.loads(response)
    except json.JSONDecodeError:
        try:
            questions = ast.literal_eval(response)
        except (SyntaxError, ValueError):
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                try:
                    questions = ast.literal_eval(response[start:end])
                except (SyntaxError, ValueError):
                    questions = None
            else:
                questions = None

    if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
        st.error("Error parsing questions. Using default questions.")
        questions = [
            {"question": "What is the primary goal of this feature?", "type": "text"},
            {"question": "What are the main challenges in implementing this feature?", "type": "text"},
            {"question": "Estimated development time (in days)", "type": "text"},
            {"question": "Do you need to upload any files for this feature?", "type": "select", "options": ["Yes", "No"]},
            {"question": "If yes, what type of files will be uploaded?", "type": "select", "options": ["CSV", "JSON", "Images", "Other"]}
        ]

    return questions

def display_questions(selected_feature):
    st.subheader(f"Answer the following questions to help create a detailed plan for '{selected_feature}':")
    
    questions = st.session_state.get(selected_feature, [])

    if not questions:
        st.error("No questions available for the selected feature.")
        return False

    if 'feature_answers' not in st.session_state:
        st.session_state.feature_answers = {}

    if selected_feature not in st.session_state.feature_answers:
        st.session_state.feature_answers[selected_feature] = {}

    for q in questions:
        question_key = q['question']
        if question_key not in st.session_state.feature_answers[selected_feature]:
            st.session_state.feature_answers[selected_feature][question_key] = ''

        key = f"{selected_feature}_{question_key}"
        if q['type'] == 'text':
            st.session_state.feature_answers[selected_feature][question_key] = st.text_input(
                question_key, 
                value=st.session_state.feature_answers[selected_feature][question_key],
                key=key
            )
        elif q['type'] == 'select' and 'options' in q:
            current_value = st.session_state.feature_answers[selected_feature][question_key]
            index = q['options'].index(current_value) if current_value in q['options'] else 0
            st.session_state.feature_answers[selected_feature][question_key] = st.selectbox(
                question_key, 
                q['options'],
                index=index,
                key=key
            )
        
        # Add file uploader if the question is about file uploads
        if "upload" in question_key.lower() and st.session_state.feature_answers[selected_feature][question_key] == "Yes":
            uploaded_file = st.file_uploader(f"Upload file for {selected_feature}", type=["csv", "json", "jpg", "png", "txt"])
            if uploaded_file is not None:
                st.session_state.feature_answers[selected_feature]["uploaded_file"] = uploaded_file
                st.success(f"File {uploaded_file.name} uploaded successfully!")
    
    return True

@st.cache_data
def generate_plan(selected_feature):
    if all(st.session_state.feature_answers[selected_feature].values()):
        feature_plan = create_feature(selected_feature, str(st.session_state.feature_answers[selected_feature]), st.session_state.chosen_model_name)
        st.write(feature_plan)
        st.session_state.feature_plan = feature_plan
    else:
        st.error("Please fill out all the fields to generate the plan.")


def get_aws_doc_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching AWS documentation: {str(e)}"
    

@st.cache_data
def generate_cft_template(final_plan, feature, user_answers, selected_model):
    # List of template types
    template_types = [
        "Core Infrastructure",
        f"{selected_model} Integration",
        "Data Processing and Analytics",
        "Monitoring and Alerting",
        "User Interface",
        "Deployment",
        "Networking",
    ]

    templates = []

    # Use ThreadPoolExecutor to manage concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Prepare futures for each template type
        futures = {}

        for template_type in template_types:
            prompt = f"""
            Based on the following information:
            - Feature: {feature}
            - Chosen AI model: {selected_model}
            - User answers: {user_answers}
            - Final implementation plan: {final_plan}

            Create a comprehensive CloudFormation template for the {template_type} component of the solution and generate a cloudformation template only nothing else.
            Follow these steps in your thought process:
            1. Analyze the requirements specific to this component.
            2. List all necessary AWS resources for this component.
            3. Consider dependencies and interactions with other components.
            4. Implement best practices for security, scalability, and maintainability.
            5. Include appropriate IAM roles and policies with least privilege principle.
            6. Implement error handling and logging mechanisms.
            7. Consider cost optimization strategies.
            8. Include relevant Parameters, Mappings, and Outputs sections.
            9. Use intrinsic functions and pseudo parameters where appropriate.
            10. Do not include any comments in the YAML format.
            11. Ensure the template is valid YAML format and follows AWS best practices.
            12. Provide a complete and correct CloudFormation template.
            13. Do not include any text outside the YAML format.
            14. If the AI model is not in the bedrock then you have to set it up on lambda using langchain.

            Provide the complete YAML format CloudFormation template, ensuring every line is correct and follows the latest AWS best practices.
            """
            # Submit the invoke_claude call to the executor
            futures[executor.submit(invoke_claude, prompt)] = template_type

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            template_type = futures[future]
            try:
                cft_yaml = future.result()
                templates.append({
                    "name": template_type,
                    "content": cft_yaml
                })
            except Exception as e:
                templates.append({
                    "name": template_type,
                    "content": f"Error generating template: {e}"
                })

    # Display templates using Streamlit's expanders
    for template in templates:
        with st.expander(f"CloudFormation Template: {template['name']}"):
            st.code(template['content'], language='yaml')

    return templates

def create_custom_feature(feature_name, task, model_name):
    prompt = f"Create a high-level implementation plan for a custom feature named '{feature_name}' given the task '{task}' and the chosen model '{model_name}'. Include key steps, considerations, and potential challenges that the customer could face based upon his budget or any other contraints. Do not inclde and cft templates. Think step by step in the <thinking> tags and provide a concise response."
    return invoke_claude(prompt)

def generate_cost_estimate(feature, implementation_details):
    prompt = f"Based on the feature '{feature}' and its implementation details: {implementation_details}, provide a rough cost estimate for development and deployment. Include potential AWS service costs that is gonna incur for this. Think step by step in the <thinking> tags and provide a concise response. Provide it in $"
    return invoke_claude(prompt)

def suggest_integrations(feature, task):
    prompt = f"Suggest potential third-party integrations or APIs that could enhance the '{feature}' feature for the task: '{task}'. Provide a brief description of how each integration could be beneficial.Think step by step in the <thinking> tags and provide a concise response."
    return invoke_claude(prompt)

def generate_testing_strategy(feature, implementation_details):
    prompt = f"Create a testing strategy for the '{feature}' feature with the following implementation details: {implementation_details}. Include unit tests, integration tests, and potential edge cases to consider. Think step by step in the <thinking> tags and provide a concise response."
    return invoke_claude(prompt)

def create_user_story(feature, task):
    prompt = f"Write a user story for the '{feature}' feature in the context of the task: '{task}'. Follow the standard user story format: 'As a [type of user], I want [goal] so that [benefit]'. Think step by step in the <thinking> tags and provide a concise response."
    return invoke_claude(prompt)

# Define the helper function for generating approaches
def generate_approaches(selected_feature, user_task, model_name):
    search_query = f"AWS deployment approaches for {selected_feature} in {user_task} using {model_name}"
    search_results = search_google(search_query)

    augmented_prompt = f"""Based on the following Google search results and your knowledge, provide exactly four comprehensive deployment approaches for the feature '{selected_feature}' for the task '{user_task}' using the model '{model_name}'. Use ONLY the following four approaches:

    1. AWS Bedrock
    2. AWS SageMaker
    3. Custom Solution (using EC2 or ECS)
    4. Hybrid Solution (combining any of the above)

    Google Search Results:
    {' '.join(search_results)}

    For each approach, provide:
    1. A brief description (2-3 sentences)
    2. Key steps for implementation (bullet points)
    3. Advantages (3-4 points)
    4. Disadvantages (2-3 points)
    5. Best practices (3-4 points)
    6. Estimated complexity (Low, Medium, High)

    Format each approach as follows:

    ## [Approach Name]

    Description: [Brief description]

    Implementation Steps:
    - Step 1
    - Step 2
    - ...

    Advantages:
    - Advantage 1
    - Advantage 2
    - ...

    Disadvantages:
    - Disadvantage 1
    - Disadvantage 2
    - ...

    Best Practices:
    - Practice 1
    - Practice 2
    - ...

    Estimated Complexity: [Low/Medium/High]

    Ensure the information is up-to-date and reflects the latest AWS best practices as of 2024.
    """

    response = invoke_claude(augmented_prompt)
    return response

# Define the final plan generation
def generate_final_plan(selected_feature, selected_approach):
    prompt = f"""Generate a detailed implementation plan for the feature '{selected_feature}' using the chosen approach '{selected_approach}'. Include all key steps, architecture, and AWS services for deployment:
    
        1. Analyze the requirements specific to this component.
        2. List all necessary AWS resources for this component.
        3. Consider dependencies and interactions with other components.
        4. Implement best practices for security, scalability, and maintainability.
        5. Include appropriate IAM roles and policies with least privilege principle.
        6. Consider cost optimization strategies.
        7. If the AI model is not in the bedrock, then you have to set it up on Lambda using LangChain.
    """
    return invoke_claude(prompt)

def extract_approach_headings(approaches_text):
    # Split the text into lines and extract headings (assuming they are capitalized or followed by a colon)
    lines = approaches_text.split('\n')
    headings = [line.strip() for line in lines if line.strip() and (line.isupper() or ':' in line)]
    return headings


# Add these helper functions after the existing helper functions
def get_public_ip():
    try:
        # Attempt to get the public IP from the instance metadata
        response = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=2)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching public IP from metadata service: {e}")

    try:
        # Fallback to an external service
        response = requests.get('https://api.ipify.org', timeout=5)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching public IP from ipify: {e}")

    return None

def generate_unique_token():
    return str(uuid.uuid4())

def find_free_port(start_port=8502):
    port = start_port
    while port < 9000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free port found.")

def run_demo_app(port, token, process_log="demo_app.log"):
    try:
        with open(process_log, "w", encoding='utf-8') as log_file:
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "demo_app.py", 
                 f"--server.port={port}", "--server.address=0.0.0.0", 
                 f"--server.baseUrlPath={token}"],
                stdout=log_file,
                stderr=log_file,
                close_fds=True
            )
    except Exception as e:
        st.error(f"Failed to launch demo app: {e}")

def is_port_open(port, host='0.0.0.0'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except:
            return False

def extract_code_from_response(response_text):
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    return None

# Functions from new101 code
@st.cache_data
def get_leaderboard_models():
    client = Client("open-llm-leaderboard/open_llm_leaderboard")
    try:
        result = client.predict(api_name="/get_latest_data_queue")
        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict):
            data_dict = result[0]
        elif isinstance(result, dict):
            data_dict = result
        else:
            raise ValueError("Unexpected data structure from the API")

        if 'data' in data_dict:
            models = []
            for item in data_dict['data']:
                if len(item) > 1:
                    model_link_html = item[0]
                    match = re.search(r'href="https://huggingface\.co/([^"]+)"', model_link_html)
                    if match:
                        model_id = match.group(1)
                        models.append(model_id)
            return models
        else:
            raise KeyError("Missing 'data' key in the API response")
    except Exception as e:
        st.error(f"ERROR: Failed to fetch data from the API. Reason: {e}")
        return []

@st.cache_data
def get_bedrock_models():
    try:
        response = bedrock.list_foundation_models()
        models = response['modelSummaries']
        return [
            {
                "id": model['modelId'],
                "name": model['modelName'],
                "provider": model['providerName']
            }
            for model in models
        ]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []
    
@st.cache_data
def get_bedrock_models():
    return aggregate_models()

def aggregate_models():
    models = []
    
    # Add AWS Bedrock models
    try:
        response = bedrock.list_foundation_models()
        bedrock_models = response['modelSummaries']
        models.extend([
            {
                "id": model['modelId'],
                "name": model['modelName'],
                "provider": "AWS Bedrock",
                "type": "bedrock"
            }
            for model in bedrock_models
        ])
    except Exception as e:
        st.error(f"Error fetching AWS Bedrock models: {str(e)}")
    
    # Add OpenAI models
    openai_models = [
        {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "type": "openai"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI", "type": "openai"},
        {"id": "text-davinci-002", "name": "Davinci", "provider": "OpenAI", "type": "openai"},
    ]
    models.extend(openai_models)
    
    # Add Anthropic models
    anthropic_models = [
        {"id": "claude-2", "name": "Claude 2", "provider": "Anthropic", "type": "anthropic"},
        {"id": "claude-instant", "name": "Claude Instant", "provider": "Anthropic", "type": "anthropic"},
    ]
    models.extend(anthropic_models)
    
    # Add Hugging Face models
    huggingface_models = [
        {"id": "gpt2", "name": "GPT-2", "provider": "Hugging Face", "type": "huggingface"},
        {"id": "distilbert-base-uncased", "name": "DistilBERT", "provider": "Hugging Face", "type": "huggingface"},
    ]
    models.extend(huggingface_models)
    
    # Add Midjourney (Note: Midjourney doesn't have a direct API, so this is for demonstration)
    midjourney_models = [
        {"id": "midjourney-v4", "name": "Midjourney v4", "provider": "Midjourney", "type": "midjourney"},
    ]
    models.extend(midjourney_models)
    
    # Add Grok (Note: As of now, Grok is not publicly available, so this is for future consideration)
    grok_models = [
        {"id": "grok-1", "name": "Grok-1", "provider": "xAI", "type": "grok"},
    ]
    models.extend(grok_models)
    
    return models
 

@st.cache_data
def filter_top_models(models, industry, use_case):
    # Get the aggregate models
    aggregate_models = get_bedrock_models()
    
    # Combine model IDs from both sources
    all_model_ids = [model['id'] for model in aggregate_models] + models
    
    # Remove duplicates while preserving order
    all_model_ids = list(dict.fromkeys(all_model_ids))
    
    prompt = f"""You are provided with a list of model IDs from various sources including the Hugging Face Leaderboard and other AI providers:
{', '.join(all_model_ids[:200])}.

Considering the industry: '{industry}' and use case: '{use_case}',
please select and list the top 5 model IDs best suited for this scenario both from the accuracy and cost perspective.

*Instructions:*
- Provide only the model IDs from the provided list.
- Separate each model ID with a comma.
- Do not include any additional text, explanations, or numbering."""

    response = invoke_claude(prompt)

    if response:
        extracted_models = re.findall(r'\b(?=.*[A-Za-z])[A-Za-z0-9\-/]+\b', response)
        valid_models = []
        for model in extracted_models:
            if model in all_model_ids:
                valid_models.append(model)
            else:
                matches = get_close_matches(model, all_model_ids, n=1, cutoff=0.8)
                if matches:
                    valid_models.append(matches[0])

        seen = set()
        valid_models = [x for x in valid_models if not (x in seen or seen.add(x))]

        if len(valid_models) < 5:
            additional_needed = 5 - len(valid_models)
            remaining_models = [m for m in all_model_ids if m not in valid_models]
            if remaining_models:
                supplementary_models = remaining_models[:additional_needed]
                valid_models.extend(supplementary_models)

        # Get dynamic information for the top 5 models
        top_models_info = []
        for model_id in valid_models[:5]:
            model_info = next((m for m in aggregate_models if m['id'] == model_id), None)
            if not model_info:
                model_info = {"id": model_id, "provider": "Hugging Face"}
            
            # Get dynamic information for the model
            dynamic_info = get_model_dynamic_info(model_info['id'], industry, use_case)
            
            model_info.update(dynamic_info)
            top_models_info.append(model_info)

        return top_models_info
    return []

def get_model_dynamic_info(model_id, industry, use_case):
    prompt = f"""
    For the AI model with ID '{model_id}', considering the industry '{industry}' and use case '{use_case}',
    provide the following information:
    1. Cost (as a numeric value per 1000 tokens)
    2. Accuracy (as a numeric value between 0 and 100)
    3. Throughput (as a numeric value in tokens per second)
    4. Content capabilities (brief description)

    Format the response as a Python dictionary.
    Example format:
    {{
        "cost": 0.02,
        "accuracy": 95.5,
        "throughput": 1000,
        "content_capabilities": "Text generation, summarization, and translation"
    }}
    Ensure that the response is a valid Python dictionary.
    """
    
    response = invoke_claude(prompt)
    
    try:
        dynamic_info = eval(response)
        if not isinstance(dynamic_info, dict):
            raise ValueError("Response is not a valid dictionary")
    except (SyntaxError, ValueError):
        # If parsing fails, use default values
        dynamic_info = {
            "cost": round(random.uniform(0.01, 0.05), 3),
            "accuracy": round(random.uniform(85, 98), 1),
            "throughput": int(random.uniform(500, 2000)),
            "content_capabilities": f"AI model suitable for {use_case} in {industry}"
        }
    
    return dynamic_info


@st.cache_data
def load_custom_dataset(file):
    try:
        df = pd.read_csv(file)
        all_cols = df.columns.tolist()
        for col in all_cols:
            df[col] = df[col].astype(str)

        label_col = next((col for col in all_cols if df[col].dtype in [np.int64, np.float64, 'object'] and df[col].nunique() <= 100), None)

        if label_col is None:
            raise ValueError("No suitable label column found in the dataset.")

        df = df.rename(columns={label_col: 'label'})
        text_columns = [col for col in df.columns if col != 'label']
        df['text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
        df = df[['text', 'label']]
        df = df[(df['text'].str.strip() != "") & (df['label'].str.strip() != "")]

        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading custom dataset: {e}")
        return None

@st.cache_resource
def get_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, dataset, k=5):
    try:
        sentence_model = get_sentence_model()
        texts = [sample['text'] for sample in dataset if isinstance(sample.get('text', ''), str) and sample['text'].strip()]
        embeddings = sentence_model.encode(texts, convert_to_numpy=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        query_embedding = sentence_model.encode([query], convert_to_numpy=True)
        distances, similar_indices = index.search(query_embedding, k=k)

        similar_indices = similar_indices.flatten()
        valid_indices = [int(idx) for idx in similar_indices if 0 <= idx < len(texts)]

        similar_samples = [dataset[idx] for idx in valid_indices]
        return similar_samples
    except Exception as e:
        st.error(f"Error during semantic search: {e}")
        return []

def perform_rag(model, query, dataset):
    relevant_docs = semantic_search(query, dataset)
    if relevant_docs:
        context = " ".join([doc['text'] for doc in relevant_docs if 'text' in doc and doc['text'].strip() != ""])
        if context.strip() != "":
            formatted_context = "\n".join([f"Record {idx+1}: {doc['text']}" for idx, doc in enumerate(relevant_docs)])
            rag_prompt = f"Here is the context from the dataset:\n{formatted_context}\n\nQuery: {query}\n\nResponse:"
            return invoke_claude(rag_prompt)
    return None

def process_model(model, dataset, rag_query):
    start_time = time.time()
    rag_response = perform_rag(model, rag_query, dataset) if rag_query else None
    elapsed_time = time.time() - start_time
    return model, rag_response, elapsed_time

def fetch_models():
    url = "https://huggingface.co/api/models"
    response = requests.get(url)
    return response.json()

# Function to create a leaderboard DataFrame
def create_leaderboard(models):
    data = {
        "Model Name": [model["modelId"] for model in models],
        "Industry": [model.get("tags", ["Unknown"])[0] for model in models],
        "Capabilities": [model.get("pipeline_tag", "N/A") for model in models],
    }
    return pd.DataFrame(data)

def find_recommended_model(models, dataset, rag_query):
    with st.spinner("Finding recommended model..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_model, model, dataset, rag_query) for model in models]
            results = [future.result() for future in as_completed(futures)]

    fastest_model = min(results, key=lambda x: x[2])[0]
    return fastest_model

def analyze_models(models, dataset, rag_query):
    if not rag_query:
        st.warning("Please enter a RAG query for analysis.")
        return None

    results = []
    with st.spinner("Analyzing models..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_model, model, dataset, rag_query) for model in models]
            results = [future.result() for future in as_completed(futures)]

    fastest_model = min(results, key=lambda x: x[2])[0]
    st.success(f"🏆 Fastest Model: {fastest_model}")

    for model, rag_response, elapsed_time in results:
        with st.expander(f"📌 {model} {'(Fastest)' if model == fastest_model else ''}"):
            st.write(f"**Processing Time:** {elapsed_time:.2f} seconds")
            if rag_response:
                st.write("**RAG Response:**")
                st.write(rag_response)
            else:
                st.write("No RAG response generated.")

    return fastest_model

def load_sample_dataset():
    # Load a sample dataset from Hugging Face
    dataset = load_dataset("lhoestq/demo1", split="train")  # Replace with the actual dataset name
    return pd.DataFrame(dataset)

def load_custom_dataset(uploaded_file):
    # Load custom dataset from the uploaded CSV file
    try:
        dataset = pd.read_csv(uploaded_file)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

if 'show_deploy_button' not in st.session_state:
    st.session_state.show_deploy_button = False
if 'show_credentials' not in st.session_state:
    st.session_state.show_credentials = False
if 'generated_templates' not in st.session_state:
    st.session_state.generated_templates = None

def toggle_credentials_view():
    st.session_state.show_credentials = True

def generate_ai_insights(leaderboard_data):
    prompt = f"""
    Analyze the following AI model leaderboard data and provide insights:
    {leaderboard_data.to_string()}

    Please provide:
    1. A summary of the top 3 performing models
    2. Any notable trends or patterns in the data
    3. Recommendations for model selection based on different use cases
    4. Suggestions for potential areas of improvement for the models

    Format the response in Markdown.
    """
    return invoke_claude(prompt)

def generate_dynamic_metrics(models):
    prompt = f"""
    Given a list of {len(models)} AI models, generate the following metrics:
    1. Average Performance Score (between 80 and 100)
    2. Average Inference Time (between 20ms and 40ms)
    3. Estimate the productivity increase (between 15% and 30%)
    4. Estimate the cost savings (between 10% and 25%)

    Provide the results in the following format:
    Average Performance Score: [score]
    Average Inference Time: [time]ms
    Productivity Increase: [percentage]%
    Cost Savings: [percentage]%
    """
    response = invoke_claude(prompt)
    return response.strip().split('\n')

def generate_model_metrics(model):
    prompt = f"""
    Generate realistic metrics for the AI model '{model}' in the following format:
    Industry: [industry name]
    Capabilities: [brief description of capabilities]
    Performance Score: [score between 80 and 100]
    Inference Time: [time between 20 and 40]ms
    """
    response = invoke_claude(prompt)
    return response.strip().split('\n')

def parse_model_metrics(metrics):
    parsed = {}
    for metric in metrics:
        key, value = safe_split(metric, ': ', 1)  # Use safe_split to ensure 2 values
        if key == 'Performance Score':
            parsed[key] = float(value) if value != "N/A" else 0.0  # Handle cases where value is "N/A"
        elif key == 'Inference Time':
            parsed[key] = float(value.rstrip('ms')) if value != "N/A" else 0.0
        else:
            parsed[key] = value

    return parsed

def safe_split(s, sep=':', maxsplit=1):
    parts = s.split(sep, maxsplit)
    return parts if len(parts) == 2 else (s, "N/A")

def create_aws_session(access_key, secret_key, region='us-east-1'):
    """Create an AWS session using provided credentials"""
    try:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        return session
    except Exception as e:
        st.error(f"Failed to create AWS session: {str(e)}")
        return None

def deploy_cloudformation(session, template_content, stack_name):
    """Deploy CloudFormation template to AWS"""
    try:
        cf_client = session.client('cloudformation')
        
        # Create/Update the stack
        try:
            cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_content,
                Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
            )
            return True, "Stack creation initiated successfully!"
        except ClientError as e:
            if 'AlreadyExistsException' in str(e):
                # If stack exists, update it
                try:
                    cf_client.update_stack(
                        StackName=stack_name,
                        TemplateBody=template_content,
                        Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
                    )
                    return True, "Stack update initiated successfully!"
                except ClientError as e:
                    if 'No updates are to be performed' in str(e):
                        return True, "No updates needed - stack is up to date!"
                    return False, f"Failed to update stack: {str(e)}"
            return False, f"Failed to create stack: {str(e)}"
    except Exception as e:
        return False, f"Error deploying CloudFormation: {str(e)}"

def show_deployment_interface(templates):
    """Show the deployment interface with AWS credentials collection"""
    st.subheader("Deploy to AWS")
    
    if st.button("Ready to Deploy!"):
        # Create columns for the credential inputs
        col1, col2 = st.columns(2)
        
        with col1:
            access_key = st.text_input("AWS Access Key", type="password")
        with col2:
            secret_key = st.text_input("AWS Secret Key", type="password")
        
        stack_name = st.text_input("Stack Name", value="my-stack")
        
        if st.button("Deploy"):
            if not access_key or not secret_key:
                st.error("Please provide both AWS Access Key and Secret Key")
                return
            
            if not stack_name:
                st.error("Please provide a stack name")
                return
            
            # Create AWS session
            session = create_aws_session(access_key, secret_key)
            if not session:
                return
            
            # Combine templates if multiple
            if isinstance(templates, list):
                combined_templates = "\n---\n".join([t['content'] for t in templates])
            else:
                combined_templates = templates
            
            # Deploy the template
            with st.spinner("Deploying CloudFormation stack..."):
                success, message = deploy_cloudformation(session, combined_templates, stack_name)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)

def main():
    
    st.markdown("""
    <style>
    .custom-header {
        font-size: 32px;
        color: #333;
        padding: 20px 0;
        text-align: center;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-item {
        text-align: center;
        flex: 1;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
        margin-top: 10px;
    }
</style>
    """, unsafe_allow_html=True)

    st.sidebar.title("AI Model Analyzer")
    page = st.sidebar.radio("Navigation", ["Home", "Model Leaderboard"])

    if page == "Home":
        st.markdown('<h1 class="custom-header">AI Model Analyzer</h1>', unsafe_allow_html=True)

        # Initialize session state for use case flow
        if 'use_case_flow' not in st.session_state:
            st.session_state.use_case_flow = 'start'
            st.session_state.questionnaire_responses = {}
            st.session_state.determined_use_case = None

        # Start of use case determination flow
        if st.session_state.use_case_flow == 'start':
            st.markdown("### Do you know your specific use case?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, I know my use case", use_container_width=True):
                    st.session_state.use_case_flow = 'direct_selection'
                    st.rerun()
            
            with col2:
                if st.button("No, help me determine it", use_container_width=True):
                    st.session_state.use_case_flow = 'questionnaire'
                    st.rerun()

        # Questionnaire flow
        elif st.session_state.use_case_flow == 'questionnaire':
            st.markdown("### Use Case Questionnaire")
            
            questions = [
                {
                    "id": "business_objective",
                    "question": "What is your primary business objective?",
                    "options": [
                        "Automate processes",
                        "Improve customer experience",
                        "Generate content",
                        "Analyze data",
                        "Other"
                    ]
                },
                {
                    "id": "data_type",
                    "question": "What type of data will you be working with?",
                    "options": [
                        "Text documents",
                        "Images",
                        "Structured data",
                        "Mixed data types",
                        "Other"
                    ]
                },
                {
                    "id": "industry_focus",
                    "question": "Which industry sector best describes your focus?",
                    "options": [
                        "Healthcare",
                        "Financial Services",
                        "Retail",
                        "Technology",
                        "Other"
                    ]
                },
            ]

            responses_complete = True
            for question in questions:
                with st.container():
                    st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
                    response = st.radio(
                        question["question"],
                        options=question["options"],
                        key=question["id"]
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if response:
                        st.session_state.questionnaire_responses[question["id"]] = response
                    else:
                        responses_complete = False

            if responses_complete and st.button("Analyze Responses", use_container_width=True):
                with st.spinner("Analyzing your responses..."):
                    # Simulate analysis based on responses
                    business_objective = st.session_state.questionnaire_responses['business_objective']
                    data_type = st.session_state.questionnaire_responses['data_type']
                    industry_focus = st.session_state.questionnaire_responses['industry_focus']
                    
                    # Simple logic to determine use case
                    if business_objective == "Generate content" and data_type == "Text documents":
                        use_case = "Content Generation"
                    elif data_type == "Images":
                        use_case = "Image Generation"
                    elif business_objective == "Improve customer experience":
                        use_case = "Chatbot"
                    else:
                        use_case = "Content Generation"  # Default case
                    
                    # Map industry focus to simplified categories
                    industry_mapping = {
                        "Healthcare": "HCL",
                        "Financial Services": "FSI",
                        "Retail": "Retail",
                        "Technology": "Creative",
                        "Other": "Others"
                    }
                    industry = industry_mapping.get(industry_focus, "Others")
                    
                    st.session_state.determined_use_case = {
                        "industry": industry,
                        "use_case": use_case
                    }
                    st.session_state.use_case_flow = 'complete'
                    st.rerun()

        # Direct selection flow
        elif st.session_state.use_case_flow == 'direct_selection':
            st.markdown("### Select Your Industry and Use Case")
            
            industries = ["HCL", "FSI", "Retail", "Creative", "Others"]
            use_cases = ["Content Generation", "Image Generation", "Chatbot"]

            industry = st.selectbox("Select an industry:", industries, key="direct_industry_selector")
            use_case = st.selectbox("Select a use case:", use_cases, key="direct_use_case_selector")

            if st.button("Proceed", use_container_width=True):
                st.session_state.determined_use_case = {
                    "industry": industry,
                    "use_case": use_case
                }
                st.session_state.use_case_flow = 'complete'
                st.rerun()

        # Complete flow - proceed with model analysis
        if st.session_state.use_case_flow == 'complete' and st.session_state.determined_use_case:
            industry = st.session_state.determined_use_case["industry"]
            use_case = st.session_state.determined_use_case["use_case"]
            
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### Selected Use Case")
                st.markdown(f"**Industry:** {industry}")
                st.markdown(f"**Use Case:** {use_case}")
                st.markdown('</div>', unsafe_allow_html=True)

            user_task = f"{use_case} for {industry}"
            st.markdown(f'<p class="big-font">Your task: {user_task}</p>', unsafe_allow_html=True)

            # Continue with existing model analysis flow
            with st.spinner("Retrieving and filtering models..."):
                all_models = get_leaderboard_models()
                if all_models:
                    top_models = filter_top_models(all_models, industry, use_case)
                    if top_models:
                        st.session_state.top_models = top_models
                        st.success("Top 5 Models Generated Successfully!")

                        # Continue with the rest of your existing code for model display and analysis
                        display_data = []
                        for model in top_models:
                            display_data.append({
                                "Name": model.get('name', model['id']),
                                "Provider": model['provider'],
                                "Cost (per 1K tokens)": f"${model['cost']:.3f}",
                                "Accuracy": f"{model['accuracy']:.1f}%",
                                "Throughput (tokens/s)": f"{model['throughput']:,}",
                                "Content Capabilities": model['content_capabilities']
                            })

                        model_df = pd.DataFrame(display_data)
                        st.table(model_df)

                        if 'top_models' in st.session_state:
                            st.subheader("📊 Model Analysis")

                            data_option = st.radio("Choose a data option:", ["Upload dataset", "Use sample dataset", "Proceed without dataset"], key="data_option")

                            if data_option == "Upload dataset":
                                uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")
                                if uploaded_file:
                                    dataset = load_custom_dataset(uploaded_file)
                                    if dataset is not None:
                                        st.success("Dataset loaded successfully!")
                                    else:
                                        st.error("Failed to load dataset. Please check the file format.")
                                        return
                                else:
                                    st.warning("Please upload a dataset to proceed.")
                                    return
                            elif data_option == "Use sample dataset":
                                dataset = load_sample_dataset()
                                if dataset is not None:
                                    st.success("Sample dataset loaded successfully!")
                                else:
                                    st.error("Failed to load sample dataset.")
                                    return
                            else:
                                dataset = None

                            # Only allow RAG query if a dataset is loaded
                            if dataset is not None:
                                rag_query = st.text_input("Enter a query for RAG analysis:", key="rag_query")

                                if st.button("Analyze All Models", key="analyze_all_models"):
                                    fastest_model = analyze_models(st.session_state.top_models, dataset, rag_query)
                                    if fastest_model:
                                        st.session_state.fastest_model = fastest_model.get('name', fastest_model['id'])

                            if 'fastest_model' in st.session_state:
                                st.write(f"Fastest model: {st.session_state.fastest_model}")
                                if st.button("Proceed with Fastest Model", key="proceed_fastest_model"):
                                    st.session_state.selected_model = st.session_state.fastest_model
                                    st.write(f"You've selected {st.session_state.selected_model} for further use.")

                            model_names = [model.get('name', model['id']) for model in st.session_state.top_models]
                            selected_model_name = st.selectbox("Or choose a specific model:", model_names, key="specific_model_selector")

                            if st.button("Proceed with Selected Model", key="proceed_selected_model"):
                                st.session_state.selected_model = selected_model
                                st.write(f"You've selected {selected_model} for further use.")

                        if 'selected_model' in st.session_state:
                            st.subheader("Differentiated Features")
                            feature_recommendations = get_feature_recommendations(user_task, st.session_state.selected_model)
                            cleaned_recommendations = [line.strip() for line in feature_recommendations.split("\n") if line.strip()]
                            st.write("\n".join(cleaned_recommendations))

                            # Let user select a feature or create a custom feature
                            feature_options = cleaned_recommendations + ["Create custom feature"]
                            selected_feature = st.selectbox(
                                "Select a feature from the recommendations or create a custom one:",
                                feature_options,
                                key="feature_selector"
                            )

                            if selected_feature == "Create custom feature":
                                custom_feature_name = st.text_input("Enter the name of your custom feature:", key="custom_feature_input")
                                if custom_feature_name:
                                    custom_feature_plan = create_custom_feature(custom_feature_name, user_task, st.session_state.selected_model)
                                    st.write(custom_feature_plan)
                                    selected_feature = custom_feature_name
                                else:
                                    st.warning("Please enter a name for your custom feature.")

                            if selected_feature:
                                st.session_state.selected_feature = selected_feature

                                if st.button("Next", key=f"generate_questions_{selected_feature}"):
                                    if selected_feature not in st.session_state:
                                        questions = generate_questions(selected_feature)
                                        st.session_state[selected_feature] = questions

                                if selected_feature in st.session_state:
                                    questions_displayed = display_questions(selected_feature)

                                    if questions_displayed and all(st.session_state.feature_answers[selected_feature].values()):
                                        st.subheader("Choose Deployment Approaches")

                                        # Generate different approaches for the user to choose from
                                        if "approaches" not in st.session_state:
                                            approaches = generate_approaches(selected_feature, user_task, st.session_state.selected_model)
                                            st.session_state.approaches = approaches
                                        
                                        st.write(st.session_state.approaches)

                                        # Let the user choose an approach
                                        approach_options = ["AWS Bedrock", "AWS SageMaker", "Custom Solution", "Hybrid Solution"]
                                        selected_approach = st.selectbox("Choose an approach for deployment:", approach_options, key="approach_selector")

                                        # Store the selected approach
                                        st.session_state.selected_approach = selected_approach

                                        if st.session_state.selected_approach:
                                            if st.button("Generate Final Plan", key="generate_final_plan"):
                                                # Generate the final plan based on the chosen approach
                                                final_plan = generate_final_plan(selected_feature, st.session_state.selected_approach)
                                                st.session_state.feature_plan = final_plan
                                                st.write(final_plan)

                                            if "feature_plan" in st.session_state:
                                                st.subheader("Further Actions")

                                                st.write("Final Plan:")
                                                st.write(st.session_state.feature_plan)

                                                if st.button("Generate Demo App", key="generate_demo_app"):
                                                    st.info("Generating application code using Claude... Please wait.")
                                                    prompt = f"""
                                                    You are an expert software developer tasked with creating a comprehensive Python Streamlit application. Based on the following app plan, generate complete, production-ready code that can be run on an EC2 instance or locally. Ensure the code is well-structured, fully functional, and includes necessary comments.

						                            Think step by step on how to implement this then how to write the code which libraries to import then what are the best practices and how the api structure is there for the specific model in the <thinking> tags.

                                                    Also try to use concurrent futures for faster processing of the app.
                                                    Also use only these services: Bedrock, Sagemaker, Lambda, S3, Step Functions for aws in eu-central-1 region

                                                    for image generation use the following model: stability.stable-diffusion-xl-v1 in us-east-1 region
                
                                                    App Plan:
                                                    {st.session_state.feature_plan}

                                                    Key Requirements:
                                                    1. Implement all features specified in the app plan.
                                                    2. Include all necessary imports and libraries.
                                                    3. Provide complete logic for each feature - no placeholders.
                                                    4. Ensure the app is fully usable and interactive.
                                                    5. Use appropriate Streamlit components and layout techniques.
                                                    6. Implement robust error handling and user-friendly fallback options.
                                                    7. Follow Python and Streamlit best practices.
                                                    8. Include helpful comments explaining complex logic or important steps.
                                                    9. Use Professional UI for the app
                                                    10. Use Hugging face model that is there initally call it using the hugging face code and then stucturise it.
                                                    11. Add a fallback for Hugging face model to bedrock models.

                                                    Model Integration:
                                                    - Primary: Use Hugging Face models as specified in the app plan.
                                                    - Fallback: If access is denied, use Amazon Bedrock with the following model:
                                                    Model: anthropic.claude-3-5-sonnet-20240620-v1:0
                                                    Region: eu-central-1
                                                    Capabilities: text generation
						                            Structure the prompt structure correctly and then write the code correct for this prompt claude.


                                                    Image Generation:
                                                    If image generation is required, use the following code snippet:
                                                    client = boto3.client("bedrock-runtime", region_name="us-east-1")

                                                    # Set the model ID
                                                    model_id = "stability.stable-diffusion-xl-v1"

                                                    Error Handling:
                                                    - Implement comprehensive error handling throughout the application.
                                                    - Do not display technical errors to the user; provide user-friendly messages or fallback options instead.

                                                    Code Structure and Quality:
                                                    1. Organize the code into logical sections or functions.
                                                    2. Use clear and descriptive variable and function names.
                                                    3. Ensure all imports are correct and necessary.
                                                    4. Implement any required authentication or API key handling securely.

                                                    Final Checks:
                                                    1. Verify all imports are correct and necessary.
                                                    2. Ensure all API integrations are properly implemented.
                                                    3. Check for any logical errors or inconsistencies in the code.
                                                    4. Confirm that all features from the app plan are implemented.

                                                    Output Format:
                                                    Provide the complete Python code for the Streamlit application, including all necessary imports, functions, and the main app logic. Wrap the entire code in <code> tags.

                                                    Generate only the Python code without any additional explanations.
                                                    """
                                                    generated_response = invoke_claude(prompt)
                                                    
                                                    if generated_response is None:
                                                        st.error("Failed to generate code.")
                                                    else:
                                                        extracted_code = extract_code_from_response(generated_response)
                                                        if not extracted_code:
                                                            st.error("Failed to extract code from the generated response.")
                                                        else:
                                                            try:
                                                                with open("demo_app.py", "w", encoding='utf-8') as f:
                                                                    f.write(extracted_code)
                                                                st.success("Application code generated and saved successfully.")
                                                            except Exception as e:
                                                                st.error(f"Failed to save demo app: {e}")

                                                            try:
                                                                demo_port = find_free_port(8502)
                                                            except RuntimeError as e:
                                                                st.error(str(e))
                                                                return

                                                            unique_token = generate_unique_token()

                                                            try:
                                                                run_demo_app(demo_port, unique_token)
                                                                st.info("Launching demo app...")
                                                            except Exception as e:
                                                                st.error(f"Failed to launch demo app: {e}")
                                                                return

                                                            timeout = 30
                                                            start_time = time.time()
                                                            while time.time() - start_time < timeout:
                                                                if is_port_open(demo_port):
                                                                    break
                                                                time.sleep(1)
                                                            else:
                                                                st.error("Demo app failed to start within the expected time. Check 'demo_app.log' for details.")
                                                                return

                                                            public_ip = get_public_ip()
                                                            if public_ip:
                                                                demo_link = f"http://{public_ip}:{demo_port}/{unique_token}/"
                                                                st.markdown(f"**Your demo app is running at:** [Click here to view your app]({demo_link})")
                                                                st.info("Note: It may take a few moments for the app to be accessible externally.")
                                                            else:
                                                                local_link = f"http://localhost:{demo_port}/{unique_token}/"
                                                                st.warning("Unable to determine the public IP address. You can try accessing the app using one of these options:")
                                                                st.markdown(f"1. Local access: [Click here to view your app locally]({local_link})")
                                                                st.markdown("2. Replace 'localhost' with your machine's IP address or domain name if accessed from another device.")
                                                                st.text_input("Enter your public IP or domain name:", key="custom_ip")
                                                                if st.session_state.custom_ip:
                                                                    custom_link = f"http://{st.session_state.custom_ip}:{demo_port}/{unique_token}/"
                                                                    st.markdown(f"Custom link: [Click here to view your app]({custom_link})")

                                                            st.markdown("**Note:** Ensure that `demo_app.py` runs a Streamlit app on the specified port. You can check `demo_app.log` for any errors if the app doesn't start.")

                                                if st.button("Generate CloudFormation Template"):
                                                    with st.spinner("Generating CloudFormation templates..."):
                                                        templates = generate_cft_template(
                                                            st.session_state.feature_plan,  # Pass the final plan
                                                            selected_feature,
                                                            st.session_state.feature_answers[selected_feature],
                                                            st.session_state.selected_model
                                                        )

                                                        if templates:
                                                            combined_templates = "\n---\n".join([t['content'] for t in templates])
                                                            combined_bytes = combined_templates.encode('utf-8')
                                                            st.download_button(
                                                                label="Download All Templates",
                                                                data=combined_bytes,
                                                                file_name="all_templates.yaml",
                                                                mime="application/x-yaml"
                                                            )

                                                            # Store templates in session state for later use
                                                            st.session_state.generated_templates = templates
                                                            st.session_state.show_deploy_button = True
                                                
                                                # Show "Ready to Deploy" button if templates are generated
                                                if st.session_state.show_deploy_button:
                                                    st.button("Ready to Deploy!", on_click=toggle_credentials_view)

                                                # Show credentials form if Ready to Deploy was clicked
                                                if st.session_state.show_credentials:
                                                    with st.form(key='aws_credentials_form'):
                                                        st.subheader("AWS Credentials")
                                                        col1, col2 = st.columns(2)
                                                        
                                                        with col1:
                                                            access_key = st.text_input("AWS Access Key", type="password")
                                                        with col2:
                                                            secret_key = st.text_input("AWS Secret Key", type="password")
                                                        
                                                        stack_name = st.text_input("Stack Name", value="my-stack")
                                                        
                                                        submit_button = st.form_submit_button(label="Deploy")
                                                        
                                                        if submit_button:
                                                            if not access_key or not secret_key:
                                                                st.error("Please provide both AWS Access Key and Secret Key")
                                                            elif not stack_name:
                                                                st.error("Please provide a stack name")
                                                            else:
                                                                # Create AWS session
                                                                session = create_aws_session(access_key, secret_key)
                                                                if session and st.session_state.generated_templates:
                                                                    # Get the stored templates
                                                                    templates = st.session_state.generated_templates
                                                                    combined_templates = "\n---\n".join([t['content'] for t in templates])
                                                                    
                                                                    # Deploy the template
                                                                    with st.spinner("Deploying CloudFormation stack..."):
                                                                        success, message = deploy_cloudformation(
                                                                            session, 
                                                                            combined_templates, 
                                                                            stack_name
                                                                        )
                                                                        
                                                                        if success:
                                                                            st.success(message)
                                                                        else:
                                                                            st.error(message)


                                                            # Generate Testing Strategy
                                                if st.button("Generate Testing Strategy"):
                                                    testing_strategy = generate_testing_strategy(selected_feature, st.session_state.feature_plan)
                                                    st.write(testing_strategy)

                                                            # Generate Cost Estimate
                                                if st.button("Generate Cost Estimate"):
                                                    cost_estimate = generate_cost_estimate(selected_feature, st.session_state.feature_plan)
                                                    st.write(cost_estimate)

                                                            # Suggest Integrations
                                                if st.button("Suggest Integrations"):
                                                    integrations = suggest_integrations(selected_feature, user_task)
                                                    st.write(integrations)

                                                            # Create User Story
                                                if st.button("Create User Story"):
                                                    user_story = create_user_story(selected_feature, user_task)
                                                    st.write(user_story)

    elif page == "Model Leaderboard":
        st.markdown('<h1 class="custom-header">Daily Model Leaderboard</h1>', unsafe_allow_html=True)
        
        # Fetch leaderboard data
        models = get_leaderboard_models()[:50]  # Limit to 50 models
        
        # Generate dynamic metrics
        metrics = generate_dynamic_metrics(models)
        
        # Display key metrics




        # Display leaderboard table
        st.subheader("Model Leaderboard")
        leaderboard_data = []
        for i, model in enumerate(models):
            metrics = generate_model_metrics(model)
            parsed_metrics = parse_model_metrics(metrics)
            leaderboard_data.append({
                "Model Name": model,
                "Industry": parsed_metrics.get('Industry', 'N/A'),
                "Capabilities": parsed_metrics.get('Capabilities', 'N/A'),
                "Performance Score": parsed_metrics.get('Performance Score', 0),
                "Inference Time (ms)": parsed_metrics.get('Inference Time', 0),
                "Last Updated": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        st.dataframe(leaderboard)

                # Performance trend chart
        st.subheader("Performance Trend")

        # Extract model names and performance scores from the leaderboard data
        performance_data = [{"Model": row["Model Name"], "Performance": row["Performance Score"]}
                            for row in leaderboard_data]

        # Create a line chart using Plotly
        fig = px.line(performance_data, x="Model", y="Performance", title="Top Models Performance Score Trend")

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Model details
        st.subheader("Model Details")
        selected_model = st.selectbox("Select a model for more details", leaderboard["Model Name"])
        
        if selected_model:
            model_data = leaderboard[leaderboard["Model Name"] == selected_model].iloc[0]
            st.markdown(f"**Model Name:** {model_data['Model Name']}")
            st.markdown(f"**Industry:** {model_data['Industry']}")
            st.markdown(f"**Capabilities:** {model_data['Capabilities']}")
            st.markdown(f"**Performance Score:** {model_data['Performance Score']}")
            st.markdown(f"**Inference Time:** {model_data['Inference Time (ms)']} ms")
            st.markdown(f"**Last Updated:** {model_data['Last Updated']}")

        # Business Impact
        st.subheader("Business Impact")
        impact_prompt = f"""
        Based on the performance metrics of the top AI models:
        1. Explain how implementing these models can increase productivity by {metrics[2].split(': ')[1]}.
        2. Describe how these models can lead to cost savings of {metrics[3].split(': ')[1]}.
        3. Provide 3 specific examples of how businesses can benefit from these AI models.

        Format the response in Markdown.
        """
        impact_analysis = invoke_claude(impact_prompt)
        st.markdown(impact_analysis)

        # Additional Insights


if __name__ == "__main__":
    main()
