import streamlit as st
import boto3
import base64
import json
import random
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import ast
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100000,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}]
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

def analyze_model_suitability(model, task):
    prompt = f"""
    Note: Only display the top 5 models based on accuracy, cost, throughput, and content capabilities.
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
    Given the task '{task}', provide information on the top 5 AI models most suitable for this task.
    For each model, provide the following information:
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
            "Name": "GPT-4",
            "Provider": "OpenAI",
            "Cost": 0.03 per 1000 token,
            "Accuracy": 95,
            "Throughput": 10,
            "Content capabilities": "Advanced language understanding and generation"
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
    st.table(df)

    return models_info

@st.cache_data
def get_feature_recommendations(task, model_name):
    prompt = f"Given the task '{task}' and the chosen model '{model_name}', suggest a set of features that can be implemented, such as RAG (Retrieval-Augmented Generation) or other relevant features. Provide a brief description for each feature and while responding directly start with features in a numeric order example start from 1. RAG .Think step by step and the processes of thinking should be done in a <thinking> tag and after that provide a concise answer"
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
    prompt = f"""Generate few specific questions that would be important as a business person who doesn't know much about technology to ask when implementing the '{feature}' feature for an AI application only 2-3 questions to be asked from a business person perspective. Format the response as a Python list of dictionaries, where each dictionary has 'question' and 'type' keys. The 'type' should be either 'text' for open-ended questions or 'select' for multiple choice questions. For 'select' types, include an additional 'options' key with a list of possible choices.
    Also include questions about integrating existing databases or sources or models available for certain tasks.
    Include a question about file upload if relevant to the feature.
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

@st.cache_data
def generate_cft_template(final_plan, feature, user_answers, chosen_model_name):
    prompt = f"""
    Think step by step in the <thinking> tags and provide a concise response.
    Create a CloudFormation template for an AI application with the following feature: '{feature}'.
    The chosen model is '{chosen_model_name}'.
    User answers to feature-specific questions: {user_answers}
    Also, there is a final plan according to which you have to create the template: {final_plan}
    
    Only provide the template
    Think step by step and the processes of thinking should be done in a <thinking> tag and after that provide a concise answer    
    The template should include:
    1. Necessary AWS resources for the feature
    2. Integration with the chosen AI model from the aws bedrock invoke model method
    3. Any databases or APIs mentioned in the user answers
    4. A deployment service using AWS CodePipeline or any other
    5. Parameters for AWS Access Key and Secret Key
    6. If file upload is mentioned, include necessary S3 bucket and IAM permissions
    7. Feature's codes which will be used in the application so include that too with the custom logic in it.
    
    Provide the template in YAML format and ensure that every single line of the CFT template is correct and the user can just copy this template and deploy it easily on their workload without the need to configure anything again.
    """
    cft_yaml = invoke_claude(prompt)
    
    # Create a Streamlit code block for easy copying
    st.code(cft_yaml, language='yaml')
    
    return cft_yaml

def create_custom_feature(feature_name, task, model_name):
    prompt = f"Create a high-level implementation plan for a custom feature named '{feature_name}' given the task '{task}' and the chosen model '{model_name}'. Include key steps, considerations, and potential challenges that the customer could face based upon his budget or any other contraints. Do not inclde and cft templates. Think step by step in the <thinking> tags and provide a concise response."
    return invoke_claude(prompt)

def generate_cost_estimate(feature, implementation_details):
    prompt = f"Based on the feature '{feature}' and its implementation details: {implementation_details}, provide a rough cost estimate for development and deployment. Include potential AWS service costs or any other cost considering the budget. Think step by step in the <thinking> tags and provide a concise response."
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
    prompt = f"Provide various deployment approaches for the feature '{selected_feature}' for the task '{user_task}' using the model '{model_name}'. Options should include a custom solution, hybrid solution, AWS SageMaker, AWS Bedrock, etc."
    return invoke_claude(prompt)

# Define the final plan generation
def generate_final_plan(selected_feature, selected_approach):
    prompt = f"Generate a detailed implementation plan for the feature '{selected_feature}' using the chosen approach '{selected_approach}'. Include all key steps, architecture, and AWS services for deployment."
    return invoke_claude(prompt)

def extract_approach_headings(approaches_text):
    # Split the text into lines and extract headings (assuming they are capitalized or followed by a colon)
    lines = approaches_text.split('\n')
    headings = [line.strip() for line in lines if line.strip() and (line.isupper() or ':' in line)]
    return headings

def main():
    st.title("Modx AI Demo")

    # Allow the user to select an industry and a use case
    industries = ["Healthcare", "Retail", "Banking", "Public Sector", "Aviation", "Construction & Real Estate"]
    use_cases = ["Content Generation", "Image Generation", "Chatbot"]

    industry = st.selectbox("Select an industry:", industries)
    use_case = st.selectbox("Select a use case:", use_cases)

    if industry and use_case:
        # Task description derived from the selected industry and use case
        user_task = f"{use_case} for {industry}"

        # Get and display the top 5 models based on the task
        top_5_models = display_top_5_models(user_task)

        # Extract the names of the top 5 models
        model_names = [model['Name'] for model in top_5_models]

        # Let the user select one of the top 5 models
        chosen_model_name = st.selectbox("Choose a model from the top 5:", model_names)

        # If the selected model changes, reset the feature recommendations
        if "chosen_model_name" not in st.session_state or st.session_state.chosen_model_name != chosen_model_name:
            st.session_state.chosen_model_name = chosen_model_name
            st.session_state.feature_recommendations = None  # Reset the recommendations

        if chosen_model_name:
            st.subheader("Feature Recommendations")

            # Get feature recommendations for the chosen model and task
            if st.session_state.feature_recommendations is None:
                feature_recommendations = get_feature_recommendations(user_task, chosen_model_name)

                # Clean up feature recommendations
                cleaned_recommendations = [line.strip() for line in feature_recommendations.split("\n") if line.strip()]
                st.session_state.feature_recommendations = cleaned_recommendations

            # Display feature recommendations
            st.write("\n".join(st.session_state.feature_recommendations))

            # Let user select a feature or create a custom feature
            feature_options = st.session_state.feature_recommendations + ["Create custom feature"]
            selected_feature = st.selectbox(
                "Select a feature from the recommendations or create a custom one:",
                feature_options,
                key="feature_selector"
            )

            if selected_feature == "Create custom feature":
                custom_feature_name = st.text_input("Enter the name of your custom feature:")
                if custom_feature_name:
                    custom_feature_plan = create_custom_feature(custom_feature_name, user_task, chosen_model_name)
                    st.write(custom_feature_plan)
                    selected_feature = custom_feature_name
                else:
                    st.warning("Please enter a name for your custom feature.")

            if selected_feature:
                st.session_state.selected_feature = selected_feature

                # Add a button to manually trigger question generation
                if st.button("Generate Questions", key=f"generate_questions_{selected_feature}"):
                    # Generate questions for the selected feature
                    if selected_feature not in st.session_state:
                        questions = generate_questions(selected_feature)
                        st.session_state[selected_feature] = questions

                # Only display questions if they have been generated
                if selected_feature in st.session_state:
                    questions_displayed = display_questions(selected_feature)

                    # Check if all questions are answered before generating approaches
                    if questions_displayed and all(st.session_state.feature_answers[selected_feature].values()):
                        st.subheader("Choose Deployment Approaches")

                        # Generate different approaches for the user to choose from
                        if "approaches" not in st.session_state:
                            approaches = generate_approaches(selected_feature, user_task, st.session_state.chosen_model_name)
                            st.session_state.approaches = approaches
                        
                        st.write(st.session_state.approaches)

                        # Extract approach options from the generated text
                        approach_options = extract_approach_headings(st.session_state.approaches)

                        # Let the user choose an approach
                        selected_approach = st.selectbox("Choose an approach for deployment:", approach_options)

                        # Store the selected approach
                        st.session_state.selected_approach = selected_approach

                        if st.session_state.selected_approach:
                            if st.button("Generate Final Plan"):
                                # Generate the final plan based on the chosen approach
                                final_plan = generate_final_plan(selected_feature, st.session_state.selected_approach)
                                st.session_state.feature_plan = final_plan
                                st.write(final_plan)

                            if "feature_plan" in st.session_state:
                                st.subheader("Further Actions")

                                # Generate CloudFormation Template
                                if st.button("Generate CloudFormation Template"):
                                    generate_cft_template(
                                        st.session_state.feature_plan,  # Pass the final plan
                                        selected_feature,
                                        st.session_state.feature_answers[selected_feature],
                                        st.session_state.chosen_model_name
                                    )

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

if __name__ == "__main__":
    main()
