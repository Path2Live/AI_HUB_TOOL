from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
import openai
import pandas as pd
import os


def load_historical_data():
    """
    Loads and processes historical data from a CSV file.

    Returns:
        dict: Processed historical data categorized by work type and sub-category.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'detailed_breakdown_costs.csv')
    df = pd.read_csv(csv_path)

    historical_data = {}
    for _, row in df.iterrows():
        category = row['category']
        sub_category = row['sub-category']
        unit = row['Unit']
        avg_price = row['average_price']
        range_price = row['range_price']

        if category not in historical_data:
            historical_data[category] = []

        historical_data[category].append({
            'sub_category': sub_category,
            'unit': unit,
            'avg_price': avg_price,
            'range_price': range_price
        })

    return historical_data


def load_pdf_contents(pdf_paths):
    """
    Loads contents from a list of PDF files.

    Args:
        pdf_paths (list): List of paths to PDF files.

    Returns:
        str: Concatenated content of all PDF files.
    """
    documents_content = ""
    for pdf in pdf_paths:
        try:
            loader = PyPDFLoader(pdf)
            document = loader.load()
            for page in document:
                documents_content += page.page_content + "\n"
        except Exception as e:
            print(f"Error loading PDF {pdf}: {e}")
    return documents_content


def setup_llm(api_key):
    """
    Sets up the language model with the provided API key.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        OpenAI: Configured language model.
    """
    openai.api_key = api_key
    return OpenAI(model='gpt-4o-2024-05-13')


def extract_project_info(payload):
    """
    Extracts project information from a given payload.

    Args:
        payload (list): List of dictionaries containing project details.

    Returns:
        dict: Extracted project information.
    """
    project_info = {}
    for item in payload:
        question = item["question"]
        answer = item["answer"]
        if "Project Type" in question:
            project_info["work_type"] = answer
        elif "gross internal area" in question:
            project_info["gross_internal_area"] = answer
        elif "expected finishes and materials" in question:
            project_info["expected_finishes"] = answer
        elif "site-specific conditions" in question:
            project_info["site_conditions"] = answer
    return project_info


def extract_cost_info(payload):
    """
    Extracts cost information from a given payload.

    Args:
        payload (list): List of dictionaries containing cost details.

    Returns:
        dict: Extracted cost information.
    """
    cost_info = {}
    for section in payload:
        section_name = section["name"]
        items = section["generic"] + section["specific"]
        cost_info[section_name] = [
            {
                "title": item["title"],
                "isChecked": item["isChecked"],
                "value": item.get("value"),
                "quantity": item.get("quantity"),
                "rate": item.get("rate")
            }
            for item in items if item["isChecked"]
        ]
    return cost_info


def formulate_question(project_details, cost_info, historical_data):
    """
    Formulates a question for the language model based on project and cost details.

    Args:
        project_details (dict): Dictionary containing project details.
        cost_info (dict): Dictionary containing cost details.
        historical_data (dict): Dictionary containing historical cost data.

    Returns:
        str: Formulated question.
    """
    historical_info = ""
    for category, items in historical_data.items():
        historical_info += f"{category}:\n"
        for item in items:
            historical_info += (f"  Sub-category: {item['sub_category']}\n"
                                f"  Unit: {item['unit']}\n"
                                f"  Average Price: {item['avg_price']}\n"
                                f"  Range Price: {item['range_price']}\n")

    question = f"""
    Project Overview:

    You are provided with the following project details:

    Project Details: {project_details}
    Cost Information: {cost_info}
    Historical Data: {historical_info}

    Just try to be approximate to the historical data costs in total and breakdown and reply with the same way as the breakdown is formatted but don't make it exact 100%.
    """
    return question


def chat_completion(messages, api_key):
    """
    Generates a chat completion using the provided messages and API key.

    Args:
        messages (list): List of messages for the chat completion.
        api_key (str): OpenAI API key.

    Returns:
        str: Response from the language model.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=messages
    )
    return response['choices'][0]['message']['content']
