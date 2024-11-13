from langchain_community.document_loaders import PyPDFLoader
import openai
import pandas as pd
import os


class DataLoader:
    @staticmethod
    def load_historical_data(csv_filename='detailed_breakdown_costs.csv'):
        """
        Loads and processes historical data from a CSV file.

        Returns:
            dict: Processed historical data categorized by work type and sub-category.
        """
        csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
        df = pd.read_csv(csv_path)

        historical_data = df.groupby('Category').apply(lambda group: group.to_dict('records')).to_dict()
        return historical_data

    @staticmethod
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
                documents_content += "\n".join(page.page_content for page in document) + "\n"
            except Exception as e:
                print(f"Error loading PDF {pdf}: {e}")
        return documents_content


def extract_project_info(payload):
    """
    Extracts project information from a given payload.
    """
    keys = {
        "Project Type": "work_type",
        "gross internal area": "gross_internal_area",
        "expected finishes and materials": "expected_finishes",
        "site-specific conditions": "site_conditions"
    }
    return {v: item["answer"] for item in payload if (v := keys.get(item["question"]))}


def extract_cost_info(payload):
    """
    Extracts cost information from a given payload.
    """
    return {
        section["name"]: [
            {
                "title": item["title"],
                "isChecked": item["isChecked"],
                "value": item.get("value"),
                "quantity": item.get("quantity"),
                "rate": item.get("rate")
            }
            for item in section["generic"] + section["specific"] if item["isChecked"]
        ]
        for section in payload
    }


def formulate_question(project_details, cost_info, historical_data):
    """
    Formulates a question for the language model based on project and cost details.
    """
    historical_info = "\n".join(
        f"{category}:\n" + "\n".join(
            f"  Qty: {item['Qty']}\n  Unit: {item['Unit']}\n  Rate (£): {item['Rate (£)']}\n  Total Cost (£): {item['Total Cost (£)']}"
            for item in items
        )
        for category, items in historical_data.items()
    )

    question = f"""
    
    Historical Data: {historical_info}

    give the total of each category of the Historical Data:
    1- make all your number rounded to whole number like hundreds no units or tens 
    2- give the total of them all 
    3-don't pass 110K in your approximation 
    4- dont reply with any other words around your answer
     
    """
    print(question)
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
        model="gpt-4",  # Specify model name
        messages=messages
    )
    return response['choices'][0]['message']['content']
