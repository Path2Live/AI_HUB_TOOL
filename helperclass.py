from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from openai import ChatCompletion


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
    return OpenAI(model='gpt-4o-2024-05-13', openai_api_key=api_key)


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


def formulate_question(project_details, cost_info):
    """
    Formulates a question for the language model based on project and cost details.

    Args:
        project_details (dict): Dictionary containing project details.
        cost_info (dict): Dictionary containing cost details.

    Returns:
        str: Formulated question.
    """
    question = f"""
    The project has the following details:
    Project Type: {project_details.get('work_type', 'N/A')}
    Gross Internal Area: {project_details.get('gross_internal_area', 'N/A')}
    Expected Finishes: {project_details.get('expected_finishes', 'N/A')}
    Site-specific Conditions: {project_details.get('site_conditions', 'N/A')}

    Examine the architectural drawing documents I provided and follow these instructions:

    1. Elevations in files containing elevation:
       - List the materials used for each elevation. Specifically, use the labels beside each material name.

    2. Floor Plan in files containing floor plan:
       - Locate the floor measurements over the drawing. Use the labels beside "scale" to get the real-life to image scale ratio, which will be followed by "@" then the paper size to show the paper size of the drawing.
       - Calculate the area of each room, the total area, and identify any new construction (indicated by blue-colored walls).

    3. Roof Plan in files containing roof plan:
       - Detect any kind of extension which will appear like a side box covered in stripes and give its area.

    4. Feasibility Estimate in files containing Feasibility Estimate:
       - Given the project's cost details:
         {cost_info}
       - And the historical data from past projects in the uploaded file "Untitled spreadsheet - Sheet1.pdf".
       - Generate a detailed cost breakdown for the new project. 
       - The breakdown should include itemized costs for each aspect of the project, calculated based on the average prices from past projects.

        Step 2: Detailed Cost Breakdown
        - Read and Understand Each Cost Section and its subsections:
            - Demolition    
            - Substructure
            - Superstructure
            - Roof
            - External Windows and Doors
            - Partitions
            - Electrical
            - Mechanical
            - Preliminaries

    Provide your final answer with the following sections only:

    Elevations: Provide details for North, South, East, and West elevations including the materials used.
    Floor Plan: Provide the scale of the floor plan and calculate the area of each room. Also, provide details of any new construction highlighted in blue.
    Roof Plan: Provide details of any extensions in the roof plan.
    Feasibility Estimate: Provide a cost estimate for the work done as the main summary and Total excl. VAT with the breakdown for (demolition, substructure, superstructure, etc.).

    Notes:
    Don't use internet, just based on the info supplied.

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
    response = ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        api_key=api_key
    )
    return response['choices'][0]['message']['content']
