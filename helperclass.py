from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from openai import ChatCompletion
import pandas as pd
import pandas as pd
import os


def load_historical_data():
    """
    Loads and processes historical data from a CSV file.

    Returns:
        dict: Processed historical data categorized by work type and sub-category.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'Final_Work_Categories_and_Price_Analysis.csv')
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
    The project has the following details:
    Project details: {project_details}
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
         {cost_info} if it says use AI in any sub category level this mean that based on the supplied drawings you as a bot will decied the cost of needed work based on the supplied drawing files
       - And the historical data:
         {historical_info}
       - Generate a detailed cost breakdown for the new project. 
       - The breakdown should include itemized costs for each aspect of the project, calculated based on the average prices from past projects.

      Step 2: Detailed Cost Breakdown
        - Read and Understand Each Cost Section and its subsections:
            Demolition

    Description:
    Demolition involves the safe and systematic dismantling of existing structures or parts of structures to make way for new construction. This can include buildings, bridges, roads, or any other man-made structures. Demolition must be planned carefully to ensure safety and efficiency, considering factors like the type of materials, structural stability, and environmental impact.
    Substructure
    
    Description:
    The substructure of a building includes all the construction elements that are below the ground level, such as foundations, basements, and underground retaining walls. Its primary purpose is to transfer the loads from the superstructure to the ground, ensuring stability and preventing settlement.
    Superstructure
    
    Description:
    The superstructure refers to all parts of a building or structure above the ground level. This includes columns, beams, floors, walls, and any architectural features. The superstructure provides the overall shape and support for the building, ensuring it can withstand various loads such as its own weight, occupants, and environmental forces.
    Roof
    
    Description:
    The roof is the top covering of a building, designed to protect the interior from weather elements like rain, snow, and sunlight. It includes the structural framework (trusses or beams), insulation, waterproofing layers, and the external covering material (tiles, shingles, metal sheets, etc.). The roof also often incorporates drainage systems to direct water away from the structure.
    External Windows and Doors
    
    Description:
    External windows and doors are openings in the building envelope that provide access, natural light, and ventilation. Windows can be fixed or operable and are made from materials like glass, wood, aluminum, or PVC. Doors serve as entry and exit points and can be made from similar materials. These elements must be designed to provide security, insulation, and aesthetics.
    Partitions
    
    Description:
    Partitions are non-load-bearing walls that divide the interior space of a building into rooms and areas. They can be made from materials such as drywall, glass, metal, or wood. Partitions are used to create functional spaces and can include soundproofing, fire resistance, and aesthetic finishes to meet specific needs.
    Electrical
    
    Description:
    The electrical system in a building encompasses all wiring, outlets, switches, lighting fixtures, and appliances. It includes the distribution of power from the main supply to various circuits within the building, ensuring all areas have access to electricity. Electrical plans must consider safety, load requirements, and efficiency.
    Mechanical
    
    Description:
    Mechanical systems in a building include HVAC (Heating, Ventilation, and Air Conditioning), plumbing, and other systems that ensure a comfortable and functional indoor environment. HVAC systems control temperature and air quality, while plumbing systems manage water supply and waste removal. Other mechanical systems might include elevators, fire protection, and gas supply.
    Preliminaries
    
    Description:
    Preliminaries are the initial tasks and requirements that need to be addressed before the actual construction begins. This includes site preparation, setting up temporary facilities like site offices, securing permits and insurance, providing temporary utilities (water, electricity), and creating access routes. Preliminaries ensure that the site is ready for construction to proceed smoothly.
    
    how they get calculated in real life :
    Demolition

    Calculations:
    
        Volume and Material Quantities: From structural drawings and site surveys, the volume of materials to be demolished is calculated. This includes concrete, steel, bricks, etc.
        Labor and Equipment: Estimating the labor hours and type of machinery required for demolition (e.g., wrecking balls, excavators).
        Disposal Costs: Costs for hauling and disposing of demolition debris are calculated based on local disposal rates.
        Safety Measures: Costs for safety measures like barriers, signage, and PPE (personal protective equipment) are included.
    
    Substructure
    
    Calculations:
    
        Excavation: Volume of soil to be excavated is calculated from foundation plans.
        Concrete and Reinforcement: Quantities of concrete and steel reinforcement are derived from the foundation layout and structural details.
        Formwork: Area of formwork needed for concrete pouring is calculated.
    
    Superstructure
    
    Calculations:
    
        Structural Elements: Quantities of beams, columns, floors, and walls are taken from structural drawings.
        Concrete, Steel, and Other Materials: Calculations are done based on dimensions provided in the drawings.
    
    Roof
    
    Calculations:
    
        Roof Area: Calculated from the roof plans.
        Materials: Quantities for trusses, beams, insulation, waterproofing, and roofing materials are determined from the drawings.
    
    External Windows and Doors
    
    Calculations:
    
        Number and Size: Quantities are determined from elevation drawings and window/door schedules.
        Materials: Types of windows and doors (e.g., wood, aluminum, glass) are specified.
    
    Partitions
    
    Calculations:
    
        Wall Area: Calculated from floor plans.
        Materials: Quantities of drywall, studs, insulation, and finishes are determined.
    
    Electrical
    
    Calculations:
    
        Wiring and Fixtures: Quantities of wiring, outlets, switches, and fixtures are calculated from electrical plans.
        Load Requirements: Ensure adequate capacity for all electrical components.
    
    Mechanical
    
    Calculations:
    
        HVAC Systems: Quantities of ducts, piping, and units are calculated from mechanical drawings.
        Plumbing: Quantities of pipes, fittings, fixtures, and equipment.
        Other Systems: Quantities for elevators, fire protection, and gas systems.
    
    Preliminaries
    
    Calculations:
    
        Site Preparation: Costs for site clearance, temporary fencing, and facilities.
        Temporary Utilities: Costs for providing temporary water, electricity, and other utilities.
        Permits and Insurance: Costs for securing necessary permits and insurance coverage.
    
    General Methodology for Calculation:
    
        Quantification: Use construction drawings, specifications, and BIM (Building Information Modeling) software to quantify materials and labor.
        Unit Rates: Apply unit rates (cost per unit of measure) from cost databases or historical data.
        Summation: Sum the costs for each category (materials, labor, equipment) to get the total cost.
        Contingencies: Add a percentage for contingencies to cover unforeseen costs.
        Overheads and Profits: Include a percentage for overheads and profits.

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
