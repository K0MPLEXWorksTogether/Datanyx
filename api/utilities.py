import requests

def get_predicted_profit_api(base_url, start_date, end_date):
    """
    Sends a GET request to the predicted profit API.
    Args:
        base_url (str): The base URL of the API.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    Returns:
        dict: JSON response if status code is 200, else an error message.
    """
    try:
        response = requests.get(f"{base_url}/get_predicted_profit", params={"start_date": start_date, "end_date": end_date})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def get_aggregated_revenue_api(base_url, start_date, end_date):
    """
    Sends a GET request to the aggregated revenue API.
    Args:
        base_url (str): The base URL of the API.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    Returns:
        dict: JSON response if status code is 200, else an error message.
    """
    try:
        response = requests.get(f"{base_url}/get_aggregated_revenue", params={"start_date": start_date, "end_date": end_date})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def get_total_profit_api(base_url, start_date, end_date):
    """
    Sends a GET request to the total profit API.
    Args:
        base_url (str): The base URL of the API.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    Returns:
        dict: JSON response if status code is 200, else an error message.
    """
    try:
        response = requests.get(f"{base_url}/get_total_revenue", params={"start_date": start_date, "end_date": end_date})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def get_top_revenue_api(base_url, start_date, end_date):
    """
    Sends a GET request to the top revenue API.
    Args:
        base_url (str): The base URL of the API.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    Returns:
        dict: JSON response if status code is 200, else an error message.
    """
    try:
        response = requests.get(f"{base_url}/get_total_revenue", params={"start_date": start_date, "end_date": end_date})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
