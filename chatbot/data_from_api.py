from chatbot.api.utilities import get_predicted_profit_api
from chatbot.api.utilities import get_aggregated_revenue_api
from chatbot.api.utilities import get_top_revenue_api
from chatbot.api.utilities import get_total_profit_api

from chatbot.api.main import base_url_aggregated_revenue
from chatbot.api.main import base_url_predicted_profit
from chatbot.api.main import base_url_top_revenue
from chatbot.api.main import base_url_total_profit

import json

def returnFromApi(start_date, end_date) -> json:
    answerList = [
        get_predicted_profit_api(base_url_predicted_profit, start_date, end_date),
        get_aggregated_revenue_api(base_url_aggregated_revenue, start_date, end_date),
        get_total_profit_api(base_url_total_profit, start_date, end_date),
        get_top_revenue_api(base_url_top_revenue, start_date, end_date)        
    ]

    return answerList

def main():
    returnFromApi("2024-10-11", "2024-10-21")