from chatbot.api.utilities import (
    get_predicted_profit_api,
    get_aggregated_revenue_api,
    get_total_profit_api,
    get_top_revenue_api
)

base_url_predicted_profit = "http://127.0.0.1:5000"
base_url_aggregated_revenue = "http://127.0.0.1:5001"
base_url_total_profit = "http://127.0.0.1:5002"
base_url_top_revenue = "http://127.0.0.1:5003"

start_date = "2024-01-01"
end_date = "2024-01-31"

print(get_predicted_profit_api(base_url_predicted_profit, start_date, end_date))
print(get_aggregated_revenue_api(base_url_aggregated_revenue, start_date, end_date))
print(get_total_profit_api(base_url_total_profit, start_date, end_date))
print(get_top_revenue_api(base_url_top_revenue, start_date, end_date))
