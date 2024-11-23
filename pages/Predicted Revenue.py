from models.functions.predicted_revenue import get_aggregated_results

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Predicted Revenue For All Flowers")

# Date input widgets
start_date = st.date_input("Select Start Date")
end_date = st.date_input("Select End Date")

# Ensure valid date range
if start_date and end_date:
    if start_date > end_date:
        st.error("Start date must be earlier than or equal to the end date.")
    else:
        # Call the function and display the results
        aggregated_results = get_aggregated_results(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        st.subheader("Aggregated Predicted Revenue")
        
        # Convert results to DataFrame for better handling
        results_df = pd.DataFrame(list(aggregated_results.items()), columns=['Flower Name', 'Predicted Revenue'])
        st.dataframe(results_df)

        # Plot using matplotlib - Pie chart
        st.subheader("Revenue Distribution (Pie Chart)")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Pie chart
        ax.pie(results_df['Predicted Revenue'], labels=results_df['Flower Name'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title("Predicted Revenue Distribution by Flower", fontsize=16)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
else:
    st.warning("Please select both start and end dates.")
