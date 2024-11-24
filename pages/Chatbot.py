import streamlit as st
from chatbot.model import Gemini
from chatbot.data_from_api import returnFromApi  # Import the function to get data

def main():
    # Initialize the app title and layout
    st.title("Gemini Chatbot")
    st.subheader("Powered by Google Gemini AI")
    st.markdown("Start a conversation with the AI below:")

    # Initialize the session state for messages, dates, and preloaded data
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "preloaded_data" not in st.session_state:
        st.session_state.preloaded_data = None
    if "dates_provided" not in st.session_state:
        st.session_state.dates_provided = False
    if "first_response" not in st.session_state:
        st.session_state.first_response = False  # Tracks if the first response has been shown

    # Date input section
    if not st.session_state.dates_provided:
        st.markdown("### Enter the date range to initialize the chatbot:")
        start_date = st.date_input("Start Date (yyyy-mm-dd)")
        end_date = st.date_input("End Date (yyyy-mm-dd)")

        if st.button("Submit"):
            if start_date and end_date:
                try:
                    # Call returnFromApi with the user-provided dates
                    st.session_state.preloaded_data = returnFromApi(
                        start_date.strftime("%Y-%m-%d"), 
                        end_date.strftime("%Y-%m-%d")
                    )
                    st.session_state.dates_provided = True
                except Exception as e:
                    st.error(f"Failed to load preloaded data: {e}")
            else:
                st.warning("Please provide both start and end dates.")

    # Only display the chatbot once dates are provided and data is loaded
    if st.session_state.dates_provided:
        # Generate the first response from Gemini if not already done
        if not st.session_state.first_response:
            gemini = Gemini()

            # Combine preloaded data and a descriptive prompt
            preloaded_data = st.session_state.preloaded_data
            explanation_prompt = (
                """
                {data[0]}
                This is the predicted profit.
                {data[1]}
                This is the total aggregated revenue for each flower.
                {data[2]}
                This is the total profit for each flower.
                {data[3]}
                This is the  top profit generated for each flower.

                This data pertains to a shop owner in rural India. This person will ask
                questions to you with regards to this data. Continue the conversation.
                """
                f"{preloaded_data}"
            )

            try:
                # Get the initial response from Gemini
                first_response = gemini.respond(explanation_prompt)
                st.session_state.messages.append({"role": "assistant", "content": first_response})
                st.session_state.first_response = True
            except Exception as e:
                st.error(f"Failed to generate the initial response: {e}")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input box for user input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Combine user input with the preloaded data as context
            input_with_context = f"Preloaded data: {st.session_state.preloaded_data}\n\nUser input: {user_input}"

            # Display the user's message in the chat
            with st.chat_message("user"):
                st.markdown(user_input)

            # Create an instance of the Gemini class
            gemini = Gemini()

            # Get the response from the Gemini model
            try:
                response = gemini.respond(input_with_context)
            except Exception as e:
                response = "Sorry, something went wrong while processing your request."
                st.error(f"Error: {e}")

            # Display the bot's response in the chat
            with st.chat_message("assistant"):
                st.markdown(response)

            # Append user and bot messages to the session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
