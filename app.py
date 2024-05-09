import streamlit as st
import pandas as pd
import pickle
from forcast import fit_data, get_next_12_months, predict, change_date_format

# Define function to display home page
def home(df=None):
    st.write("# Forcasting future Energy Consumptions")
    st.write("Upload your future Energy Consumptions (prefered if it is above 5 years)")
    # st.write("Please use the navigation bar on the left to see the forcast for the next year")

    if df is not None:
        forcast_next_year(df)
    else:
        st.write("No data available for prediction.")

    

def clustering(df):
    st.write("# Clustering")

    # Load the KMeans model
    with open('birch_model.pkl', 'rb') as f:
        birch = pickle.load(f)

    # User input
    # user_input = st.number_input("Enter a value:")

    cluser_value = df.groupby("ds").agg({"y": "mean"})["y"][0]

    # Predict against the user input
    predicted_cluster = birch.predict([[cluser_value]])
    st.write("Predicted Cluster:", predicted_cluster[0])



# Define function to display page for uploading CSV file
def upload_csv():
    st.write("# Upload CSV File")
    st.write("Please upload a CSV file with two columns.")

    # Allow user to upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Check if the DataFrame has exactly two columns
        if len(df.columns) != 2:
            st.error("Please upload a CSV file with exactly two columns.")
            st.write("No data available for prediction.")
            return None
        else:
            # Display the DataFrame
            df.columns = ["ds", "y"]
            df["ds"] = df.apply(lambda x: change_date_format(x.ds,'%Y-%m-%d'), axis=1)
            # df['ds'] = pd.to_datetime(df['ds'])# .dt.to_timestamp()
            # df['ds'] = df['ds'].dt.strftime('%Y-%m-28')
            # df['ds'] = pd.to_datetime(df['ds'])
            st.write("## CSV File Contents")
            st.write(df)
            clustering(df)
            forcast_next_year(df)
            return df
    return None


    

def forcast_next_year(df):
    model, last_date = fit_data(df)
    future = get_next_12_months(last_date)
    predict(model, future)


# Define function to create navigation bar
def navigation_bar():
    # Define navigation items
    pages = {
        "Prediction": upload_csv,
        # "Prediction": home,   
    }

    # Display navigation bar
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute selected page function
    df = pages[selection]()
    return df
    # if df is not None and selection == "Prediction":
    #     home(df)

# Main function to run the app
def main():
    st.title("Energy Consumptions")
    st.write("# Forcasting Energy Consumption")
    st.write("Uplod your households Energy Consumption (prefered if it is above 5 years)")

    # Create navigation bar
    # navigation_bar()
    navigation_bar()
    # if df is not None:
    #     home(df)

if __name__ == "__main__":
    main()
