# File: app.py
"""
File to render a basic streamlit app for showcasing the created Titanic model.

"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


def get_csv_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Function to simply return the csv data.

    Returns
    -------
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
        Five pandas dataframes are returned containing training and testing data.

    """
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    features = ["Pclass", "Age", "Sex", "Name", "SibSp", "Parch", "Fare"]

    X_train = train_data[features]
    y_train = train_data["Survived"]
    X_test = test_data[features]

    return train_data, test_data, X_train, y_train, X_test


def extract_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the titles from the names of passengers.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to extract names from.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the extracted titles.

    """
    copy = df.copy()
    names = copy["Name"]
    titles = names.str.extract(r' ([A-Za-z]+)\.', expand=False)

    misc = [
        title
        for title in titles
        if title not in ["Ms", "Mrs", "Miss", "Mr", "Master"]
    ]

    titles = titles.replace("Ms", "Miss")
    titles = titles.replace(misc, "Misc")

    return pd.DataFrame(titles)


def predict_info(output: pd.DataFrame) -> None:
    """
    Function to show the predicted percentage of deaths and survivors.

    Parameters
    ----------
    output : pd.DataFrame
        The dataframe containing the prediction data.

    """
    total = output["Survived"]
    dead = output.loc[output.Survived == 0]["Survived"]
    survived = output.loc[output.Survived == 1]["Survived"]

    passenger_data = {
        "% of survivors": f"{(len(survived)/len(total)):.2%}",
        "% of deaths": f"{(len(dead)/len(total)):.2%}",
    }

    st.markdown("Passenger Percentages")
    st.table(passenger_data)


def create_confusion_matrix(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    test_size: float = 0.2,
) -> None:
    """
    Function to create a confusion matrix.

    Parameters
    ----------
    X_train : pd.DataFrame
        The X_train data.
    y_train : pd.DataFrame
        The y_train data.
    test_size : float, optional
        The test size of the testing and training data split
        (defaults to 0.2, or 20% of the data will be reserved for testing).

    """
    _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=test_size)

    # Using a validation split as we discussed before
    y_predict = model.predict(X_val)
    matrix = confusion_matrix(y_val, y_predict)
    ConfusionMatrixDisplay(matrix).plot()
    st.pyplot(fig=plt)


def check_for_errors(selection: dict) -> bool:
    """
    Function to check if selection has None for required fields.

    Parameters
    ----------
    selection : dict
        The dictionary to check.

    Returns
    -------
    bool
        The boolean that is true if errors, false if no errors.

    """
    required_dict = {
        "title": "Title",
        "first_name": "First Name",
        "last_name": "Last Name",
        "p_class": "Passenger Class",
        "sibsp": "Siblings/Spouses",
        "parch": "Parents/Children",
        "age": "Age",
    }

    errors = False
    error_message = "The following fields are required:"

    for field in required_dict:
        if selection[field] in (None, ""):
            error_message += f" {required_dict[field]},"
            errors = True

    if errors is True:
        st.error(error_message.rstrip(","))

    return errors


def clean_selection(selection: dict) -> pd.DataFrame:
    """
    Function to clean the selection and prep it for the titanic model.

    Parameters
    ----------
    selection : dict
        The selection dictionary to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned dictionary in a Pandas DataFrame.

    """
    clean = {
        "Pclass": selection["p_class"],
        "Age": selection["age"],
        "Sex": selection["sex"].lower(),
        "Name": (
            f"{selection['last_name']}, {selection['title']}. "
            f"{selection['first_name']} {selection['middle_initial']}"
        ),
        "SibSp": selection["sibsp"],
        "Parch": selection["parch"],
        "Fare": selection["fare"],
    }

    return pd.DataFrame([clean])


def main() -> None:
    """
    The main function to render the streamlit widgets.

    """
    train_data, test_data, X_train, y_train, X_test = get_csv_data()

    titanic_model = joblib.load("models/titanic_model.pkl")
    grid_search_model = joblib.load("models/grid_search_model.pkl")

    st.header("Titanic Model Demo", divider="rainbow", anchor="False")

    st.space()

    tabs = st.tabs(["Model Statistics", "Self Prediction"])

    with tabs[0]:
        with st.expander("Training Data", expanded=True):
            st.dataframe(
                train_data,
                hide_index=True,
            )
        st.space()

        with st.expander("Testing Data", expanded=True):
            st.dataframe(
                test_data,
                hide_index=True,
            )
        st.space()

        if st.button("Predict!", type="primary"):
            prediction = titanic_model.predict(X_test)
            output = pd.DataFrame(
                {"PassengerId": test_data.PassengerId, "Survived": prediction},
            )
            predict_info(output)

            scores = cross_val_score(titanic_model, X_train, y_train, cv=5)

            scores_data = {
                "Average Accuracy": f"{scores.mean():.2%}",
                "Standard Deviation": f"{scores.std():.2%}",
            }
            st.markdown("Scores")
            st.table(scores_data)

            param_data = {}
            st.markdown(f"Best Parameters for Titanic Model")
            for param, value in grid_search_model.best_params_.items():
                param_data[f"{param[12:]}"] = f"{value:.2f}"
            st.table(param_data)

            with st.expander("Model Prediction Data", expanded=True):
                st.dataframe(
                    output,
                    hide_index=True,
                )

            with st.expander("Confusion Matrix", expanded=True):
                create_confusion_matrix(titanic_model, X_train=X_train, y_train=y_train)

        st.space()

    with tabs[1]:
        st.subheader("Test Yourself!", anchor=False, width="stretch")

        container = st.container(
            border=True,
            horizontal=True,
            horizontal_alignment="center",
        )

        selection = {}

        selection["title"] = container.selectbox(
            "Title",
            options=["Dr", "Mr", "Mrs", "Miss", "Ms", "Master"],
            index=None,
            placeholder="Select a title (Required)",
            width=300,
        )
        selection["first_name"] = container.text_input(
            "First Name",
            placeholder="Insert your First Name (Required)",
            width=300,
        )
        selection["middle_initial"] = container.text_input(
            "Middle Initial",
            placeholder="Insert your Middle Initial (Optional)",
            width=300,
        )
        selection["last_name"] = container.text_input(
            "Last Name",
            placeholder="Insert your Last Name (Required)",
            width=300,
        )

        p_options = {1: "First Class", 2: "Second Class", 3: "Third Class"}
        selection["p_class"] = container.selectbox(
            "Passenger Class",
            options=p_options,
            index=None,
            placeholder="Select your class (Required)",
            format_func=lambda x: p_options[x],
            width=400,
        )

        selection["sex"] = container.selectbox(
            "Sex",
            options=["Male", "Female"],
            index=None,
            placeholder="Select your prefered sex (Required)",
            width=400,
        )

        selection["age"] = container.number_input(
            "Age",
            min_value=1,
            max_value=100,
            step=1,
            placeholder="Select your age (Required)",
        )

        selection["sibsp"] = container.number_input(
            "Siblings/Spouses",
            min_value=0,
            max_value=train_data["SibSp"].max(),
            step=1,
            placeholder="Select the amount of siblings and spouses (Required)",
        )
        selection["parch"] = container.number_input(
            "Parents/Children",
            min_value=0,
            max_value=train_data["Parch"].max(),
            step=1,
            placeholder="Select the amount of parents and children (Required)",
        )
        selection["fare"] = container.slider(
            "Select the Fare amount (Optional)",
            min_value=0.0,
            max_value=train_data["Fare"].max(),
            step=0.01,
            width=1000,
        )

        if st.button("Predict Me!", type="primary", width="stretch"):
            error = check_for_errors(selection)

            if error is False:
                selection = clean_selection(selection)
                self_prediction = titanic_model.predict(selection)

                if self_prediction[0] == 1:
                    st.success("You Survived!")
                else:
                    st.warning("You did not Survive...")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Titanic Demo",
        page_icon="🚢",
        layout="wide",
    )
    col = st.columns([1, 8, 1])
    with col[1]:
        main()