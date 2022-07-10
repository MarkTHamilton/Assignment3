#from sklearn.externals
import joblib


# Load our trained model
model = joblib.load('Freedom_to_make_life_choices.pkl')

#Print statement prompting user to input values from 1-10
print("Welcome to the country happiness predictor! Type in from 0-10 how you feel your country rates.")
Score=float(input("How do you rate the your country from 0-10: "))
GDP_per_capita=float(input("How do you rate the GDP_per_capita from 0-2: "))
Social_support=float(input("How do you rate Social_support  from 0-2: "))
Healthy_life_expectancy = float(input("How do you rate the life expectancy of your country 0-2: "))
Generosity = float(input("Rate your countries generosity 0-1: "))
Perceptions_of_corruption = float(input("Rate your countries perception of corruption 0-1: "))

# Define the house that we want to value (with the values in the same order as in the training data)
freedom_1 = [
    Score,
    GDP_per_capita,
    Social_support,
    Healthy_life_expectancy,   # Healthy_life_expectancy,
    Generosity,                # Generosity
    Perceptions_of_corruption, # Perceptions_of_corruption
]
freedom = [
        freedom_1
            ]

# Make a prediction for each house in the homes array (we only have one)

freedom_values = model.predict(freedom)

# Since we are only predicting the price of one house, grab the first prediction returned
predicted_value = freedom_values[0]

# Print the results
print("Country details:")
print(f"Estimated freedom value: {predicted_value:,.2f}")

