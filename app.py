import gradio as gr
import joblib

# --- 1. Load the saved model and vectorizer ---
# We do this once when the app starts up.
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('sentiment_model.pkl')
except FileNotFoundError:
    # This is a fallback for if someone runs the app without the model files
    print("Error: Model files not found. Please run the analysis.ipynb notebook first to train and save the model.")
    # We can exit or handle this gracefully. For now, we'll print and let it error on the next line.


# --- 2. Define the prediction function ---
# This is the exact same function from your notebook!
def predict_sentiment(text):
    """
    Takes a text string and predicts its sentiment using our loaded model.
    """
    # Vectorize the input text using the loaded vectorizer
    vectorized_text = vectorizer.transform([text])
    
    # Use the loaded model to predict
    prediction = model.predict(vectorized_text)
    
    # Convert the numerical prediction back to a readable label
    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"

# --- 3. Create and launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_sentiment,       # The function to wrap the UI around
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review here..."), # Input component
    outputs="text",             # Output component
    title="IMDb Movie Review Sentiment Analyzer",
    description="This app analyzes a movie review and predicts whether the sentiment is Positive or Negative. Built with Scikit-learn and Gradio."
)

# Launch the app!
iface.launch()