import streamlit as st # type: ignore
import requests # type: ignore
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer # type: ignore
from langchain_huggingface import HuggingFacePipeline # type: ignore
from langchain.prompts import PromptTemplate # type: ignore

# Weather API settings
weather_api_key = '73e5ff9e5e82f199a767ef2ba8c1c822'  # Replace with your actual API key
weather_api_url = f'https://api.openweathermap.org/data/2.5/weather?q={{}}&appid={weather_api_key}&units=metric'

# Hugging Face model settings
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def get_weather(location):
    url = weather_api_url.format(location)
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

def summarize_text(input_text):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}\n\n\n\nSummary:",
    )
    llm = HuggingFacePipeline(pipeline=summarizer)
    formatted_prompt = prompt_template.template.format(text=input_text)
    generated_result = llm(prompt=formatted_prompt, max_length=100)
    if isinstance(generated_result, str):
        return generated_result  # If it's a string, return as is
    elif isinstance(generated_result, list) and len(generated_result) > 0:
        return generated_result[0]['generated_text']  # Access 'generated_text' from the first result in the list
    else:
        return None

# Streamlit app
def main():
    st.title("Weather Summarizer")

    location = st.text_input("Enter location (e.g., city name):")

    if st.button("Get Weather and Summarize"):
        weather_data = get_weather(location)
        if weather_data:
            weather_summary = f"Weather in {weather_data['name']} - {weather_data['weather'][0]['description']}, temperature: {weather_data['main']['temp']}°C"
            summary = summarize_text(weather_summary)
            st.subheader(weather_summary)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.error("Failed to fetch weather data. Please check the location.")

if __name__ == "__main__":
    main()
