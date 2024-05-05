import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from apikey import apikey 


os.environ['OPENAI_API_KEY'] = apikey 
st.title('👾 Text Summary Generator')

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

def read_file(uploaded_file):
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        return content
    return None

summary_template = PromptTemplate(
    input_variables=['text'],
    template='Give a Summary of the following text using one third of the total words used below and explain its contents : \n {text}'
)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.9)

summary_memory = ConversationBufferMemory(input_key='text', memory_key='chat_history')

summary_chain = LLMChain(llm=llm, prompt=summary_template, verbose=True, output_key='summary', memory=summary_memory)

# Main logic
if uploaded_file is not None:
    text_content = read_file(uploaded_file)
    if text_content:
        summary = summary_chain.run(text=text_content)
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Failed to read file content.")
