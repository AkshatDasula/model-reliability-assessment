import streamlit as st
import json
from streamlit_extras.switch_page_button import switch_page
import os
from utils import get_fishbone, get_prompt, separated_results, determine_dtype_ft, get_ollama_template, create_drift_indicator
import ollama
import pandas as pd


import langchain
import langchain_ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


st.set_page_config(page_title='Landing', page_icon='👨🏻‍💻', layout='wide')
st.markdown("<h1 style='margin-top:55px;padding-top:5px;text-align: center'>A Tool for Drift Detection, Root-Cause Analysis, Visualization and Explainability for Deployed ML Models</h1>", unsafe_allow_html=True)

no_sidebar_style = """
<style>
[data-testid="stSidebarHeader"] {display: none;}
</style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

no_sidebar_style_ = """
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.markdown(no_sidebar_style_, unsafe_allow_html=True)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


with open("./pages/models.json", "r") as m:
    model_dict = json.load(m)

with open("./pages/results.json", "r") as res:
    results = json.load(res)

with open("./pages/model_info.json", "r") as o:
    model_info = json.load(o)
# with open("./pages/results.json", "r") as results:
#     results_dict = json.load(results)    
  
  
model_type = st.sidebar.selectbox("Select Model Type:", options=list(model_dict.keys()))
model_name = st.sidebar.selectbox("Select Model:", options=model_dict[model_type])


# production_runs = {"Cardiovascular Disease Prediction": sorted([(datetime.today() - timedelta(days=7)*i).strftime('%Y-%m-%d') for i in range(1,4)]),
#                    "Medical Cost Prediction": sorted([(datetime(2023, 8, 9) - timedelta(days=7)*i).strftime('%Y-%m-%d') for i in range(1,4)])}
if model_name in os.listdir(f"./pages/models/"):
    production_runs_ = os.listdir(f"./pages/models/{model_name}/Production Runs")
    
    # dates = [datetime.strptime(i[:-4], '%dth %B %Y') for i in production_runs_]
    production_run = st.sidebar.selectbox("Select Production Run:", options=sorted([i[:-4] for  i in production_runs_]))
    # if output_dict[model_name] == "Text":
    #     analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis"]
    # else:
    #     analysis = ["Performance Drift Analysis", "Data Drift Analysis", "Data Quality Analysis", "Prediction Drift Analysis", "Model Explanations/Interpretability"]
    # analysis_type = st.sidebar.selectbox("Select Analysis Type:",options=analysis)
    
    st.session_state['model_name'] = model_name
    st.session_state['model_type'] = model_type
    st.session_state['production_date'] = production_run
    st.session_state['all_production_runs'] = production_runs_


else:
    st.write("###")
    st.write("No such model")
    exit(0)
    

if st.sidebar.button("Upload Model/Production Run"):
    switch_page("add model")
 
 
   
results_for_run = results[model_name][production_run]
benchmark_metrics = model_info[model_name]['benchmark_metrics']

# st.markdown(f"<p><h3>Fishbone for {model_name} of {production_run} run:</h3></p>", unsafe_allow_html=True)
# # st.write(results_for_run)

drift_legend = create_drift_indicator(model_name=model_name, model_type=model_type, production_run_date=production_run)
st.plotly_chart(drift_legend, use_container_width=True)

st.plotly_chart(get_fishbone(model_type, results_for_run, model_name, production_run,benchmark_metrics), 
                use_container_width=True)

if st.button("Go to Analysis Page"):
    switch_page("analysis page")


llm_name = "_".join(model_name.lower().split())
st.write(f"<h4>LLM Name: {llm_name}_LLM</h4>", unsafe_allow_html=True)

cat_ft, num_ft = determine_dtype_ft(pd.read_csv(f"./pages/models/{model_name}/baseline.csv").drop("target", axis=1))
prod_summary, performance_dict, data_drift, data_quality, prediction_drift = separated_results(results,
                                                                                            model_name,
                                                                                              production_run,
                                                                                             num_ft, cat_ft, model_type)

prompt_col, analysis_col = st.columns(2) 

with prompt_col:
    input_prompt = st.text_input("Prompt your model:")
with analysis_col:
    analysis_type = st.selectbox("Prompt Related to:", options=["Input Features Related", "Output Features/Performance Related"])

template = get_ollama_template(model_type)
    
    
prompt = ChatPromptTemplate.from_template(template)
llm = OllamaLLM(model="llama3.2")
    

input_prompt = get_prompt(input_prompt, production_run, prod_summary, performance_dict, data_drift, data_quality, prediction_drift, analysis_type)
# st.write(input_prompt)
    
param_dict = {"model_name": model_name,
                                    "model_type": model_type,
                                    "domain_knowledge": model_info[model_name]['domain_knowledge'],
                                    "baseline_stats": model_info[model_name]['baseline_statistics'],
                                    "llm_name": llm_name,
                                    "human_prompt": input_prompt,
                                    "output_type": None if model_info[model_name]['output_type'] != "Text" else model_info[model_name]['output_type']}


if st.button("Generate Response"):
    
    st.markdown(f"<h4>Response:</h4>", unsafe_allow_html=True)

    with st.spinner('Generating response...'):
            
            # response = ollama.chat(model=f'{llm_name}_mistralLLM', messages=[
            #     {
            #         'role': 'user',
            #         'content': prompt,
            #     },
            #     ])

        chain = prompt | llm
        response = chain.invoke(param_dict)
        st.write(response)
        
# st.write(f"LLM not available")

        

    



    # st.markdown(f"<p><h2>{analysis_type}</h2></p>", unsafe_allow_html=True)
    # st.write("---")