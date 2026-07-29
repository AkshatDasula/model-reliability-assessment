import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import datetime
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle
import json
from utils import one_hot, create_and_save_embeddings, get_helper_csv_nlp, get_glcm_csv, prep_and_predict, CV_data_quality, get_text_scores_df
from utils import get_results, create_ollama_model, get_baseline_statistics, validity_check
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title='Add Model', layout='wide')

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


model_type = st.sidebar.selectbox("Select type of model to add:", options=['Regression', 'Classification', "Natural Language Processing (NLP)", "Computer Vision (CV)"])
st.markdown(f"<p><h2>{model_type}</h2></p>", unsafe_allow_html=True)

st.write("---")

# with open("./pages/benchmark.json", "r") as af:
#     benchmark_metrics = json.load(af)

with open("./pages/models.json", "r") as m:
    models = json.load(m)

# with open("./pages/output.json", "r") as o:
#     outputs = json.load(o)
    
with open("./pages/results.json", "r") as r:
    results = json.load(r)
    
with open("./pages/domain_knowledge.json", "r") as d:
    domain_dict = json.load(d)

with open("./pages/model_info.json", "r") as e:
    model_info = json.load(e)


def get_probs(data, model, scaler=None):
    
    if "Unnamed: 0" in data.columns:
        data = data.drop('Unnamed: 0', axis=1)    
    if 'target' in data.columns:
        actual_data = data.drop('target', axis=1)
    
    if model_type == "Natural Language Processing (NLP)":
        transformed = scaler.transform(data['text'])
        y_probs = model.predict_proba(transformed)
        maxs = np.max(y_probs, axis=1)
        data['probs'] = maxs
        return data
    else:
        one_h = one_hot(actual_data)
        if scaler != None:
            one_h = scaler.transform(one_h)
        y_probs = model.predict_proba(one_h)
        maxs = np.max(y_probs, axis=1)
        data['probs'] = maxs
        return data


def Regression_Classification():
    st.subheader("Upload Model files and Baseline Data")
    model_name = st.text_input("Enter Model Name", placeholder="Eg: Medical Cost Prediction")
    baseline_data, target_name = st.columns(2)
    columns_b = []
    with baseline_data:
        baseline_file = st.file_uploader("Upload Baseline Data:", type=['csv','xlsx'])
        
        if (baseline_file !=  None) and 'csv' in baseline_file.name.split('.'):
            baseline = pd.read_csv(baseline_file)
            columns_b = baseline.columns
            
        if (baseline_file !=  None) and 'xlsx' in baseline_file.name.split('.'):
            baseline = pd.read_excel(baseline_file)
            columns_b = baseline.columns
            
    with target_name:
        target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b)
    
    
    scaler_col, model_col = st.columns(2)
    
    with scaler_col: 
        scaler_file = st.file_uploader("Upload Scaler if used:", type=['joblib']) 
    
    with model_col:
        model_file = st.file_uploader("Upload Model:", type=['joblib', 'pkl', 'h5'])
    
        # baseline = baseline.rename(columns={target:'target'})
        # st.dataframe(baseline)
    
    domain_info = st.text_area("Enter Domain Information:", placeholder="Eg: Medical Cost Prediction model for predicting the cost of medical treatment for patients")
    
    st.subheader("Enter Benchmark metrics:")
    
    if model_type == "Regression":
        
        r2_bench, rmse_bench, mae_bench, mape_bench = st.columns(4)
        with r2_bench:
            r2 = st.number_input("R2 Score")
        with rmse_bench:
            rmse = st.number_input("Root Mean Squared Error")
        with mae_bench:
            mae = st.number_input("Mean Absolute Error")
        with mape_bench:
            mape = st.number_input("Mean Percentage Error")
        
        new_bench = {"R2 Score": r2, "Root Mean Squared Error": rmse, "Mean Absolute Error": mae, "Mean Absolute Percentage Error": mape}

        
    if model_type == "Classification":
        
        f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with roc_bench:
            roc = st.number_input("ROC")
        with accuracy_bench:
            accuracy = st.number_input("Accuracy")
        with fpr_bench:
            fpr = st.number_input("False Positive Rate")
        
        new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}
            
    if st.button("Submit"):
        
        model_info[model_name] = dict()
        
        os.mkdir(f'./pages/models/{model_name}')
        print("Model Directory Created...")
        
        baseline = baseline.rename(columns={target:'target'})
        baseline.to_csv(f"./pages/models/{model_name}/baseline.csv", index=False)
        print("Baseline saved...")
        
        
        if scaler_file != None:
            scaler = joblib.load(scaler_file)
            joblib.dump(scaler, f"./pages/models/{model_name}/scaler.joblib")
            print("Scaler file saved...")
        
        if 'pkl' in model_file.name.split('.'):
            model = pickle.load(model_file)
            pickle.dump(model, f"./pages/models/{model_name}/model.pkl")
        if 'joblib' in model_file.name.split('.'):
            model = joblib.load(model_file)
            joblib.dump(model, f"./pages/models/{model_name}/model.joblib")
        if 'h5' in model_file.name.split('.'):
            model = load_model(model_file)
            model.save(f"./pages/models/{model_name}/model.h5")
        
        # if model_type == "Regression":
        #     baseline_one = one_hot(baseline.drop('target', axis=1))
        #     transformed_array = scaler.transform(baseline_one) if scaler_file != None else baseline_one
        #     transformed = pd.DataFrame(transformed_array, columns=baseline_one.columns)
        #     explainer = shap.Explainer(model)
        #     shap_values_baseline = explainer(transformed)
            
        #     joblib.dump(shap_values_baseline, f"./pages/models/{model_name}/baseline_shap.pkl")
            
        # if model_type == "Classification":
            
        #     baseline_one = one_hot(baseline.drop('target', axis=1))
        #     transformed_array = scaler.transform(baseline_one) if scaler_file != None else baseline_one
        #     transformed = pd.DataFrame(transformed_array, columns=baseline_one.columns)
        #     explainer = shap.Explainer(lambda x: model.predict_proba(x), transformed)
        #     shap_values_baseline = explainer(transformed)
            
        #     joblib.dump(shap_values_baseline, f"./pages/models/{model_name}/baseline_shap.pkl")    
        
        # print("Baseline SHAP values saved...")
            
        # scaler = joblib.load(f"./pages/models/{model_name}/scaler.joblib") if scaler_file != None else None
        
        print("Model file saved...")
            
        model_info[model_name]['benchmark_metrics'] = new_bench
        # with open("./pages/benchmark.json", "w") as f:
        #     json.dump(benchmark_metrics, f)
        
        print("Benchmark metrics saved...")

        models[model_type].append(model_name)
        with open("./pages/models.json", "w") as mf:
            json.dump(models, mf)
        
        os.makedirs(f'./pages/models/{model_name}/Ground Truths')
        os.makedirs(f'./pages/models/{model_name}/Production Runs')
        
        print("Directories for Ground Truths and Production Runs created...")

        model_info[model_name]['output_type'] = model_type
        # with open("./pages/output.json", "w") as f:
        #     json.dump(outputs, f)
        
        domain_dict[model_name] = domain_info
        with open("./pages/domain_knowledge.json", "w") as domain:
            json.dump(domain_dict, domain)
        print("Domain knowledge saved...")
        
        baseline_stats = get_baseline_statistics(baseline_data=baseline, model_type=model_type, model_name=model_name)
        model_info[model_name]['type'] = model_type
        model_info[model_name]['target'] = target
        model_info[model_name]['domain_knowledge'] = domain_info
        model_info[model_name]['baseline_statistics'] = baseline_stats[model_name]
        print("Baseline statistics saved...")

        # status = create_ollama_model(model_name=model_name, model_type=model_type, domain_knowledge=domain_info, baseline_stats=baseline_stats)
        # print(f"Ollama model created...{status}")
        # except:
        #     st.error("Failed to create Ollama model")
        
        with open("./pages/model_info.json", "w") as mf:
            json.dump(model_info, mf)
        
        # results[model_name] = dict()
        # with open("./pages/results.json", "w") as r:
        #     json.dump(results, r)
        # print("Results JSON saved...")
        # os.makedirs(f"./pages/models/{model_name}/Production SHAPS/")
        # print("Directory for Production SHAPs created...")
        
        print("Model info saved...")
        print("Model added successfully...")
        st.success("Model added successfully...")
        
            
    st.write("---")
    
    st.subheader("Upload Production Data")
    
    name, date_p = st.columns(2)
    with name:
        model_name = st.selectbox("Select Model Name", options=models[model_type])
    with date_p:
        date = st.date_input("Enter Date of production run", datetime.date.today())
    # model_name = st.selectbox("Select Model Name", options=models[model_type])
    
    upload_gt, upload_p = st.columns(2)
    with upload_gt:
        gt_file = st.file_uploader("Upload Ground Truth file:", type=["csv",'xlsx'])
        
        if (gt_file !=  None) and 'csv' in gt_file.name.split('.'):
            gt = pd.read_csv(gt_file)
            # columns_p = gt.columns
            
        if (gt_file !=  None) and 'xlsx' in gt_file.name.split('.'):
            gt = pd.read_excel(gt_file)
            # columns_p = gt.columns
            
            
    with upload_p:
        prod_file = st.file_uploader("Upload Production file:", type=["csv",'xlsx'])
        
        if (prod_file !=  None) and 'csv' in prod_file.name.split('.'):
            prod = pd.read_csv(prod_file)
            
        if (prod_file !=  None) and 'xlsx' in prod_file.name.split('.'):
            prod = pd.read_excel(prod_file)
            
    
    # date_p, target_col = st.columns(2)
    # with date_p:
    #     date = st.date_input("Enter Date of production run", datetime.date(2023, 12, 12))
    # with target_col:
    #     p_target = st.selectbox("Select Target Label (Y) for the model:", options=columns_p, key="prod")
    
    
    if st.button("Upload run"):
        
        benchmark_metrics = model_info[model_name]['benchmark_metrics']
        print("Benchmark metrics loaded...")
        
        print("Production Run submitted...")
        p_target = model_info[model_name]['target']
        gt = gt.rename(columns={p_target:'target'})
        prod = prod.rename(columns={p_target:'target'})
        
        print("Target column renamed...")
        
        baseline = pd.read_csv(f"./pages/models/{model_name}/baseline.csv")
        print("Baseline loaded...")
        
        # cat_has_int, miss_dict, score, indices, outliers_index, num_invalid = validity_check(baseline, prod)
        
        # if len(indices) > 0:
        #     for_shap = prod.drop(indices, axis=0)
        #     print("Invalid rows dropped for SHAP...")
        
        
        
        if model_type == "Classification":
            
            dirs = os.listdir(f"./pages/models/{model_name}")
            
            if 'scaler.joblib' in dirs:
                scaler = joblib.load(f"./pages/models/{model_name}/scaler.joblib")
            else:
                scaler = None
            if 'model.joblib' in dirs:
                model = joblib.load(f"./pages/models/{model_name}/model.joblib")
            if 'model.pkl' in dirs:
                model = pickle.load(f"./pages/models/{model_name}/model.pkl")
            if 'model.h5' in dirs:
                model = load_model(f"./pages/models/{model_name}/model.h5")
            
            
            # baseline_one = one_hot(baseline.drop('target', axis=1))
            # prod_one = one_hot(for_shap.drop('target', axis=1))
            # prod_one = prod_one[baseline_one.columns].dropna()
            # p_transformed_array = scaler.transform(prod_one) if scaler_file != None else prod_one
            # p_transformed = pd.DataFrame(p_transformed_array, columns=prod_one.columns)
            # p_explainer = shap.Explainer(lambda x: model.predict_proba(x), p_transformed)
            # shap_values_production = p_explainer(p_transformed)
            
            
            
            # prod = get_probs(prod, model, scaler)
            # print("Probabilities calculated...")
            
        # if model_type == "Regression":
            
        #    baseline_one = one_hot(baseline.drop('target', axis=1))
        #    prod_one = one_hot(for_shap.drop('target', axis=1))
        #    prod_one = prod_one[baseline_one.columns].dropna()
        #    p_transformed_array = scaler.transform(prod_one) if scaler_file != None else prod_one
        #    p_transformed = pd.DataFrame(p_transformed_array, columns=prod_one.columns)
        #    p_explainer = shap.Explainer(model)
        #    shap_values_production = p_explainer(p_transformed)
            
        
        # joblib.dump(shap_values_production, f"./pages/models/{model_name}/Production SHAPs/{date}.pkl")
        # print("Production SHAP values saved...")
            
        
        new_results = get_results(model_name=model_name, model_type=model_type, ground_truth=gt, production_data=prod.drop('probs', axis=1) if (model_type == 'Classification') and ('probs' in prod.columns) else prod, date=date, baseline_data=baseline, benchmark_metrics=benchmark_metrics)
        print("Results calculated...")
        
        if model_name not in list(results.keys()):
            results[model_name] = dict()

        results[model_name].update(new_results[model_name]) 
        
        # results.update(new_results)
        with open("./pages/results.json", "w") as r:
            json.dump(results, r)
        print("Results saved...")
        
        gt.to_csv(f"./pages/models/{model_name}/Ground Truths/{date}.csv", index=False)
        prod.to_csv(f"./pages/models/{model_name}/Production Runs/{date}.csv", index=False)  
        print("Ground Truths and Production Runs saved...")     


def NLP():
    
    st.subheader("Upload Model files and Baseline Data")
    
    model_name_col, output_type_col = st.columns(2)
    
    with model_name_col:
        model_name = st.text_input("Enter Model Name", placeholder="Eg: Medical Cost Prediction", key="NLP_name")
    with output_type_col:
        output = st.selectbox("Select task:", options=['Classification', 'Text'])
    
    baseline_data, text_name = st.columns(2)
    columns_b = []
    
    with baseline_data:
        baseline_file = st.file_uploader("Upload Baseline Data:", type=['csv','xlsx'], key="NLP_baseline")
        
        if (baseline_file !=  None) and 'csv' in baseline_file.name.split('.'):
            baseline = pd.read_csv(baseline_file)
            columns_b = baseline.columns
            
        if (baseline_file !=  None) and 'xlsx' in baseline_file.name.split('.'):
            baseline = pd.read_excel(baseline_file)
            columns_b = baseline.columns
            
    
    with text_name:
        text = st.selectbox("Select Text Column for the model:", options=columns_b, key="NLP_text")
    
    if output == "Text":
        target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b,  key="NLP_target")
    else:
        p_target_col, model_col = st.columns(2)
        
        target = st.selectbox("Select Target Label (Y) for the model:", options=columns_b,  key="NLP_target")
        with p_target_col:
            vectorizer_file = st.file_uploader("Upload vectorizer:", type=['joblib'])
        
        with model_col:
            model_file = st.file_uploader("Upload Model:", type=['joblib', 'pkl', 'h5'], key="NLP_model")
    
    domain_info = st.text_area("Enter Domain Information:", placeholder="Eg: Medical Cost Prediction model for predicting the cost of medical treatment for patients")
    
    st.subheader("Enter Benchmark Metrics:")
    if output == "Classification":
        
        f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with roc_bench:
            roc = st.number_input("ROC")
        with accuracy_bench:
            accuracy = st.number_input("Accuracy")
        with fpr_bench:
            fpr = st.number_input("False Positive Rate")
        
        new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}
    
    if output == "Text":
        
        f1_bench, precision_bench, recall_bench, bleu_bench, sim_bench = st.columns(5)
        
        with f1_bench:
            f1 = st.number_input("F1 Score")
        with precision_bench:
            precision = st.number_input("Precision")
        with recall_bench:
            recall = st.number_input("Recall")
        with bleu_bench:
            bleu = st.number_input("BLEU")
        with sim_bench:
            sim = st.number_input("Similarity") 

        new_bench = {"Similarity": sim, "F1": f1, "Recall": recall, "Precision": precision, "BLEU": bleu}
        
    if st.button("Submit", key="NLP_button"):
        
        
        
        os.mkdir(f'./pages/models/{model_name}')
        baseline = baseline.rename(columns={target:'target', text:'text'})
        baseline.to_csv(f"./pages/models/{model_name}/baseline.csv", index=False)
        
        if output != "Text":
            
            if vectorizer_file != None:
                vectorizer = joblib.load(vectorizer_file)
                joblib.dump(vectorizer, f"./pages/models/{model_name}/vectorizer.joblib")
            if 'pkl' in model_file.name.split('.'):
                model = pickle.load(model_file)
                pickle.dump(model, f"./pages/models/{model_name}/model.pkl")
            if 'joblib' in model_file.name.split('.'):
                model = joblib.load(model_file)
                joblib.dump(model, f"./pages/models/{model_name}/model.joblib")
            # if 'h5' in model_file.name.split('.'):
            #     model = load_model(model_file)
            #     model.save(f"./pages/models/{model_name}/model.h5")
            
            if 'h5' in model_file.name.split('.'):
            # st.write(model_cv)
            
                with open(f"./pages/models/{model_name}/model.h5", "wb") as f:
                    f.write(model_file.getbuffer())
        
            print("Vectorizer and Model file saved...")
            
        
        
        # benchmark_metrics[model_name] = new_bench
        # with open("./pages/benchmark.json", "w") as f:
        #     json.dump(benchmark_metrics, f)
        
        
        os.makedirs(f'./pages/models/{model_name}/Ground Truths')
        os.makedirs(f'./pages/models/{model_name}/Production Runs')
        os.makedirs(f'./pages/models/{model_name}/embeds')
        os.makedirs(f'./pages/models/{model_name}/helper')
        
        print("Directories for Ground Truths and Production Runs created...")
        
        if output == "Text":
            os.makedirs(f'./pages/models/{model_name}/embeds_target/Ground Truths')
            os.makedirs(f'./pages/models/{model_name}/helper_target/Ground Truths')
            os.makedirs(f'./pages/models/{model_name}/embeds_target/Production Runs')
            os.makedirs(f'./pages/models/{model_name}/helper_target/Production Runs')
            
            print("Directories for Text target created...")
        
        
        create_and_save_embeddings(baseline, 'text', f'./pages/models/{model_name}/embeds/baseline')
        print("Baseline embeddings created and saved...")
        
        helper_df = get_helper_csv_nlp(baseline)
        helper_df.to_csv(f'./pages/models/{model_name}/helper/baseline.csv', index=False)
        helper_df['target'] = baseline['target']
        print("Baseline helper csv created and saved...")
        
        baseline_stats = get_baseline_statistics(baseline_data= helper_df, model_type=model_type, model_name=model_name,
                        output_type=output, target_file=None if output == "Classification" else get_helper_csv_nlp(baseline, 'text'))
        
        print("Baseline statistics calculated...")
        
        # try:
        #     status = create_ollama_model(model_name=model_name, model_type=model_type, domain_knowledge=domain_info, baseline_stats=baseline_stats, output_type=output)
        #     print(f"Ollama model created...{status}")
        # except:
        #     print("Failed to create Ollama model")
        
        # outputs[model_name] = output
        # with open("./pages/output.json", "w") as f:
        #     json.dump(outputs, f)

        model_info[model_name] = dict()
        model_info[model_name]['type'] = model_type
        model_info[model_name]['target'] = target
        model_info[model_name]['domain_knowledge'] = domain_info
        model_info[model_name]['benchmark_metrics'] = new_bench
        model_info[model_name]['output_type'] = output
        model_info[model_name]['text_column'] = text
        model_info[model_name]['baseline_statistics'] = baseline_stats[model_name]
        
        domain_dict[model_name] = domain_info

        models[model_type].append(model_name)
        with open("./pages/models.json", "w") as mf:
            json.dump(models, mf)
        print("Models JSON saved...")
        
        with open("./pages/model_info.json", "w") as mf:
            json.dump(model_info, mf)
        print("Model info saved...")
        
        with open("./pages/domain_knowledge.json", "w") as domain:
            json.dump(domain_dict, domain)
        print("Domain knowledge saved...")
        
        # results[model_name] = dict()
        # with open("./pages/results.json", "w") as r:
        #     json.dump(results, r)
        # print("Results JSON saved...")
        
        print("Model added successfully...")
            
   
    st.write("---")
    columns_p = []
    st.subheader("Upload Production Data")
    
    name_, date_p = st.columns(2)
    with name_:
        model_name = st.selectbox("Select Model Name", options=models[model_type])
    with date_p:
        date = st.date_input("Enter Date of production run", datetime.date.today())
    
    upload_gt, upload_p = st.columns(2)
    with upload_gt:
        gt_file = st.file_uploader("Upload Ground Truth file:", type=["csv",'xlsx'])
        
        if (gt_file !=  None) and 'csv' in gt_file.name.split('.'):
            gt = pd.read_csv(gt_file)
            columns_p = gt.columns
            
        if (gt_file !=  None) and 'xlsx' in gt_file.name.split('.'):
            gt = pd.read_excel(gt_file)
            columns_p = gt.columns
            
            
    with upload_p:
        prod_file = st.file_uploader("Upload Production file:", type=["csv",'xlsx'])
        
        if (prod_file !=  None) and 'csv' in prod_file.name.split('.'):
            prod = pd.read_csv(prod_file)
            
        if (prod_file !=  None) and 'xlsx' in prod_file.name.split('.'):
            prod = pd.read_excel(prod_file)
            
    
    # text_name, target_col = st.columns(2)
    # with text_name:
    #     text_col = st.selectbox("Select Text column",options=columns_p, key="textprod")
    # with target_col:
    #     p_target = st.selectbox("Select Target Label (Y) for the model:", options=columns_p, key="prod")
    
    
    if st.button("Upload run"):
        
        output = model_info[model_name]['output_type']
        text_col = model_info[model_name]['text_column']
        p_target = model_info[model_name]['target']
        
        benchmark_metrics = model_info[model_name]['benchmark_metrics']
        print("Benchmark metrics loaded...")
        
        baseline = pd.read_csv(f"./pages/models/{model_name}/baseline.csv")
        print("Baseline loaded...")
        
        gt = gt.rename(columns={p_target:'target', text_col:'text'})
        prod = prod.rename(columns={p_target:'target', text_col:'text'})
        
        print("Text and Target column renamed...")
    
            
        dirs = os.listdir(f"./pages/models/{model_name}")
        
        if output != "Text":
            if 'vectorizer.joblib' in dirs:
                vectorizer = joblib.load(f"./pages/models/{model_name}/vectorizer.joblib")
            else:
                vectorizer = None

            if 'model.joblib' in dirs:
                model = joblib.load(f"./pages/models/{model_name}/model.joblib")
            if 'model.pkl' in dirs:
                model = pickle.load(f"./pages/models/{model_name}/model.pkl")
            if 'model.h5' in dirs:
                model = load_model(f"./pages/models/{model_name}/model.h5")
            
            print("Model and Vectorizer loaded...")
    
            prod = get_probs(prod, model, vectorizer)
            print("Probabilities calculated...")
        
            gt.to_csv(f"./pages/models/{model_name}/Ground Truths/{date}.csv", index=False)
            prod.to_csv(f"./pages/models/{model_name}/Production Runs/{date}.csv", index=False)
            
            print("Ground Truths and Production Runs saved...")
            
            create_and_save_embeddings(prod, 'text', f'./pages/models/{model_name}/embeds/{date}')
            helper_df = get_helper_csv_nlp(prod)
            helper_df.to_csv(f'./pages/models/{model_name}/helper/{date}.csv', index=False)
            
            print("Production embeddings created and saved...")
            
            new_prod = prod.drop("embeds", axis=1) if 'embeds' in prod.columns else prod
            
            new_results = get_results(model_name, date, gt, new_prod.drop("probs", axis=1), baseline, model_type, benchmark_metrics)
            print("Results calculated...")
        
        if output == "Text":
            
            gt.to_csv(f"./pages/models/{model_name}/Ground Truths/{date}.csv", index=False)
            prod.to_csv(f"./pages/models/{model_name}/Production Runs/{date}.csv", index=False)
            
            print("Ground Truths and Production Runs saved...")
            
            create_and_save_embeddings(prod, 'text', f'./pages/models/{model_name}/embeds/{date}')
            helper_df_text = get_helper_csv_nlp(prod)
            helper_df_text.to_csv(f'./pages/models/{model_name}/helper/{date}.csv', index=False)
            
            print("Production embeddings created and saved...")
            
            create_and_save_embeddings(gt, 'target', f'./pages/models/{model_name}/embeds_target/Ground Truths/{date}')
            helper_df_target_gt = get_helper_csv_nlp(gt, 'target', f'./pages/models/{model_name}/embeds/{date}.csv', f'./pages/models/{model_name}/embeds_target/Ground Truths/{date}.csv')
            
            helper_df_target_gt.to_csv(f'./pages/models/{model_name}/helper_target/Ground Truths/{date}.csv', index=False)
            
            create_and_save_embeddings(prod, 'target', f'./pages/models/{model_name}/embeds_target/Production Runs/{date}')
            helper_df = get_helper_csv_nlp(prod, 'target')
            
            scores_df = get_text_scores_df(gt, prod, pd.read_csv(f'./pages/models/{model_name}/embeds_target/Ground Truths/{date}.csv'), pd.read_csv(f'./pages/models/{model_name}/embeds_target/Production Runs/{date}.csv'), pd.read_csv(f'./pages/models/{model_name}/embeds/{date}.csv'))
            
            combined_df = pd.concat([helper_df, scores_df], axis=1)
            combined_df.to_csv(f'./pages/models/{model_name}/helper_target/Production Runs/{date}.csv', index=False)
            
            new_prod = prod.drop("embeds", axis=1) if 'embeds' in prod.columns else prod
            new_results = get_results(model_name, date, gt, new_prod, baseline, "Text", benchmark_metrics)
            print("Results calculated...")
            
            
            
            print("Helper csv for target created and saved...")
        
        if model_name not in list(results.keys()):
            results[model_name] = dict()

        results[model_name].update(new_results[model_name]) 
        print("Results updated...")
        
        with open("./pages/results.json", "w") as r:
            json.dump(results, r)  
        
        print("Production run uploaded successfully...")
    

def CV():

    st.warning("Please upload images for one class at a time. Eg: Upload all images for 'Normal' class, click on upload images, repeat for 'Pneumonia' class.")
    st.subheader("Upload Model files and Baseline Data")
    
    model_name, label_name = st.columns(2)

    with model_name:
        name = st.text_input("Enter Model Name:", placeholder="X-Ray Classification")

    with label_name:
        label = st.text_input("Enter class name:")
    
    domain_info = st.text_area("Enter Domain Information:", placeholder="Eg: Medical Cost Prediction model for predicting the cost of medical treatment for patients")
    
    model_, images_col = st.columns(2)
    with model_:
        model_cv = st.file_uploader("Upload Model:", type=['pkl', 'h5'])
        
    baseline_dir = f"./pages/models/{name}/Baseline/"
    
    with images_col:
    
        with st.form("my-form", clear_on_submit=True):
            baseline_images = st.file_uploader("Upload baseline Images", type=['jpg','jpeg'], accept_multiple_files=True, key=f'Baseline')
            if st.form_submit_button("Upload Images"):
                os.makedirs(f"{baseline_dir}/{label}")
                for i in baseline_images:
                    image = Image.open(i)
                    image.save(f"{baseline_dir}/{label}/{i.name}")
                
        
    st.subheader("Enter Benchmark metrics:")
    f1_bench, precision_bench, recall_bench, roc_bench, accuracy_bench, fpr_bench = st.columns(6)

    with f1_bench:
        f1 = st.number_input("F1 Score")
    with precision_bench:
        precision = st.number_input("Precision")
    with recall_bench:
        recall = st.number_input("Recall")
    with roc_bench:
        roc = st.number_input("ROC")
    with accuracy_bench:
        accuracy = st.number_input("Accuracy")
    with fpr_bench:
        fpr = st.number_input("False Positive Rate")
    
    new_bench = {"Accuracy": accuracy, "F1": f1, "Recall": recall, "Precision": precision, "ROC_AUC": roc, "False Positive Rate": fpr}

    # upload_image, model_ = st.columns(2)
    # with upload_image:
    #     with st.form("my-form", clear_on_submit=True):
    #         baseline_images = st.file_uploader("Upload baseline Images", type=['jpg','jpeg'], accept_multiple_files=True, key=f'Baseline')
    #         if st.form_submit_button("Upload Images"):
    #             os.makedirs(f"{baseline_dir}/{label}")
    #             for i in baseline_images:
    #                 image = Image.open(i)
    #                 image.save(f"{baseline_dir}/{label}/{i.name}")
    # with model_:
    #     model_cv = st.file_uploader("Upload Model:", type=['pkl', 'h5'])
    
    st.write("---")
    st.write("Please click on submit ONLY after you are done uploading all images")


    if st.button("Submit"):
        
        print("Model Submitted...")
        print("Directories Created...")
        
        baseline_paths, baseline_labels = [], []
        baseline_lists = os.listdir(baseline_dir)

        for i in baseline_lists:
            for j in os.listdir(f"./pages/models/{name}/Baseline/{i}"):
                baseline_paths.append(f"./pages/models/{name}/Baseline/{i}/{j}")
                baseline_labels.append(i)

        baseline_paths_df = pd.DataFrame({"paths": baseline_paths,
                                        "target": baseline_labels})
        
        print("Paths df created...")
        
        baseline_paths_df.to_csv(f"{baseline_dir}/baseline_paths_df.csv", index=False)

        baseline_glcm = get_glcm_csv(baseline_paths_df, [2], [np.pi/2])
        print("GLCM df created...")
        
        baseline_quality = CV_data_quality(baseline_glcm).drop("resolution", axis=1)
        baseline_quality['Anomaly'] = baseline_quality['Anomaly'].astype('object')
        print("Baseline Quality df created...")
        
        baseline_quality.to_csv(f"./pages/models/{name}/baseline.csv", index=False)
        print("Baseline df saved...")
        
        os.makedirs(f'./pages/models/{name}/Ground Truths')
        os.makedirs(f'./pages/models/{name}/Production Runs')
        print("Directories for Ground Truths and Production Runs created...")

        if 'pkl' in model_cv.name.split('.'):
            model = pickle.load(model_cv)
            pickle.dump(model, f"./pages/models/{name}/model.pkl")
        
        if 'h5' in model_cv.name.split('.'):
            # st.write(model_cv)
            
            with open(f"./pages/models/{name}/model.h5", "wb") as f:
                f.write(model_cv.getbuffer())
            # model = load_model("model.h5")
            # model.save(f"{model_name}_model.h5")
        
        print("Model file saved...")
        
        model_info[name] = dict()
        model_info[name]['benchmark_metrics'] = new_bench
        print("Benchmark metrics saved...")
        
        model_info[name]['output_type'] = 'Classification'
        model_info[name]['type'] = model_type
        model_info[name]['target'] = 'target'
        model_info[name]['domain_knowledge'] = domain_info
        
        print("Model type, output type, domain knowledge and target saved...")
        
        baseline_stats = get_baseline_statistics(baseline_data=baseline_quality.drop('paths',axis=1), model_type=model_type, model_name=name)
        model_info[name]['baseline_statistics'] = baseline_stats[name]
        print("Baseline statistics saved...")
        
        models[model_type].append(name)
        
        domain_dict[name] = domain_info
        print("Domain knowledge saved...")
        with open("./pages/domain_knowledge.json", "w") as domain:
            json.dump(domain_dict, domain)
        
        with open("./pages/model_info.json", "w") as mf:
            json.dump(model_info, mf)
        
        print("Model info saved...")
        
        with open("./pages/models.json", "w") as mf:
            json.dump(models, mf)
        
        print("Models updated...")
        
        # print("Creating Ollama model...")
        # try:
        #     status = create_ollama_model(model_name=model_name, model_type=model_type, domain_knowledge=domain_info, baseline_stats=baseline_stats)
        #     print(f"Ollama model created...{status}")
        # except:
        #     print("Failed to create Ollama model")
        
        print("Model added successfully...")
        st.success("Model added successfully...")
    
    
    st.write("---")
    st.subheader("Upload Production Data")
    model_name, label_name = st.columns(2)

    with model_name:
        name = st.selectbox("Select Model Name:", options=models[model_type])

    with label_name:
        label = st.text_input("Enter class name:", key="prod")
    

    date_cv, upload_image_p = st.columns(2)
    with date_cv:
        date = st.date_input("Enter Date of production run", datetime.date.today())

    production_dir = f"./pages/models/{name}/Production/{date}"

    with upload_image_p:
        with st.form("my-form-p", clear_on_submit=True):
            production_images = st.file_uploader("Upload Production Images", type=['jpg','jpeg'], accept_multiple_files=True, key=f'Production')
            if st.form_submit_button("Upload Images"):
                os.makedirs(f"{production_dir}/{label}")
                for i in production_images:
                    image = Image.open(i)
                    image.save(f"{production_dir}/{label}/{i.name}")
    
    print("Images Saved...")
    
    st.write("---")
    st.write("Please click on submit after you are done uploading the images")


    if st.button("Submit", key="CV_prod"):

        production_paths, production_labels = [], []
        production_lists = os.listdir(production_dir)

        benchmark_metrics = model_info[name]['benchmark_metrics']
        print("Benchmark metrics loaded...")       
        
        baseline = pd.read_csv(f"./pages/models/{name}/baseline.csv") 
        print("Baseline loaded...")
        
        for i in production_lists:
            for j in os.listdir(f"./pages/models/{name}/Production/{date}/{i}"):
                production_paths.append(f"./pages/models/{name}/Production/{date}/{i}/{j}")
                production_labels.append(i)

        production_paths_df = pd.DataFrame({"paths": production_paths,
                                        "target": production_labels})
        print("Paths df created...")
        
        data_quality_prod = CV_data_quality(production_paths_df)
        print("Data Quality df created...")

        date_glcm = get_glcm_csv(data_quality_prod, [2], [np.pi/2])
        date_glcm['Anomaly'] = date_glcm['Anomaly'].astype('object')
        print("GLCM df created...")
        
        model = load_model(f"./pages/models/{name}/model.h5")
        print("Model loaded...")
        
        production_run_csv = prep_and_predict(model, date_glcm.drop("target", axis=1), production_lists)
        baseline['Anomaly'] = baseline['Anomaly'].astype('object')
        production_run_csv['Anomaly'] = production_run_csv['Anomaly'].astype('object')
        print("Predictions made...")
        
        new_results = get_results(model_name=name, model_type=model_type, ground_truth=date_glcm.drop(["paths", "resolution"], axis=1), 
                                  production_data=production_run_csv.drop(["probs", "paths", "resolution"], axis=1), 
                              date=date, baseline_data=baseline.drop("paths", axis=1), benchmark_metrics=benchmark_metrics, df=production_paths_df)
        
        print("Results calculated...")
        
        if name not in list(results.keys()):
            results[name] = dict()

        results[name][str(date)] = new_results[name][str(date)]
        
        # results.update(new_results)
        with open("./pages/results.json", "w") as r:
            json.dump(results, r)
        print("Results saved...")
        
        
        date_glcm.to_csv(f"./pages/models/{name}/Ground Truths/{date}.csv", index=False)
        print("Ground Truths saved...")
        
        production_paths_df.to_csv(f"{production_dir}/production_paths_df.csv", index=False)
        print("Production paths df saved...") 
        
        production_run_csv.to_csv(f"./pages/models/{name}/Production Runs/{date}.csv", index=False)
        print("Production Run df saved...")

        
        # Determine Ground Truth files and Production Runs


if model_type == "Regression":
    Regression_Classification()
elif model_type == "Classification":
    Regression_Classification()
elif model_type == "Natural Language Processing (NLP)":
    NLP()
elif model_type == "Computer Vision (CV)":
    CV()

try:
    
    st.sidebar.markdown(f"<h2>Welcome, {st.session_state['username']}</h2>",unsafe_allow_html=True)
    st.sidebar.write(f"ID: {st.session_state['id']+1}")
    st.sidebar.write(f"Occupation: {st.session_state['occupation']}")
    st.sidebar.write(f"Email: {st.session_state['email']}")
    logout = st.sidebar.button('Logout')
    if logout and len(st.session_state) != 0:
        st.session_state.clear()
        switch_page("Analysis_Page")
    elif logout and len(st.session_state) == 0:
        st.sidebar.error("Please login first")

except:
    pass

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
