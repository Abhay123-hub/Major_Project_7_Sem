import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict,Annotated
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.types import Command
from langchain_core.tools import tool
from pydantic import BaseModel,Field,field_validator
from typing import List,Dict
import os


load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-4",api_key = api_key)


data = pd.read_csv("data\hospitals_database_modified (1).csv")

class State(TypedDict):
    query_dict:dict
    gender_response:dict
    specialization_response:dict
    online_consultation_response:dict
    consultation_fee_response:dict
    disease_related_response:dict
    response_dict:dict
    final_data:str
    final_df:pd.DataFrame
    final_response:str
    dfs:List[pd.DataFrame]


def gender_filter(state:State)->dict:
    query_dict = state.get("query_dict",{})
    gender_response = state.get("gender_response",{})
    gender = query_dict.get("gender")
    if gender == None:
        gender_response["gender"] = None 
        
    else:
        filtered = data[data["gender"] == gender]
        gender_response["gender"] = filtered

    return {"gender_response":gender_response}     


def online_consultation_filter(state:State)->dict:
    query_dict = state.get("query_dict",{})
    
    online_consultation_response= state.get("online_consultation_response",{})
    online_consultation = query_dict.get("online_consultation")
    if online_consultation == None:
        online_consultation_response["online_consultation"] = None 
        
    else:
        filtered = data[data["available_for_online_consultation"] == online_consultation]
        online_consultation_response["online_consultation"] = filtered

    return {"online_consultation_response":online_consultation_response}


def consultation_fees_filter(state:State)->dict:
    query_dict = state.get("query_dict",{})
    consultation_fee_response = state.get("consultation_fee_response",{})
    consultation_fees = query_dict.get("consultation_fees")
    if consultation_fees == None:
        consultation_fee_response["consultation_fees"] = None 
        
    else:
        filtered = data[data["consultation_fee"] <= consultation_fees]
        consultation_fee_response["consultation_fees"] = filtered

    return {"consultation_fee_response":consultation_fee_response}


def specialization_filter(state:State)->dict:
    query_dict = state.get("query_dict",{})
    specialization_response = state.get("specialization_response",{})
    specialization = query_dict.get("specialization")
    if specialization == None:
        specialization_response["specialization"] = None 
        
    else:
        filtered = data[data["specialization"] == specialization]
        specialization_response["specialization"] = filtered

    return {"specialization_response":specialization_response}


def disease_filter(state:State)->dict:
    query_dict = state.get("query_dict",{})
    disease_related_response = state.get("disease_related_response",{})
    disease = query_dict.get("disease_related")
    if disease == None:
        disease_related_response["disease_related"] = None 
        
    else:
        filtered = data[data["disease_related"] == disease]
        disease_related_response["disease_related"] = filtered

    return {"disease_related_response":disease_related_response}


def generate_response(state:State):

    query_dict = state.get("query_dict",{})
    response_dict = state.get("response_dict",{})
    gender = state.get("gender_response").get("gender")
    online_consultation = state.get("online_consultation_response").get("online_consultation")
    consultation_fees = state.get("consultation_fee_response").get("consultation_fees")
    specialization = state.get("specialization_response").get("specialization")
    disease_related = state.get("disease_related_response").get("disease_related")
    dfs = [
    gender,
    online_consultation,
    consultation_fees,
    specialization,
    disease_related
]

    
    valid_dfs = [df for df in dfs if df is not None]
    final_df = valid_dfs[0].head()
    
    ## I am picking only the first five rows of the data 
    gender_list = final_df["gender"].to_list()
    hospital_name_list = final_df["hospital_name"].to_list()
    specialization_list = final_df["specialization"].to_list()
    consultation_fee_list = final_df["consultation_fee"].to_list()
    disease_related_list = final_df["disease_related"].to_list()
    available_for_online_consultation_list = final_df["available_for_online_consultation"].to_list()
    hospital_name = final_df["hospital_name"].to_list()
    doctor_name = final_df["doctor_name"].to_list()
    location = final_df["hospital_exact_location"].to_list()
    available_slots = final_df["avialable_slots"].to_list()

    final_data = " "

    for i in range(len(final_df)):
        response = f"Doctor Name: {doctor_name[i]}, Hospital Name: {hospital_name[i]}, Specialization: {specialization_list[i]}, Gender: {gender_list[i]}, Consultation Fee: {consultation_fee_list[i]}, Disease Related: {disease_related_list[i]}, Available for Online Consultation: {available_for_online_consultation_list[i]}, Location: {location[i]}, Available Slots: {available_slots[i]}"
        final_data += response + "\n\n"

    template = """You are an expert medical assistant. Based on the following filtered data, generate a concise and informative response for the user.
    Do not generate any response if there is no data available after filtering.Just say no doctors found matching the criteria.
    Think step by step and provide the final response.This is the filtered data: {final_data}"""



    
    
    prompt = PromptTemplate(template=template,input_variables=["final_data"])
    chain = prompt | llm
    result = chain.invoke({"final_data":final_data}).content
    return {"final_response":result,"final_data":final_data,"final_df":final_df,"dfs":dfs}




graph = StateGraph(State)
graph.add_node("gender_filter",gender_filter)
graph.add_node("online_consultation_filter",online_consultation_filter)
graph.add_node("consultation_fees_filter",consultation_fees_filter)
graph.add_node("specialization_filter",specialization_filter)
graph.add_node("disease_filter",disease_filter)
graph.add_node("generate_response",generate_response)
graph.add_edge(START,"gender_filter")
graph.add_edge(START,"online_consultation_filter")
graph.add_edge(START,"consultation_fees_filter")
graph.add_edge(START,"specialization_filter")
graph.add_edge(START,"disease_filter")
graph.add_edge("gender_filter","generate_response")
graph.add_edge("online_consultation_filter","generate_response")
graph.add_edge("consultation_fees_filter","generate_response")
graph.add_edge("specialization_filter","generate_response")
graph.add_edge("disease_filter","generate_response")
graph.add_edge("generate_response",END)

app = graph.compile()


def get_response(input:dict)->dict:
    response = app.invoke(input)
    return response["final_response"]


if __name__ == "__main__":
    inputs = {"query_dict":{"gender":"Male","consultation_fees":100,"specialization":"Cardiologist","disease_related":"Heart Disease","online_consultation":"Yes"}}


    response = get_response(inputs)
    print(response)
