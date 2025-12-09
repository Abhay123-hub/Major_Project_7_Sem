

# 🏥 AI-Powered Doctor & Hospital Recommendation System

**FastAPI + LangChain + LangGraph + LLMs + Pandas**

This project is an **AI-driven hospital/doctor recommendation system** that filters doctors based on user preferences (gender, specialization, disease, consultation fees, online availability, etc.) and generates a **human-friendly explanation** using an LLM.

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── hospitals_database_modified(1).csv
│                
│
├── experiment_1.ipynb              # Jupyter notebook used for experimentation
├── modular_code.py                 # Modular version of the entire pipeline
├── app.py                          # FastAPI application
├── requirements.txt
└── README.md
```

---

## 🚀 What This System Does

### ✔ Accepts user input such as:

```json
{
  "query_dict": {
    "gender": "Male",
    "consultation_fees": 100,
    "specialization": "Cardiologist",
    "disease_related": "Heart Disease",
    "online_consultation": "No"
  }
}
```

### ✔ Filters the dataset using Pandas based on:

* Gender
* Disease type
* Specialization
* Fee range
* Online consultation availability

### ✔ Sends the filtered records to an LLM

The LLM then generates a **clear, friendly summary** for the end user.

### ✔ Returns output like:

```json
{
  "result": "Based on the filtered data, there are five doctors available at Aastha Hospital with different specializations.

Dr. Suresh Thakur is a General Surgeon available for online consultation with a fee of 1426 and can be consulted for general surgery related issues...

(etc...)"
}
```

---

## 🧠 Core Technologies Used

| Component                    | Usage                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| **FastAPI**                  | API layer to accept user queries                                                    |
| **LangChain**                | Prompting, chaining, structured output                                              |
| **LangGraph**                | Workflow orchestration for each step (gender filter, fee filter, generate_response) |
| **Pandas**                   | Data filtering & intersections                                                      |
| **Pydantic**                 | Enforcing LLM-generated JSON structure                                              |
| **LLM (OpenAI/Groq/Gemini)** | Summarizing doctor data                                                             |
| **Notebook**                 | Testing & experimentation                                                           |

---

## 🔄 System Flow

1. **User provides input** to FastAPI.
2. Input goes through **LangGraph nodes**:

   * gender filter
   * specialization filter
   * fee filter
   * disease filter
3. All filters return Pandas subsets.
4. Common rows = final filtered dataset.
5. This is sent to the **generate_response** node.
6. LLM returns structured JSON → formatted for the user.
7. FastAPI returns final response.

---

## ▶️ Running the API

### 1️⃣ Create virtual environment

```bash
python -m venv venv
```

### 2️⃣ Activate environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run FastAPI

```bash
uvicorn app:app --reload
```

### 5️⃣ Open API Docs

Go to:

```
http://127.0.0.1:8000/docs
```

You can now send JSON requests and test the system.

---

## 📊 Data Format (hospitals.csv)

Your CSV must contain:

* **hospital_name**
* **doctor_name**
* **qualification**
* **specialization**
* **doctor_phone**
* **doctor_email**
* **doctors_age**
* **yoe**
* **hospital_exact_location**
* **disease_related**
* **gender**
* **available_for_online_consultation**
* **available_slots**
* **ratings(out_of_5)**
* **consultation_fee**

---

## 🧪 Experiment Notebook

The notebook **experiment_1.ipynb** includes:

* Data cleaning
* Building filters
* Debugging the response generator
* Testing PromptTemplates
* Running LangGraph manually
* Validating Pydantic schema
* Checking errors like:

  * Missing `"response"` key
  * PromptTemplate variable errors
  * Schema mismatch
  * Structured output failures

Use this notebook when you want to test pipeline changes before putting them into the API.

---

## 🧩 modular_code.py

This file contains:

* All reusable functions
* Filter nodes
* LLM prompt generation
* LangGraph builder
* Response generator

This is your **core logic** in modular form.

---

## 🌐 app.py (FastAPI)

`app.py` exposes API endpoints:

### `POST /recommend`

Takes user filters and returns doctor summaries.

---

## 🛠 requirements.txt

Includes:

```
fastapi
uvicorn
langchain
langgraph
openai
pydantic
pandas
python-dotenv
```

Add or modify depending on the LLM provider and features you use.

---

## 🚀 Future Improvements

* Add ranking/scoring system
* Add location-based filtering
* Add embeddings + similarity search
* Add UI with Streamlit
* Add feedback loop to improve LLM summaries

---

## 🙌 Author

**Abhay**
Passionate about LLMs, AI systems, and building real-world ML applications.


