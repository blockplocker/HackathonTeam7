# Hackathon Team7

## Overview

This project implements an AI-powered equipment analytics dashboard for Volvo construction equipment. The application combines data analysis, predictive modeling, and conversational AI to provide insights into machine performance, fault patterns, and maintenance recommendations. The system processes multiple data sources including ActiveCare claims, historical maintenance data, technical support cases, and Matris telemetry logs to deliver comprehensive equipment intelligence through an interactive Streamlit web interface.

## Setup Instructions

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended):
    ```
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```
        source venv/bin/activate
        ```
5. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

To run the chatbot user interface, execute the following command:

```
streamlit run app.py
```

Follow the instructions in the web interface to interact with the chatbot.
