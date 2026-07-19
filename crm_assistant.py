import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

CRM_FILE = "crm_database.csv"

def initialize_crm():
    """Creates a local mock CRM database file if it does not already exist."""
    if not os.path.exists(CRM_FILE):
        data = {
            "Customer_ID": [101, 102, 103, 104],
            "Customer_Name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"],
            "Email": ["alice@example.com", "bob@example.com", "charlie@example.com", "diana@example.com"],
            "Support_Tickets_Month": [1, 5, 0, 4],
            "Months_Active": [12, 3, 24, 6],
            "Last_Interaction_Summary": ["Initial onboarding smooth.", "Complained about UI bugs.", "Renewed annual contract.", "Requested refund details."],
            "Next_Action": ["Check in next month", "Follow up on bug fix", "Send holiday card", "Escalate to manager"],
            "Churn_Risk": ["Low", "High", "Low", "High"]
        }
        df = pd.DataFrame(data)
        df.to_csv(CRM_FILE, index=False)
        print("[CRM System] New CRM database initialized.")
    else:
        print("[CRM System] Existing CRM database loaded.")

print("Loading AI Models (this may take a minute on the first run)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def train_churn_model():
    """Trains a Random Forest classifier using mock historical metrics to predict churn risk."""
    X_train = [[0, 24], [1, 12], [5, 3], [4, 6], [1, 36], [6, 2], [0, 8], [3, 5]]
    y_train = [0, 0, 1, 1, 0, 1, 0, 1]  # 0 = Retained, 1 = Churned
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

churn_predictor = train_churn_model()

def analyze_and_process_call(customer_id, call_transcript):
    """Executes the AI CRM pipeline: Summarize, Suggest Actions, Predict Churn, Draft Email, Update CRM."""
    print(f"\n--- Processing Call for Customer ID: {customer_id} ---")
    
    summary_out = summarizer(call_transcript, max_length=45, min_length=10, do_sample=False)
    summary = summary_out[0]['summary_text']
    print(f"[Summary]: {summary}")
    
    sentiment = classifier(call_transcript)[0]['label']
    if sentiment == "NEGATIVE":
        next_action = "Schedule urgent retention call with account manager."
    else:
        next_action = "Send standard feature update list and request review."
    print(f"[Suggested Next Action]: {next_action}")
    
    df = pd.read_csv(CRM_FILE)
    if customer_id in df['Customer_ID'].values:
        cust_row = df[df['Customer_ID'] == customer_id].iloc[0]
        tickets = int(cust_row['Support_Tickets_Month'])
        months = int(cust_row['Months_Active'])
        
        churn_prob = churn_predictor.predict([[tickets, months]])[0]
        churn_status = "High" if (churn_prob == 1 or sentiment == "NEGATIVE") else "Low"
        print(f"[Predicted Churn Risk]: {churn_status}")
        
        email_draft = f"""
        Subject: Following up on our recent conversation
        Hi {cust_row['Customer_Name']},
        
        Thank you for speaking with us today. We noted that: "{summary}"
        We are actively working on your account. { "We want to make things right." if churn_status == "High" else "Thank you for being a valued partner!" }
        
        Best regards,
        Your Account Team
        """
        print(f"[Generated Follow-up Email]:\n{email_draft}")
        
        df.loc[df['Customer_ID'] == customer_id, 'Last_Interaction_Summary'] = summary
        df.loc[df['Customer_ID'] == customer_id, 'Next_Action'] = next_action
        df.loc[df['Customer_ID'] == customer_id, 'Churn_Risk'] = churn_status
        df.to_csv(CRM_FILE, index=False)
        print("[CRM System] CRM updated successfully!")
        
    else:
        print(f"[Error] Customer ID {customer_id} not found in the database.")

if __name__ == "__main__":
    initialize_crm()
    
    print("\n--- Current CRM Database State ---")
    print(pd.read_csv(CRM_FILE)[['Customer_ID', 'Customer_Name', 'Churn_Risk', 'Next_Action']])
    
    bob_transcript = (
        "Hello, this is Bob. I am incredibly frustrated with the recent platform updates. "
        "The dashboard keeps crashing every time I try to export my monthly reports. "
        "If this isn't fixed by Friday, I will be forced to cancel my subscription and move to a competitor."
    )
    
    analyze_and_process_call(customer_id=102, call_transcript=bob_transcript)
    
    print("\n--- Updated CRM Database State ---")
    print(pd.read_csv(CRM_FILE)[['Customer_ID', 'Customer_Name', 'Churn_Risk', 'Next_Action']])