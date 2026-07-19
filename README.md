# AI-CRM-Assistant
AI-powered CRM assistant that automates customer call analysis using NLP and machine learning. Summarizes conversations, detects sentiment, predicts churn risk, generates personalized follow-up emails, and updates CRM records automatically using Hugging Face Transformers and Scikit-learn.


## AI CRM Assistant

An AI-powered Customer Relationship Management (CRM) assistant that automates customer interaction analysis using **Natural Language Processing (NLP)** and **Machine Learning**.

The application processes customer call transcripts, generates concise summaries, analyzes customer sentiment, predicts churn risk, drafts personalized follow-up emails, and automatically updates a local CRM database.

## Features

- **Automatic Call Summarization**
  - Uses Facebook's **BART Large CNN** model to summarize long customer conversations.

- **Sentiment Analysis**
  - Detects customer sentiment (Positive/Negative) using a pre-trained DistilBERT model.

- **Customer Churn Prediction**
  - Predicts customer churn risk using a Random Forest classifier trained on mock CRM metrics.

- **Automated Follow-up Email Generation**
  - Creates personalized email drafts based on conversation summaries and churn risk.

- **Automatic CRM Updates**
  - Updates customer interaction summaries, recommended next actions, and churn risk in the CRM database.

- **Local CSV-Based CRM**
  - Uses a lightweight CSV file as a mock CRM system for demonstration purposes.

## Project Structure

```text
AI-CRM-Assistant/
│
├── crm_assistant.py      # Main application
├── crm_database.csv      # Mock CRM database
├── requirements.txt      # Project dependencies
└── README.md
```

## Technologies Used

- Python 3.x
- Pandas
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Random Forest Classifier
- BART Large CNN
- DistilBERT

## AI Models

### 1. Text Summarization

**Model**

```text
facebook/bart-large-cnn
```

**Purpose**

Summarizes customer call transcripts into concise interaction notes.

### 2. Sentiment Analysis

**Model**

```text
distilbert-base-uncased-finetuned-sst-2-english
```

**Purpose**

Determines whether the customer conversation is positive or negative.

### 3. Churn Prediction

**Algorithm**

- Random Forest Classifier

**Input Features**

- Support Tickets per Month
- Months Active

**Output**

- Low Churn Risk
- High Churn Risk

## CRM Workflow

```text
Customer Call Transcript
            │
            ▼
   AI Text Summarization
            │
            ▼
   Sentiment Analysis
            │
            ▼
  Churn Prediction Model
            │
            ▼
 Generate Follow-up Email
            │
            ▼
 Update CRM Database
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abhinav-cloud482/AI-CRM-Assistant.git
cd AI-CRM-Assistant
```

### 2. Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the application:

```bash
python crm_assistant.py
```

During the first execution, Hugging Face will automatically download the required transformer models.

## CRM Database

The application maintains a local CRM database in:

```text
crm_database.csv
```

Each customer record contains:

| Field | Description |
|--------|-------------|
| Customer_ID | Unique customer identifier |
| Customer_Name | Customer name |
| Email | Customer email |
| Support_Tickets_Month | Number of support tickets |
| Months_Active | Customer tenure |
| Last_Interaction_Summary | Latest interaction summary |
| Next_Action | Suggested follow-up action |
| Churn_Risk | Predicted churn level |

## Example Workflow

**Input Transcript**

> "The dashboard keeps crashing every time I export reports. If this isn't fixed soon, I'll cancel my subscription."

The assistant will:

- Generate a summary
- Detect negative sentiment
- Predict high churn risk
- Draft a retention email
- Update the CRM database automatically

## Sample Generated Email

```text
Subject: Following up on our recent conversation

Hi Bob Jones,

Thank you for speaking with us today.

We noted that:
"The customer reported recurring dashboard crashes while exporting reports."

We are actively working on your account.
We want to make things right.

Best regards,

Your Account Team
```

## Sample Output

```text
--- Processing Call for Customer ID: 102 ---

Summary:
The customer reported repeated dashboard crashes while exporting reports.

Suggested Next Action:
Schedule urgent retention call with account manager.

Predicted Churn Risk:
High

CRM updated successfully.
```

## Future Improvements

- REST API using FastAPI or Flask
- Integration with Salesforce or HubSpot
- Speech-to-Text support
- OpenAI/GPT-powered email generation
- Real-time analytics dashboard
- Customer segmentation
- Database support (MySQL/PostgreSQL)
- Docker deployment
- Web interface using Streamlit

## Requirements

```text
pandas
scikit-learn
transformers
torch
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Notes

- This project uses a mock CRM database stored as a CSV file.
- The Random Forest model is trained on sample data for demonstration purposes.
- Transformer models are downloaded automatically during the first run and may take several minutes depending on your internet connection.
- Internet access is required only for the initial model download.

## Contributing

Contributions, issues, and feature requests are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute it for educational and personal projects.

## Author

**Abhinav Dixit**

GitHub: https://github.com/Abhinav-cloud482

If you found this project useful, consider giving it a star on GitHub.
