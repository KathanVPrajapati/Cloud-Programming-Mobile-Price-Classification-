# Mobile Price Classification

## Project Overview
This project uses AWS SageMaker to classify mobile prices based on features such as RAM, Battery, etc. The system leverages a machine learning model built using XGBoost and deployed using AWS services like Lambda and DynamoDB for data storage.

## Technologies Used
- **AWS SageMaker**: For model training and deployment
- **Lambda**: For serverless functions
- **DynamoDB**: For NoSQL database storage
- **XGBoost**: For machine learning model

## Project Structure
- `notebooks/`: Jupyter notebooks for data analysis and model training
- `scripts/`: Python scripts used for backend and deployment
- `lambda_function.py`: The Lambda function used for event-driven data processing
- `README.md`: This file
- `requirements.txt`: List of Python dependencies

## How to Run the Code

### Prerequisites
- Python installed on your local system
- AWS CLI configured with appropriate credentials
- Install the required dependencies using:
  ```bash
  pip install -r requirements.txt
  
### [Link to the Webpage](https://mobileprice-284279361159.us-central1.run.app/)
