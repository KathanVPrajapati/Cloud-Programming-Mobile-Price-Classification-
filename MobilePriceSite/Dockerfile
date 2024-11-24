FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy Django project
COPY . .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
