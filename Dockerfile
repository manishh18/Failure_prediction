# # Use an official Python runtime as a parent image
# FROM python:3.11-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy requirements.txt first to leverage Docker cache
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of your application code
# COPY . .

# # Expose ports for FastAPI (8000) and Streamlit (8501)
# EXPOSE 8000
# EXPOSE 8501

# # Default command: run both FastAPI and Streamlit
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run Home.py --server.port 8501 --server.address 0.0.0.0"]


# FROM python:3.11-slim

# # Install dependencies
# RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# # Set workdir
# WORKDIR /app

# # Install Python deps
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . /app

# # Copy nginx config
# COPY nginx.conf /etc/nginx/nginx.conf

# # Expose Cloud Run port
# EXPOSE 8080

# # Start everything together
# CMD sh -c "\
#     uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
#     streamlit run Home.py --server.port 8501 --server.address 0.0.0.0 & \
#     nginx -g 'daemon off;'"


# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Install nginx
# RUN apt-get update && apt-get install -y nginx && \
#     rm /etc/nginx/sites-enabled/default && \
#     apt-get clean

# # Copy nginx config
# COPY nginx.conf /etc/nginx/nginx.conf

# # Expose Cloud Run default port
# EXPOSE 8080

# # Start FastAPI + Streamlit + nginx
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run Home.py --server.port 8501 --server.address 0.0.0.0 & nginx -g 'daemon off;'"]


FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects traffic on port 8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run Home.py --server.port 8501 --server.address 0.0.0.0"]
