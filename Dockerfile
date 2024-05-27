# Use the official Python image as the base image
FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Install PyTorch
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install opencv-contrib-python timm
# Copy the entire project directory
COPY . .

# Expose the port for the Flask app
EXPOSE 5000

# Set the entry point for the container
CMD ["python", "main.py"]