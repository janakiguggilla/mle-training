# Use an official Python image as the base image
FROM python:3.6

# Set the working directory
WORKDIR /dist

# Copy the .whl file into the image
COPY housing_price_janaki-0.0.1-py3-none-any.whl .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install the required packages
#RUN pip install argparse sklearn pandas pickle mlflow numpy scipy.stats tarfile urllib.request
RUN pip install argparse==1.4.0 \
    scikit-learn==0.23.1 \
    pandas==1.0.3 \
    cloudpickle==2.2.1 \
    mlflow==1.7.2 \
    numpy==1.18.5 \
    scipy==1.4.1 \
   # tarfile==20.0 \
    urllib3==1.25.10

# Install the .whl file
RUN pip install housing_price_janaki-0.0.1-py3-none-any.whl

# Set the environment variables
ENV PYTHONPATH "${PYTHONPATH}:/dist"

# Set the default command to run when the container starts
CMD ["python3", "-m", "unittest", "tests.unit_tests"]

