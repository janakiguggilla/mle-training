# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python3 < nonstandardcode.py >

Command to activate environment 
conda activate mle-dev

command to create env.yml file
conda env create -f env.yml

## to install black and isort
pip install black
pip install isort

## to run python file using black and isort
black < nonstandardcode.py >
isort < nonstandardcode.py >

## installing flake8 and checking the python file with flake8
pip install flake8

flake8 --max-linelength=88 nonstandardcode.py

## to build docker container
first create Dockeerfile
and build container using===>  docker build -t image name

## to run the image
docker run --name containername imagename

##push docker image
docker push imagename
##pulling
docker pull imagename

