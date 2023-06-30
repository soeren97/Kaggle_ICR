## Setup enviroment

Create a conda enviroment and activate it

* conda create -n ICR python=3.10.9
* conda activate ICR

Install pip and required packages. If GPU is available write it in the brackets otherwise write CPU
(GPU not working yet)

* conda install .[GPU]

Activate pre-commit hooks 

* pre-commit install

The enviroment is now ready to be used.