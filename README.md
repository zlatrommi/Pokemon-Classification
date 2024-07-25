# Pokemon-Classification

 Project Description >  This project's goal is to take in visual data of a pokemon and be able to identify what pokemon it is using an AI learning module

![image](https://github.com/user-attachments/assets/37195418-dd28-436e-bbc7-ea0301876d9a)

## The Algorithm
- For my project, I decided to train an  AI or Artificial Intelligence to classify different Pokemon. Classification AI classifies different images into predefined classes or catgeories based on data that is given to it. I decided to do 150 Pokemon because doing more then that would require a larger computing device. I obtained a Pokemon Database from Kaggle (https://www.kaggle.com/datasets/lantian773030/pokemonclassification) Which had 7000 photos of 150 different Pokemon. I input these images into my learning module on the Jetson Nano. I then trained the model to classify each image into it's correct class.
- Training a model involved feeding the correctly labeled data into the chosen algorithm. The algorithm learns from the data to create a model that, in my case, correctly classifies different images of different Pokemon. This process involves adjusting the model to minimize the errors.
- After the AI is trained, we test it. The model's performance is evaluated using a separate set of data called the test images (test). Once trained and evaluated, the model can be used to make predictions on images it's never seen before. The input features are fed into the model, which then outputs the predicted class based on what it's already learned. To run my re-trained ResNet-18 model for testing I needed to convert it into ONNX format. ONNX is an open model format that supports many popular machine learning frameworks, and it simplifies the process of sharing models between tools.
- In conclusion, the algorithm teaches computers to sort things into the correct classes by showing them lots of examples. It learns from these examples and then makes predictions about new things based on what it learned. In my project, the AI classified different Pokemon.
- For example, we originally retrained a model that classifed cats and dogs.
1. First we would show the AI many cats and dogs, telling it which is which.
2. Then, we let the AI learn from these examples, by training it.
3. Finally, we test the AI with new pictures of cats and dogs it hasn't seen yet to see if it can correctly identify and classify them.
## Running this project

1. Add steps for running this project.
2. Make sure to include any required libraries that need to be installed for your project to run.

[View a video explanation here](video link)
