# Pokemon-Classification

 Project Description >  This project's goal is to take in visual data of a pokemon and be able to identify what pokemon it is using an AI learning module

![image](https://github.com/user-attachments/assets/37195418-dd28-436e-bbc7-ea0301876d9a)

## The Algorithm
- For my project, I decided to train an  AI or Artificial Intelligence to classify different Pokemon. Classification AI categorizes different images into predefined classes or catgeories based on data that is given to it. I decided to do 150 Pokemon because doing more then that would require a larger computing device. I obtained a Pokemon database from Kaggle 
![image](https://github.com/user-attachments/assets/3040bcb4-8c48-4301-8db6-6d2a400e8071)
 (https://www.kaggle.com/datasets/lantian773030/pokemonclassification) which had 7000 photos of 150 different Pokemon. I inputted these images into my learning module on the Jetson Nano. I then trained the model to classify each image into it's correct class.
- Training a model involved feeding the correctly labeled data into the chosen algorithm. The algorithm learns from the data to create a model that, in my case, correctly classifies different images of different Pokemon. This process involves adjusting the model to minimize the errors.
- After the AI is trained, we test it. The model's performance is evaluated using a separate set of data called the test images (test). Once trained and evaluated, the model can be used to make predictions on images it's never seen before. The input features are fed into the model, which then outputs the predicted class based on what it's already learned. To run my re-trained ResNet-18 model for testing I needed to convert it into ONNX format. ONNX is an open model format that supports many popular machine learning frameworks, and it simplifies the process of sharing models between tools.
- In conclusion, the algorithm teaches computers to sort things into the correct classes by showing them lots of examples. It learns from these examples and then makes predictions about new things based on what it learned. In my project, the AI classified different Pokemon.
- Sources https://student.idtech.com/courses/331/modules 
## Running this project
### Steps:

1. Ensure you have a Jetson Nano (from NVIDIA) and log into Python with your Jetson Interface Nano.
![image](https://github.com/user-attachments/assets/4e25b092-99c8-484f-8b1e-e0a9078ba941)

   - Use PuTTY to connect your Jetson Nano to your device. Once connected, set up Wi-Fi through your device's hotspot. Confirm the connection by running `nmcli device`.
   - Use `ifconfig wlan0` to check your IP address.

2. Connect to Visual Studio Code using your IP address. Open a new terminal in Visual Studio Code to begin coding.

3. Once everything is set up, ensure you have installed Jetson Inference and Docker Image from [Jetson Interface GitHub] https://github.com/dusty-nv/jetson-inference

4. Change directories to `jetson-inference/python/training/classification/data` and download the Pokemon dataset using the command:
`wget https://www.kaggle.com/datasets/lantian773030/pokemonclassification -O pokemon_classification.tar.gz`
`tar xvzf pokemon_classification.tar.gz`
5. Navigate back to `nvidia/jetson-inference/` and run the following command in your terminal to configure memory overcommitment: echo 1 | sudo tee /proc/sys/vm/overcommit_memory
6. In the `jetson-inference` directory, run `./docker/run.sh` to start the Docker container. Once inside the container, navigate to `jetson-inference/python/training/classification`.
7. Start the training script to re-train the network. Specify where the model should be saved and where the data is located:
python3 train.py --model-dir=models/pokemon --data-dir=data/pokemon --batch-size=32 --epochs=50
8. Export the trained network to ONNX format. While still in the Docker container and in `jetson-inference/python/training/classification`, run:
python3 onnx_export.py --model-dir=models/pokemon
Check `jetson-inference/python/training/classification/models/pokemon/` for the exported model `resnet18.onnx`.

9. Exit the Docker container (`Ctrl + D`) and navigate to `jetson-inference/python/training/classification`.

- Use `ls models/pokemon/` to ensure the model `resnet18.onnx` is on the Nano.

10. Test your AI by running the following command to classify a Pokemon image:
 ```
 imagenet.py --model=models/pokemon/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/pokemon/labels.txt data/pokemon/test/charizard.jpg charizard_result.jpg
 ```
![191](https://github.com/user-attachments/assets/581e65c1-4051-479b-ade0-8870095458fd)

[View a video explanation here](video link)
