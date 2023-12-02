# ComputerVision_TeamProject
<br>
## Introduction
<img width="1318" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/47feca60-dcb3-44c6-806b-c2ab0c70b93a">

## Dataset
#### CUB_200_2011_repackage_class50
The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a widely utilized dataset for fine-grained visual categorization tasks, specifically designed for bird species recognition. It comprises 11,788 images across 200 bird subcategories. The dataset is split into 5,994 images for training and 5,794 for testing. Each image is meticulously annotated with details such as a subcategory label, 15 part locations, 312 binary attributes, and a bounding box. Additionally, it includes textual information in the form of ten single-sentence descriptions for each image, sourced through Amazon Mechanical Turk, with a requirement of at least 10 words without revealing subcategories and actions​​.[인용](https://paperswithcode.com/dataset/cub-200-2011)

## Method
#### Data Augmentation
**rotation**: Rotates the image 30 degrees.
##### Color: brightness is a value between 0.5 and 0.8, contrast is a value between 0 and 1, saturation is a value between 0.5 and 0.8, and hue is a value between 0 and 0.5. Random values ​​are continuously changed and applied.
horizontal_flip: Flips the image horizontally.
zoom: Fix the image to 448x448 size and then enlarge it.
crop: Randomly resize and crop to 448x448 size.
random: Apply one of the above techniques to the image randomly.

After applying various augmentation techniques to the original image, learning was performed by creating a combination of the original image and various techniques.
<img width="897" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/0878080c-1988-4bf6-b9ad-513c15a06114">


The summer-galaxy line in the chart above: In the first attempt, only one augmentation technique was applied to the original image and the model was trained only on this data, showing the lowest accuracy and highest loss value. 
<img width="900" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/6f95070b-8d8a-4dd9-828b-45ed98a557ff">


##### **After modification,**
<img width="902" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/24602342-aa66-405e-be54-a04227f52e4e">

We continued to change the training conditions in a direction that increased the loss to the smallest and accuracy to the highest.
As a result of augmentation by mixing as many different combinations as possible, 
<img width="629" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/96a0c7ec-2520-424f-8ae2-9e830e0e20e9">

It was confirmed that the highest accuracy was output in these four cases.
However, in the case of the random color augmentation method used in number 1, if training was performed after fixing the seed value, accuracy was lowered, so it was not included in the next training.

In each of the three cases, after going through the hyperparameter tuning process, through a random search process, the parameter combination that yields the best accuracy value is batch size=32, learning rate=0.1, optimizer=sgd, epoch=20, and momentum=0 (default). I was able to confirm. 
<img width="583" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/1167a3d2-d9c7-4408-b26c-f737e11b8e1f">

Among these augmentation methods, the original image that resulted in the highest accuracy and the dataset obtained by rotating, flipping, cropping, and zooming the images are used, respectively.
The parameters are the final training result using batch size=32, learning rate=0.1, optimizer=sgd, epoch=20, momentum=0 (default).
<img width="586" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/47dbe7a3-c6d3-490b-86f6-ff3274707720">

<img width="601" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/aa3a0e8d-d766-4311-80cb-10300466cbc3">
(The reason there are up to 40 steps is because the codes with and without augmentation were recorded simultaneously, so 2 steps were recorded per epoch.)

#### HyperParameter tuning
<img width="584" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/541ffe53-db4e-4bfe-ba7d-b5b1df0d5f83">
Selects the best accuracy from each params set and outputs the params with the best accuracy among several accuracies. -> This params set must be applied to the test.

(Hyperparameters were randomly selected and augmented data was tried 10 times.)
Reason for randomly selecting hyperparameter: Because there are a large number of cases that can be sampled, execution time takes a long time. -> In order to select the optimal value within a given time considering the running time, the parameter with the highest accuracy among the randomly rotated parameters is selected as the final parameter and used for test data.

##### Learning rate
<img width="590" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/8aad9c8b-e9c6-4be3-9740-9db8f22530f5">

After fixing other values, the learning rate values ​​were categorized into four and compared,
The validation accuracy value of learning rate 0.1 was the highest.

optimizer
<img width="585" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/59afcd8e-5ebf-4d0c-b839-c14b4e7a5149">

As a result of performing the same experiment by changing the optimizer to adamw, When comparing the values ​​of validation_accuracy and test_accuracy with SGD, much lower values ​​are obtained.
As a result of performing the same experiment by changing the optimizer to adam, Getting bad results.(Results show that Adam lags far behind SGD in generalization, including momentum, in some tasks, especially computer vision tasks.)
->Therefore, SGD has the most optimal results.
- result
As a result of searching for the best parameter combination through random search, The best results were obtained at batch size=32 and learning rate=0.1.
Therefore, after fixing the values ​​(the epoch is fixed to 10 due to GPU issues), Performing by composing combinations of different parameters.

augmentation=rotation+flip+zoom
batch size=32, learning rate=0.1, optimizer=sgd, epoch=10, momentum=0, scheduler=stepLR(step_size=10, gamma=0.1)
<img width="520" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/47a605b4-9fcd-4665-9419-2d5898ecac1e">

batch size=32, learning rate=0.1, optimizer=sgd, epoch=10, momentum=0.9, scheduler=stepLR(step_size=10, gamma=0.1)
<img width="515" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/e6a4f6d2-cf12-4c42-8bfb-9ca24fb519ab">

#### Using Resnet 50
<img width="827" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/fae4cf62-f690-4026-86f8-d7fda964a631">

#### Compared Resnet50-B with C, D
<img width="1084" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/abf9c40b-4547-41fa-8e35-95b76a057e12">
<img width="1065" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/4c575eae-d6ef-4a67-a403-854f8a22cd0d">

## Experiments
<img width="702" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/76bc4cbd-fbdc-4466-92c8-a0cf77fa24ee">

### Setting Optimizers
Default Optimizer : SGD
More Optimizer : Adam
Small data, in Computer Vision task 👉 Adam is far behind in generalization compared to SGD. Therefore, SGD has the most optimal result value.

## Conclusion
#### Last Epoch & Test Accuracy
<img width="1134" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/c2093089-0bc2-4de2-91ff-8529b04040da">

#### Compare the grad cam results from our trained model with the original image and reference code.
<img width="465" alt="image" src="https://github.com/Chae0510/ComputerVision_TeamProject/assets/85086390/41bcd300-b8c2-401a-b6c1-2a15c5445645">
the appearance of paying more attention to the characteristic parts of the bird.

After data augmentation, good model selection, and hyperparameter tuning, the model can be seen focusing on objects after zero focus on birds.

Focus on the bird's surroundings too
That bird is "Downy Woodpecker"
     --> Often hanging from trees is also a feature of birds, 
     so you can see the model is well focused.

## Reference
- [resnet 변형](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
- [위 논문 설명](https://bo-10000.tistory.com/133)
- [resnet](https://paperswithcode.com/model/resnet-d?variant=resnet34d)
- [resnet 구조](https://wjunsea.tistory.com/99)
- [resnet 18 설명](https://hnsuk.tistory.com/31)
- https://ropiens.tistory.com/32
- https://www.kaggle.com/code/a4885534/mlrc2022-re-hist
- https://dl.acm.org/doi/pdf/10.1145/3581783.3612165
