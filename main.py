import cv2
import dlib
import knn
import math
import numpy as np
import openai
from PIL import Image
import requests
import keyboard
# Load the pre-trained facial landmark detector
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


def generate_image(emotion):

    # Set up your OpenAI API key
    openai.api_key = 'your-API-key-here'

    def generate_image_from_text(input_text):
        # Call the OpenAI API to generate an image based on the input text
        response = openai.Image.create(
            prompt=input_text
        )

        # Extract the image URL from the API response
        image_url = response['data'][0]['url']

        # Fetch the image from the URL and open it using PIL
        img = Image.open(requests.get(image_url, stream=True).raw)
        img.show()

    # Example usage
    input_text = "art express"+emotion+"ness in Van GAogh style picture"
    generate_image_from_text(input_text)



def detect_landmarks(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)

    # Detect faces in the grayscale image
    detector = dlib.get_frontal_face_detector()
    faces = detector(image_np)

    # Ensure that at least one face is detected
    if len(faces) > 0:
        # Take the first detected face
        face = faces[0]

        # Predict facial landmarks for the detected face
        landmarks = predictor(image_np, face)
        landmarksList = []

        # Draw landmarks on the image
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            coordinate_dict = {'X': x, 'Y': y}
            landmarksList.append(coordinate_dict)
            cv2.circle(image_np, (x, y), 1, (0, 255, 0), -1)

        return landmarksList
    else:
        return []


def generateVector(analyzedFaces):
    """
    Generates a feature vector representing facial landmarks distances.

    Args:
        analyzedFaces (list of dictionaries): List of dictionaries representing facial landmarks.

    Returns:
        list: A feature vector representing the distances between facial landmarks normalized by the length and width units.
    """
    facesVectors = []
    # Calculate distances between specific facial landmarks
    LeyebowL = distancePoint(analyzedFaces[37], analyzedFaces[18], 'Y')
    LeyebowM = distancePoint(analyzedFaces[37], analyzedFaces[19], 'Y')
    LeyebowR = distancePoint(analyzedFaces[38], analyzedFaces[20], 'Y')
    ReyebowL = distancePoint(analyzedFaces[25], analyzedFaces[44], 'Y')
    ReyebowM = distancePoint(analyzedFaces[24], analyzedFaces[44], 'Y')
    ReyebowR = distancePoint(analyzedFaces[23], analyzedFaces[43], 'Y')
    LeyeL = distancePoint(analyzedFaces[37], analyzedFaces[41], 'Y')
    LeyeR = distancePoint(analyzedFaces[38], analyzedFaces[40], 'Y')
    ReyeL = distancePoint(analyzedFaces[43], analyzedFaces[47], 'Y')
    ReyeR = distancePoint(analyzedFaces[44], analyzedFaces[46], 'Y')
    lengthMouseL = distancePoint(analyzedFaces[61], analyzedFaces[67], 'Y')
    lengthMouseM = distancePoint(analyzedFaces[62], analyzedFaces[66], 'Y')
    lengthMouseR = distancePoint(analyzedFaces[63], analyzedFaces[65], 'Y')
    widthMouse = distancePoint(analyzedFaces[54], analyzedFaces[48], 'X')
    lengthUnit = distancePoint(analyzedFaces[0], analyzedFaces[1], 'Y')
    widthUnit = distancePoint(analyzedFaces[0], analyzedFaces[16], 'X')

    # Normalize distances by length and width units and add them to the feature vector
    facesVectors += [LeyebowL / lengthUnit, LeyebowM / lengthUnit, LeyebowR / lengthUnit, ReyebowL / lengthUnit,
                     ReyebowM / lengthUnit, ReyebowR / lengthUnit, LeyeL / lengthUnit, LeyeR / lengthUnit,
                     ReyeL / lengthUnit, ReyeR / lengthUnit, lengthMouseL / lengthUnit, lengthMouseM / lengthUnit,
                     lengthMouseR / lengthUnit, widthMouse / widthUnit]

    return facesVectors


def distancePoint(dict1, dict2, str):
    """
    Calculates the distance between two points represented by dictionaries.

    Args:
        dict1 (dictionary): First point dictionary containing coordinates.
        dict2 (dictionary): Second point dictionary containing coordinates.
        str (str): Indicates the dimension to calculate the distance, 'X' for width and 'Y' for height.

    Returns:
        float: The distance between the two points in the specified dimension.
    """
    return math.fabs(dict2[str] - dict1[str])


def vectorDistance(vectorImage1, vectorImage2):
    """
    Calculates the Euclidean distance between two feature vectors.

    Args:
        vectorImage1 (list): Feature vector of the first image.
        vectorImage2 (list): Feature vector of the second image.

    Returns:
        float: The Euclidean distance between the two feature vectors.
    """
    sum = 0
    for i in range(len(vectorImage1) - 1):
        sum += math.pow(vectorImage1[i] - vectorImage2[i], 2)
    return math.sqrt(sum)


def main():
    # Train the data
    dataset = knn.train()

    # Test the data
    knn.test(dataset)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is accessible
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the resulting frame
        cv2.imshow('Press Space to Capture', frame)

        # Check for spacebar press
        key = cv2.waitKey(1)
        if key == 13:
            # Close the windows
            cv2.destroyAllWindows()
            break
        if key == ord(' '):
            # Capture the current frame as a photo
            captured_photo = frame.copy()

            # Display the captured photo
            cv2.imshow('Captured Photo', captured_photo)

            # Detect facial landmarks on the captured photo
            landmarksList = detect_landmarks(captured_photo)
            if landmarksList is None:
                print("Error: Failed to detect facial landmarks.")
                break

            # Generate distance vector
            distanceVec = generateVector(landmarksList)

            # Classify by KNN
            imClass = knn.knn(distanceVec, dataset)

            # Print the result
            print("You are " + imClass)

            print("do you want to generate an art image describes your face expression?\ny/n")
            key = input()
            if key=="y":
                print("generating picture...")
                # generate and display the art that describes your feeling
                # ##################### TO GENERATE AN IMAGE, UNCOMMENT THE NEXT LINE. #####################
                generate_image(imClass)
                # Display facial landmarks on the captured photo

    # Release the webcam
    cap.release()


if __name__ == "__main__":
    # Call the main function when the script is run
    main()
