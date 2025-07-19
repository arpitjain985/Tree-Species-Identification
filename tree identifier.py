import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

class TreeSpeciesIdentifier:
    def __init__(self, model_path='models/tree_species_model.h5'):
        self.model = load_model(model_path)
        self.class_names = np.load('models/class_names.npy', allow_pickle=True)
        self.img_size = (224, 224)
        
    def preprocess_image(self, img_path):
        # Load and preprocess image
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    
    def predict_species(self, img_path):
        # Make prediction
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.class_names[predicted_class], confidence
    
    def predict_from_camera(self):
        # Capture image from camera and predict
        cap = cv2.VideoCapture(0)
        
        print("Press 'c' to capture image, 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow('Tree Species Identifier', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Save captured image temporarily
                temp_path = 'temp_capture.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Predict species
                species, confidence = self.predict_species(temp_path)
                print(f"Predicted species: {species} (Confidence: {confidence:.2%})")
                
                # Display prediction on the image
                frame = cv2.putText(
                    frame, 
                    f"{species} ({confidence:.0%})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                cv2.imshow('Result', frame)
                cv2.waitKey(2000)  # Show result for 2 seconds
                
                # Remove temporary file
                os.remove(temp_path)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    identifier = TreeSpeciesIdentifier()
    
    print("Tree Species Identification System")
    print("1. Identify from image file")
    print("2. Identify from camera capture")
    
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        img_path = input("Enter image path: ")
        species, confidence = identifier.predict_species(img_path)
        print(f"\nPredicted species: {species}")
        print(f"Confidence: {confidence:.2%}")
    elif choice == '2':
        identifier.predict_from_camera()
    else:
        print("Invalid choice")
