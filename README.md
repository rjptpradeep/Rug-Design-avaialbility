# Rug-Design-avaialbility
Rug Desisgn availability
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Load the pre-trained VGG16 model
# model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# # Define the feature extraction function
# def get_feature_vector(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features = model.predict(x)
#     return features.flatten()

# # Folder path containing images
# folder_path = r'G:\Data Science\Shade Card'

# # Loop through all images in the folder and extract features
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # filter image files
#         img_path = os.path.join(folder_path, filename)
#         features = get_feature_vector(img_path)
#         print(f"Features for {filename}:\n", features)
# import pandas as pd

# # Assuming 'data' is a list of lists: [filename, feature1, feature2, ..., feature512]
# # Example:
# # data = [
# #     ['image1.jpg', 0.012, 0.234, ..., 0.876],
# #     ['image2.jpg', 0.134, 0.453, ..., 0.921],
# #     ...
# # ]

# # Create column names
# columns = ['filename'] + [f'feat_{i}' for i in range(len(data[0]) - 1)]

# # Create DataFrame
# df = pd.DataFrame(data, columns=columns)

# # Save to CSV
# df.to_csv('vgg16_image_features.csv', index=False)
# print("✅ Features saved successfully to vgg16_image_features.csv")


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import os

# # Example image path (change filename as needed)
# img_path = r'G:\Data Science\Shade Card\image1.jpg'

# # Load and show the image
# img = mpimg.imread(img_path)
# plt.imshow(img)
# plt.axis('off')  # Hide axes
# plt.title("Preview of image1.jpg")
# plt.show()


#_____________________________________________________
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import os

# # Load the pre-trained VGG16 model
# model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# # Define the feature extraction function
# def get_feature_vector(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features = model.predict(x)
#     return features.flatten()

# # Folder path containing images
# folder_path = r'G:\Data Science\Shade Card'

# # Initialize data list to store filename + features
# data = []

# # Loop through all images in the folder and extract features
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # filter image files
#         img_path = os.path.join(folder_path, filename)
#         features = get_feature_vector(img_path)
#         data.append([filename] + features.tolist())  # Store filename and feature vector
#         print(f"Features for {filename} extracted.")

# # Save the features to CSV
# if data:  # only if data is not empty
#     columns = ['filename'] + [f'feat_{i}' for i in range(len(data[0]) - 1)]
#     df = pd.DataFrame(data, columns=columns)
#     df.to_csv('vgg16_image_features.csv', index=False)
#     print("✅ Features saved successfully to vgg16_image_features.csv")
# else:
#     print("⚠️ No features to save. No images were processed.")

# # Display one of the images (example)
# img_path = r'G:\Data Science\Shade Card - Copy\abcd.jpg'  # Make sure this file exists!
# if os.path.exists(img_path):
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
#     plt.axis('off')  # Hide axes
#     plt.title("Preview of abcd.jpg")
#     plt.show()
# else:
#     print("⚠️ The image to display was not found.")


# if __name__ == "__main__":
#     # 👉 Only run this part if testing image display
#     img_path = r'G:\Data Science\Shade Card - Copy\abcd.jpg'
#     display_image(img_path)

#     # 👉 If needed later, uncomment the below section
#     # for filename in os.listdir(folder_path):
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Feature extraction from single image
def get_feature_vector(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Extract features from folder
def extract_features_from_folder(folder_path, output_csv='vgg16_image_features.csv'):
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            try:
                features = get_feature_vector(img_path)
                data.append([filename] + features.tolist())
                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    if data:
        columns = ['filename'] + [f'feat_{i}' for i in range(len(data[0]) - 1)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Features saved successfully to {output_csv}")
    else:
        print("⚠️ No valid image files were processed.")

# Find the closest match image
def find_closest_match(new_img_path, feature_csv='vgg16_image_features.csv'):
    if not os.path.exists(feature_csv):
        print("⚠️ Feature file not found. Please extract features first.")
        return

    df = pd.read_csv(feature_csv)
    new_features = get_feature_vector(new_img_path).reshape(1, -1)
    feature_vectors = df.iloc[:, 1:].values
    similarities = cosine_similarity(new_features, feature_vectors)[0]

    most_similar_idx = np.argmax(similarities)
    most_similar_filename = df.iloc[most_similar_idx]['filename']
    similarity_score = similarities[most_similar_idx]

    print(f"\n✅ Closest match: {most_similar_filename}")
    print(f"📊 Similarity score: {similarity_score:.4f}")

    matched_img_path = os.path.join(os.path.dirname(new_img_path), most_similar_filename)
    display_image(matched_img_path)

# Display an image
def display_image(img_path):
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Preview: {os.path.basename(img_path)}")
        plt.show()
    else:
        print("⚠️ The image to display was not found.")

# Main menu
def main():
    while True:
        print("\n====== MENU ======")
        print("1. Extract features from image folder")
        print("2. Display an image")
        print("3. Find closest match to a new image")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            folder = input("Enter full folder path (e.g., G:\\Data Science\\Shade Card):\n")
            extract_features_from_folder(folder)
        elif choice == '2':
            img_path = input("Enter full image path to preview:\n")
            display_image(img_path)
        elif choice == '3':
            img_path = input("Enter full path of the new image:\n")
            find_closest_match(img_path)
        elif choice == '0':
            print("👋 Exiting. Have a great day!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()

