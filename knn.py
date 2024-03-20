import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = '/content/9.png'
image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)

# Reshape the image array to a 2D array (flatten it)
height, width= image_array.shape
image_2d = image_array.reshape((height * width, 1))

# Perform k-means clustering to create clusters for the pixels
num_clusters = 5  # You can adjust this parameter based on the desired number of segments
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(image_2d)
labels = kmeans.labels_

# Train a kNN classifier on the clustered data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(image_2d, labels)

# Predict labels for each pixel in the image using kNN
predicted_labels = knn.predict(image_2d)

# Reshape the predicted labels back to the original image shape
segmented_image = predicted_labels.reshape((height, width))

# Visualize the segmented image
plt.imshow(segmented_image, cmap='viridis')  # You can choose a colormap based on your preference
plt.axis('off')  # Turn off axis labels
plt.show()
