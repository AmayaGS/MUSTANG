# MUSTANG
Multi Stain Graph Attention Multiple Instance Learning

<img src="pic_trulli.jpg" alt="Italian Trulli">

The MUSTANG pipeline is composed of:

• <b>A - Segmentation</b>: A automated segmentation step, where UNet is used to segment tissue areas on the WSIs. The user can use the trained weights provided on our GitHub repository or use their own.

• <b>B - Patching</b>: After segmentation, the tissue area is divided into patches at a size chosen by the user, which can be overlapping or non-overlapping.

• <b>C - Feature extraction</b>: Each image patch is passed through a VGG16 CNN feature extractor and embedded into a [1 × 1024] feature vector. All feature vectors from a given patient are aggregated into a matrix. The number of rows in the matrix will vary as each patient has a variable set of WSIs, each with their own dimensions.

• <b>D - k-Nearest-Neighbour Graph</b>: The matrix of feature vectors of each patient is used to create a sparse directed k-NNG using the Euclidean distance metric, with a default of k = 5. The attribute of each node corresponds to a [1 × 1024] feature vector. This graph is used as input to the GNN.

• <b>E - Graph classification</b>: The k-NNG is successively passed through four Graph Attention Network layers (GAT) [34] and SAGPooling layers [26]. The SAGPooling readouts from each layer are concatenated and passed through three MLP layers and finally classified.

• <b>F - Prediction</b>: A pathotype or diagnosis prediction is obtained at the patient-level.