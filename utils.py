import glob
import math
import torch
import cv2 
import tqdm
import math

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

def normalize_data(data):
    # Normalize and scale data to be in the 0-255 range
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    return (data * 255).astype(np.uint8)

def compute_metrics(true_labels, predicted_labels):
    metrics = {
        'Adjusted Rand Index (ARI)': adjusted_rand_score(true_labels, predicted_labels),
        'Normalized Mutual Information (NMI)': normalized_mutual_info_score(true_labels, predicted_labels),
        'Homogeneity': homogeneity_score(true_labels, predicted_labels),
        'Completeness': completeness_score(true_labels, predicted_labels)
    }
    return metrics
    
def evaluate_clustering(true_labels, predicted_labels, steps, save_results=False, results_file='clustering_evaluation.csv', mode="standard"):
    """
    Evaluate clustering performance using various metrics and steps used in clustering.
    
    Args:
        true_labels (array-like): The true class labels.
        predicted_labels (array-like): The cluster labels predicted by the clustering method.
        steps (list of dicts): Steps used in the clustering pipeline, including method names and parameters.
        save_results (bool): Whether to save the results to a CSV file.
        results_file (str): Path to the CSV file to save the results.
        
    Returns:
        dict: A dictionary containing the computed metrics.
    """

    metrics = compute_metrics(true_labels, predicted_labels)
    # Concatenate method names and parameters from steps for logging
    method_name = mode + ', ' + ', '.join([step['type'] for step in steps])
    parameters = '; '.join([f"{step['type']}: {step['params']}" for step in steps])

    if save_results:
        import csv
        import os

        fieldnames = ['Method', 'Parameters'] + list(metrics.keys())
        file_exists = os.path.isfile(results_file)

        with open(results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # Write header only once

            row = {'Method': method_name, 'Parameters': parameters}
            row.update(metrics)
            writer.writerow(row)

    return metrics
    
def apply_min_max_normalization(features):
    min_image = np.min(features)
    max_image = np.max(features)
    features_standardized = (features - min_image)/(max_image - min_image + 1e-15)
    return features_standardized
    

def apply_pca(features, n_components=0.95):
    pca = PCA(n_components=n_components)
    print(features.shape)
    pca.fit(features)
    return pca, pca.transform(features)

def apply_tsne(features, n_components=2, learning_rate='auto', init='random', perplexity=30):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(features)

def apply_dbscan(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(features)

def apply_kmeans(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    return kmeans, kmeans.predict(features)

def extract_sift_features(images):
    min_image = np.min(images, axis=-1).reshape(-1,1,1)
    max_image = np.max(images, axis=-1).reshape(-1,1,1)
    images = images.reshape(-1, im_size, im_size)
    images = np.uint8(255*(images - min_image)/(max_image - min_image + 1e-15))


    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des)
 
    all_descriptors = np.vstack([des for des in descriptors_list if des is not None])
    return all_descriptors, descriptors_list

def apply_histogram_clustering(features, n_clusters=10):
    histograms = [np.histogram(feature.flatten(), bins=32, range=(0, 256))[0] for feature in features]
    histograms = np.array(histograms)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(histograms)
    return kmeans, kmeans.predict(histograms)

def create_visual_vocabulary(descriptors, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptors)
    return kmeans

def create_histograms(descriptor_list, visual_vocabulary):
    histograms = []
    for descriptors in descriptor_list:
        if descriptors is not None:
            labels = visual_vocabulary.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(visual_vocabulary.n_clusters+1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(visual_vocabulary.n_clusters)) 
    histograms = np.array(histograms)
    return histograms

def extract_features(dataloader, property_names=["class"]):
    all_features = []
    propertiers = {}
    for name in property_names:
        propertiers[name] = []
    for batch in tqdm.tqdm(dataloader):
        features = batch['features'].cpu().detach().numpy() 
        all_features.append(features)
        for name in property_names:
            if name in batch:
                try:
                    property_data = batch[name].cpu().detach().numpy()
                except:
                    property_data = np.array([(cl) for cl in batch[name]])
                propertiers[name].append(property_data.reshape(-1))
            else:
                propertiers[name].append(np.full(len(features), -1))
    all_features = np.vstack(all_features)
    all_features = all_features.reshape(all_features.shape[0], -1)
    
    for name in property_names: 
        propertiers[name] = np.concatenate(propertiers[name])
    return all_features, propertiers

def visualize_feautres(tsne_results, ):
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1],  alpha=0.5)
    plt.title('Clustering visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    
def visualize_clusters(tsne_results, clusters, cmap="viridis"):
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap=cmap, alpha=0.5)
    plt.title('Clustering visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()

def train_and_store_models(features, steps):
    models_info = {}
    transformed_features = features

    for step in steps:
        if step['type'] == 'pca':
            models_info["pca_features"] = transformed_features.copy()
            models_info["pca_params"] = step['params']

        elif step['type'] == 'tsne':
            models_info["tsne_features"] = transformed_features.copy()
            models_info["tsne_params"] = step['params']
            
        elif step['type'] == 'dbscan':
            models_info['dbscan_features'] = transformed_features
            models_info['dbscan_params'] = step['params']

        elif step['type'] == 'kmeans':
            models_info['kmeans_features'] = transformed_features
            models_info['kmeans_params'] = step['params']

        elif step['type'] == 'histogram_kmeans':
            model, labels = apply_histogram_clustering(features, **step['params'])  # Note: uses original features
            models_info['histogram_kmeans'] = model
            models_info['histogram_kmeans_labels'] = labels

        elif step['type'] == 'sift_kmeans':
            all_descriptors, descriptors_list = extract_sift_features(features)  # Note: uses original features
            visual_vocabulary = create_visual_vocabulary(all_descriptors, **step['params'])
            histograms = create_histograms(descriptors_list, visual_vocabulary)
            sift_kmeans = KMeans(**step['params_kmeans'])
            sift_kmeans.fit(histograms)
            labels = sift_kmeans.predict(histograms)
            models_info['sift_visual_vocabulary'] = visual_vocabulary
            models_info['sift_kmeans'] = sift_kmeans
            models_info['sift_kmeans_labels'] = labels

    return models_info, transformed_features
    
def apply_models_to_test_data(features, models_info):
    transformed_features = features

    if 'pca_features' in models_info:
        united_features = np.vstack([models_info['pca_features'], transformed_features])
        pca, united_features = apply_pca(united_features, **models_info["pca_params"])
        models_info["pca_transformed_train_features"] = united_features[:len(models_info['pca_features'])]
        models_info["pca_transformed_test_features"] = united_features[len(models_info['pca_features']):]
        models_info["pca"] = pca

        united_features = models_info['pca_features']
        pca, pca_transformed_train_features = apply_pca(united_features, **models_info["pca_params"])
        models_info["pca_transformed_train_features"] = pca_transformed_train_features
        models_info["pca"] = pca

        united_features = transformed_features
        pca, pca_transformed_test_features = apply_pca(united_features, **models_info["pca_params"])
        models_info["pca_transformed_test_features"] = pca_transformed_test_features
        
    if 'tsne_features' in models_info:
        united_features = np.vstack([models_info['tsne_features'], transformed_features])
        united_features = apply_tsne(united_features, **models_info["tsne_params"])
        models_info["tsne_transformed_train_features"] = united_features[:len(models_info['tsne_features'])]
        models_info["tsne_transformed_test_features"] = united_features[len(models_info['tsne_features']):]
        
    if 'kmeans_features' in models_info:
        united_features = np.vstack([models_info['kmeans_features'], transformed_features])
        kmeans, labels = apply_kmeans(united_features, **models_info["kmeans_params"])
        models_info["kmeans"] = kmeans
        models_info["kmeans_labels"] = labels[:len(models_info['kmeans_features'])]
        models_info["kmeans_labels_test"] = labels[len(models_info['kmeans_features']):]

        united_features = models_info['kmeans_features']
        kmeans, labels = apply_kmeans(united_features, **models_info["kmeans_params"])
        models_info["kmeans_labels"] = labels
        models_info["kmeans"] = kmeans

        united_features = transformed_features
        kmeans, labels = apply_kmeans(united_features, **models_info["kmeans_params"])
        models_info["kmeans_labels_test"] = labels

    if 'dbscan_features' in models_info:
        united_features = np.vstack([models_info['dbscan_features'], transformed_features])
        labels = apply_dbscan(united_features, **models_info["dbscan_params"])
        models_info["dbscan_labels"] = labels[:len(models_info['dbscan_features'])]
        models_info["dbscan_labels_test"] = labels[len(models_info['dbscan_features']):]
        
    if 'histogram_kmeans' in models_info:
        histograms = [np.histogram(feature.flatten(), bins=32, range=(0, 256))[0] for feature in features]
        histograms = np.array(histograms)
        labels = models_info['histogram_kmeans'].predict(histograms)
        models_info['histogram_kmeans_labels_test'] = labels

    if 'sift_kmeans' in models_info:
        _, descriptors_list = extract_sift_features(features)
        histograms = create_histograms(descriptors_list, models_info['sift_visual_vocabulary'])
        labels = models_info['sift_kmeans'].predict(histograms)
        models_info['sift_kmeans_labels_test'] = labels


    return models_info, transformed_features  
    
def replace_string_values(thermal_components):
    # Map each unique thermal component to an integer
    unique_components = np.unique(thermal_components)
    component_mapping = {component: idx for idx, component in enumerate(unique_components)}
    
    # Replace thermal components with their corresponding integer values
    replaced_components = np.vectorize(component_mapping.get)(thermal_components)
    
    return replaced_components




def visualize_np_images_from_cluster(cluster_num, cluster_assignments, images, max_images=32, norm_func=np.log1p):
    # Filter indices of images that belong to the cluster
    cluster_indices = [i for i, cluster_id in enumerate(cluster_assignments) if cluster_id == cluster_num]

    # Determine the number of images to display (up to max_images)
    num_images = min(len(cluster_indices), max_images)

    # Calculate grid size for plotting
    num_cols = int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows), squeeze=False)
    axs = axs.flatten()  # Flatten the array of axes for easy indexing

    for ax in axs[num_images:]:  
        ax.axis('off')

    count = 0
    for i, data in enumerate(images):
        if i in cluster_indices:
            # Scale the image data accroding to norm_func if needed
            img = norm_func(data)

            axs[count].imshow(img[0], cmap='inferno') 
            axs[count].axis('off')

            count += 1
            if count >= num_images:  
                return

def combine_predictions(y_predicted, y_test):
    predictions = {y: np.unique(y_predicted[y_test == y]) for y in np.unique(y_test)}
    return predictions


def visualize_batch_images_from_cluster(cluster_num, cluster_assignments, data_loader, max_images=32, norm_func=np.log1p):
    cluster_indices = [i for i, cluster_id in enumerate(cluster_assignments) if cluster_id == cluster_num]

    num_images = min(len(cluster_indices), max_images)
    
    num_cols = int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(num_images / num_cols))
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    axs = axs.flatten()  
    
    for ax in axs[num_images:]: 
        ax.axis('off')

    count = 0
    for i, batch in enumerate(data_loader):
        for j, data in enumerate(batch['data']):
            batch_index = i * data_loader.batch_size + j
            if batch_index in cluster_indices:
                img = norm_func(data.numpy())
                axs[count].imshow(img[0], cmap='inferno')
                axs[count].axis('off')
                
                count += 1
                if count >= num_images: 
                    plt.show()
                    return
    plt.show()
    
def visualize_np_images_from_cluster(cluster_num, cluster_assignments, images, max_images=32, norm_func=np.log1p):
    cluster_indices = [i for i, cluster_id in enumerate(cluster_assignments) if cluster_id == cluster_num]

    num_images = min(len(cluster_indices), max_images)

    num_cols = int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows), squeeze=False)
    axs = axs.flatten() 

    for ax in axs[num_images:]: 
        ax.axis('off')

    count = 0
    for i, data in enumerate(images):
        if i in cluster_indices:
            img = norm_func(data)

            axs[count].imshow(img[0], cmap='inferno') 
            axs[count].axis('off')

            count += 1
            if count >= num_images: 
                return
    
def visualize_histograms_from_cluster(cluster_num, cluster_assignments, images, max_images=32, num_bins=32, alpha=0.7,norm_func=np.log1p):
    cluster_indices = [i for i, cluster_id in enumerate(cluster_assignments) if cluster_id == cluster_num]

    num_images = min(len(cluster_indices), max_images)
    
    num_cols = int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(num_images / num_cols))
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    axs = axs.flatten()
    
    for ax in axs[num_images:]:
        ax.axis('off')

    count = 0
    for i, data in enumerate(images):
        if i in cluster_indices:
            img = norm_func(data)

            axs[count].hist(img[0].flatten(), bins=num_bins, color='red', alpha=alpha)
            axs[count].axis('off')
                
            count += 1
            if count >= num_images: 
                plt.show()
                return
    plt.show()
    
def test_visualize_images_all_cluster_zero(data_loader, norm_func=np.log1p, max_images=32):
    # Mock cluster assignments: all images are in cluster 0
    num_images = len(data_loader.dataset)
    cluster_assignments = [0] * num_images  # Assign all images to cluster 0

    # Call the visualization function for cluster 0
    cluster_num = 0  
    visualize_batch_images_from_cluster(cluster_num, cluster_assignments, data_loader, max_images=max_images, norm_func=norm_func)

    print("Test complete - all images should belong to cluster 0 and be visualized.")


