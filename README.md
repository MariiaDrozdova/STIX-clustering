# Visualizing three years of STIX X-ray flare observations using self-supervised learning - official implementation of the work present at EGU conference in 2024

## Overview
Operating continuously for over three years, Solar Orbiter's Spectrometer/Telescope for Imaging X-rays (STIX) has captured more than 43,000 X-ray flares. This project harnesses self-supervised learning to organize and visualize these flares based on their visual properties. Through the application of networks like Masked Siamese Networks and Autoencoders, we extract latent space embeddings that encapsulate the core characteristics of each flare. This approach facilitates a nuanced exploration of the data, allowing for the grouping of flares by shared morphological features and aiding in the identification of unique or noteworthy events within the extensive dataset.

## Repository Contents
- poster/: The final poster presented at EGU 2024.
- outputs/: The folder where notebook will write the data.

## Installation

```
git clone https://github.com/yourusername/STIX-clustering.git
cd STIX-clustering
pip install -r requirements.txt
```
You can download our data from here:

```
https://drive.switch.ch/index.php/s/S8hIRBa9wlKtDr8
```

## How to Use

1. **Configure Paths:**
   - Open `general.py` and modify the `DATA_PATH` and `TEST_PATH` variables to point to the locations of your datasets.
   - `DATA_PATH` should point to the folder containing your large unlabeled dataset (the folder named `stix_data`).
   - `TEST_PATH` should point to the folder containing your test set with `.fits` files organized into class-specific subfolders (the folder named `stx_reconstructions`).
     
   Example settings if your data is in the same directory as your code:
```
DATA_PATH = "./" # Directory containing the 'stix_data' folder
TEST_PATH = "./" # Directory containing the 'stx_reconstructions' folder
```

2. **Prepare File List for Analysis:**
- Create a text file named `saved_filtered_filenames_final.txt` and list the filenames you intend to analyze.
- If you plan to use all files in the `stix_data` directory, generate a list of all file names. For example:
  ```
  stix_data/your_file_1.txt
  stix_data/your_file_2.txt
  ```

3. **Usage of Notebooks:**
- Use `Visualize_clusters.ipynb` to visualize and filter clusters. The last cell of this notebook will generate a file named `saved_filtered_filenames_new.txt`, which you can rename to `saved_filtered_filenames_latest.txt` if you you decided to keep only certain clusters (filtering).
- Use `anomaly_detection.ipynb` to perform anomaly detection. This notebook includes methods like Isolation Forest and a simple classifier for detecting anomalies.


## Citation
If you use this work or dataset in your research, please cite:

```
Drozdova, M., Kinakh, V., Ramunno, F., Lastufka, E., and Voloshynovskiy, S.: Visualizing three years of STIX X-ray flare observations using self-supervised learning, EGU General Assembly 2024, Vienna, Austria, 14–19 Apr 2024, EGU24-18534, https://doi.org/10.5194/egusphere-egu24-18534, 2024.
```

## Contact
For any inquiries or collaboration opportunities, please reach out to Mariia Drozdova at `mariia.drozdova at unige.ch` and Slava Voloshynovskiy at `svolos at unige.ch`.
