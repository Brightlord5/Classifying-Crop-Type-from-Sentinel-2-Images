# app.py
import streamlit as st
import os
import subprocess
from pathlib import Path
import time
import pandas as pd
import numpy as np
import tifffile as tiff
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.plot import show
from scipy.ndimage import binary_dilation
import matplotlib.patches as mpatches
import glob
import datetime
from tqdm.notebook import tqdm
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from scipy.stats.mstats import gmean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import log_loss
import os
import cv2
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
from huggingface_hub import hf_hub_download, login
import tempfile

warnings.filterwarnings("ignore")

login(token="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")  # Replace with your actual token

    # Crop types dictionary for legend
crop_types = {
    1: 'Maize',
    2: 'Cassava',
    3: 'Common Bean',
    4: 'Maize & Common Bean (intercropping)',
    5: 'Maize & Cassava (intercropping)',
    6: 'Maize & Soybean (intercropping)',
    7: 'Cassava & Common Bean (intercropping)'
}

# Add image cache dictionary
image_cache = {}

# Add Timer class for performance tracking
class Timer:
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        st.write(f"{self.description}: {self.duration:.2f} seconds")
        
def read_tiff_from_hub(repo_id, file_path):
    """
    Downloads and reads a TIFF file from Hugging Face Hub.
    
    This function handles the process of:
    1. Downloading the file from the Hub repository
    2. Reading it into memory using tifffile
    3. Returning the image data as a numpy array
    
    Parameters:
        repo_id (str): The Hugging Face repository ID (username/repo-name)
        file_path (str): Path to the file within the repository
        
    Returns:
        numpy.ndarray: The image data
    """
    try:
        # Download the file from Hugging Face Hub
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset"
        )
        
        # Read the downloaded file using tifffile
        return tiff.imread(local_path)
        
    except Exception as e:
        print(f"Error reading file from Hub: {str(e)}")
        return None

def pipeline(file_path):
    # List of dates that an observation from Sentinel-2 is provided in the training dataset
    dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
            datetime.datetime(2019, 7, 1, 8, 10, 4),
            datetime.datetime(2019, 7, 6, 8, 10, 8),
            datetime.datetime(2019, 7, 11, 8, 10, 4),
            datetime.datetime(2019, 7, 21, 8, 10, 4),
            datetime.datetime(2019, 8, 5, 8, 10, 7),
            datetime.datetime(2019, 8, 15, 8, 10, 6),
            datetime.datetime(2019, 8, 25, 8, 10, 4),
            datetime.datetime(2019, 9, 9, 8, 9, 58),
            datetime.datetime(2019, 9, 19, 8, 9, 59),
            datetime.datetime(2019, 9, 24, 8, 9, 59),
            datetime.datetime(2019, 10, 4, 8, 10),
            datetime.datetime(2019, 11, 3, 8, 10)]


    bands_all = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    sample_submission = pd.read_csv("SampleSubmission.csv")

    # Define all dates from the training data
    # These dates represent the complete temporal sequence we need to match
    all_dates = [
        "20190606", "20190701", "20190706", "20190711", "20190721",
        "20190805", "20190815", "20190825", "20190909", "20190919",
        "20190924", "20191004", "20191103"
    ]

    # First, let's read our reference DataFrame that contains field locations
    df_generated = pd.read_csv("try_this.csv")

    def extract_date_and_tile(zip_file_path):
        # Remove any file extension and get just the base name
        clean_name = zip_file_path.split('/')[-1].split('.')[0]
        
        # Split the filename into parts at each underscore
        name_parts = clean_name.split('_')
        
        # Make sure we have enough parts
        if len(name_parts) < 2:
            raise ValueError("Filename doesn't match expected pattern")
            
        tile_number = int(name_parts[-2])
        date_string = int(name_parts[-1])
        
        return tile_number, date_string

    # Let's examine what tiles we have in our data
    unique_tiles = df_generated['tile'].unique()
    print(f"Tiles in our data: {unique_tiles}")

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    # Define our bands that we'll be processing
    all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B11", "B12", "CLD"]

    # Example usage:
    filename = file_path
    tile, date = extract_date_and_tile(filename)
    # print(f"Tile: {tile}")  
    # print(f"Date: {date}")  

    current_date = date
    current_tile = tile

    # Get the data for just this tile
    tile_data = df_generated[df_generated['tile'] == current_tile].copy()
    # print(f"\nProcessing Tile {current_tile}")
    # print(f"Number of locations in this tile: {len(tile_data)}")

    # Initialize columns for ALL dates with zeros
    # This ensures we have the same structure as the training data
    print("\nInitializing columns for all dates:")
    for date in all_dates:
        for band in all_bands:
            column_name = f"{date}_{band}"
            df_generated[column_name] = 0
            # print(f"Created column: {column_name}")

    print("\nProcessing actual data for current date and tile:")
    # Process each band for the current date
    for band in all_bands:
        try:
            # Construct the path to the band file
            # band_path = f"data/ref_african_crops_kenya_02_source/ref_african_crops_kenya_02_tile_0{current_tile}_{current_date}/{band}.tif"
            # print(f"\nProcessing {band_path}")
            
            hub_path = f"ref_african_crops_kenya_02_tile_0{current_tile}_{current_date}/{band}.tif"
            print(f"\nProcessing {hub_path}")
            
            # # Read the image using tifffile
            # image = tiff.imread(band_path)
            # print(f"Image shape: {image.shape}")
            
            # Read the image from Hugging Face Hub
            image = read_tiff_from_hub(
                repo_id="Armaan0510/data",
                file_path=hub_path
            )
            
            if image is None:
                print(f"Failed to load {band} from Hub")
                continue
            
            print(f"Image shape: {image.shape}")
            
            # Create the column name
            column_name = f"{current_date}_{band}"
            
            # Update values for this band for the current tile only
            for idx, row in tile_data.iterrows():
                try:
                    pixel_value = image[row['row_loc'], row['col_loc']]
                    df_generated.loc[idx, column_name] = pixel_value
                except IndexError as e:
                    print(f"Warning: Location ({row['row_loc']}, {row['col_loc']}) outside bounds for {band}")
                    continue
                    
            # Print some statistics to verify the update
            updated_values = df_generated[df_generated['tile'] == current_tile][column_name]
            # print(f"Statistics for {band}:")
            # print(f"  Non-zero values: {(updated_values != 0).sum()}")
            # print(f"  Mean value: {updated_values.mean():.2f}")
            
        except Exception as e:
            print(f"Error processing {band}: {str(e)}")
            continue

    df_ungrouped = df_generated.copy()

    # Spatial features to be merged with dataset (by Field ID) later on
    row_size = df_ungrouped.groupby("fid")["row_loc"].nunique()
    column_size = df_ungrouped.groupby("fid")["col_loc"].nunique()
    num_pixels = df_ungrouped.groupby("fid")["label"].count()

    # Grouped Data
    df_grouped = df_ungrouped.groupby("fid", as_index = False).mean()

    # Dataframe for 1st modelling
    df_all = df_grouped.copy()

    # Dataframe for 2nd modelling
    df_pixels = df_grouped.copy()

    # Drop non-allowed features. Field ID is dropped later on as it's still needed for further processing
    df_all = df_all.drop(columns = ["row_loc", "col_loc", "tile"])
    df_pixels = df_pixels.drop(columns = ["row_loc", "col_loc", "tile"])

    cloud_columns = ['20190606_CLD', '20190701_CLD', '20190706_CLD', '20190711_CLD', '20190721_CLD', '20190805_CLD', '20190815_CLD', '20190825_CLD', '20190909_CLD', '20190919_CLD', '20190924_CLD', '20191004_CLD', '20191103_CLD']

    df_pixels.drop(columns = cloud_columns + ["fid"], inplace = True)

    ## Spectral Indices calculation

    spectral_indices = ["NDVI", "GNDVI", "EVI", "EVI2", "AVI", "BSI", "SI", "NDWI", "NDMI", "NPCRI"]

    for i in range(13):
    #     Band Pixel values per timestamp
        b1 = df_all.filter(like = "B01").values[:,i]
        b2 = df_all.filter(like = "B02").values[:,i]
        b3 = df_all.filter(like = "B03").values[:,i]
        b4 = df_all.filter(like = "B04").values[:,i]
        b5 = df_all.filter(like = "B05").values[:,i]
        b6 = df_all.filter(like = "B06").values[:,i]
        b7 = df_all.filter(like = "B07").values[:,i]
        b8 = df_all.filter(like = "B08").values[:,i]
        b8a = df_all.filter(like = "B8A").values[:,i]
        b9 = df_all.filter(like = "B09").values[:,i]    
        b11 = df_all.filter(like = "B11").values[:,i]
        b12 = df_all.filter(like = "B12").values[:,i]
        
    #     Computation of indices
        ndvi = (b8 - b4) / (b8 + b4)
        gndvi = (b8 - b3) / (b8 + b3)
        evi = 2.5 * (b8 - b4) / ((b8 + 6.0 * b4 - 7.5 * b2) + 1.0)    
        evi2 = 2.4 * (b8 - b4) / (b8 + b4 + 1.0)
        avi = (b8 * (1 - b4) * (b8 - b4))
        bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))
        si = ((1 - b2) * (1 - b3) * (1 - b4))
        ndwi = (b3 - b8) / (b3 + b8)
        ndmi = (b8 - b11) / (b8 + b11)
        npcri = (b4 - b2) / (b4 + b2) 
        
    #     Add indices as features to 1st dataframe per timestamp
        df_all[f'NDVI_{dates[i]}'] = ndvi 
        df_all[f'GNDVI_{dates[i]}'] = gndvi
        df_all[f'EVI_{dates[i]}'] = evi
        df_all[f'EVI2_{dates[i]}'] = evi2
        df_all[f'AVI_{dates[i]}'] = avi
        df_all[f'BSI_{dates[i]}'] = bsi
        df_all[f'SI_{dates[i]}'] = si    
        df_all[f'NDWI_{dates[i]}'] = ndwi
        df_all[f'NDMI_{dates[i]}'] = ndmi
        df_all[f'NPCRI_{dates[i]}'] = npcri
        
    for i in spectral_indices:
        df_all[f'{i}_min'] = df_all.filter(regex = f'^{i}').min(axis = 1)
        df_all[f'{i}_max'] = df_all.filter(regex = f'^{i}').max(axis = 1)
        df_all[f'{i}_avg'] = df_all.filter(regex = f'^{i}').mean(axis = 1)
        df_all[f'{i}_std'] = df_all.filter(regex = f'^{i}').std(axis = 1) 
        
    for i in bands:
        df_pixels[f'{i}_std'] = df_pixels.filter(like = f'_{i}').std(axis = 1)
        df_pixels[f'{i}_max'] = df_pixels.filter(like = f'_{i}').max(axis = 1)
        df_pixels[f'{i}_min'] = df_pixels.filter(like = f'_{i}').min(axis = 1)
        df_pixels[f'{i}_avg'] = df_pixels.filter(like = f'_{i}').mean(axis = 1)
        
    new_df = df_all.copy()

    new_df["row_size"] = new_df.fid.map(row_size)
    new_df["col_size"] = new_df.fid.map(column_size)
    new_df["area"] = new_df.apply(lambda row: row.row_size * row.col_size, axis = 1)
    # number of pixels covered by a field in the area computed
    new_df["num_pixels"] = new_df.fid.map(num_pixels)

    new_df = new_df.drop(columns = ["fid"])

    # Splitting the data into train and test
    # df_all_train = new_df[new_df.label != 0].copy()
    df_all_test = new_df[new_df.label == 0].copy()

    # df_all_train = df_all_train.reset_index(drop = True)
    df_all_test = df_all_test.reset_index(drop = True)

    # df_pixels_train = df_pixels[df_pixels.label != 0].copy()
    df_pixels_test = df_pixels[df_pixels.label == 0].copy()

    # df_pixels_train = df_pixels_train.reset_index(drop = True)
    df_pixels_test = df_pixels_test.reset_index(drop = True)

    df_all_test.drop("label", inplace = True, axis = 1)
    df_pixels_test.drop("label", inplace = True, axis = 1)

    features_to_drop = [
        '20190815_CLD', '20191004_CLD', '20190919_CLD', '20190701_CLD',
        '20190924_CLD', '20190706_CLD', 'SI_avg', '20190606_B07',
        'BSI_max', '20190805_B04', '20191004_B03',
        'SI_std', '20190919_B03', '20190805_B08',
        'EVI_min', '20190721_B07', '20190711_B07',
        '20190721_B05', '20190805_B09',
        '20190706_B09', '20191004_B08',
        '20190825_B8A', '20190919_B01', '20190805_B06',
        '20190706_B11',
        'col_size', '20190701_B04',
        'NPCRI_max', '20190805_B12',
        '20190919_B04', 'BSI_avg',
        '20190919_B8A', '20190909_B05', '20190706_B06',
        '20190924_B8A', '20190909_B09', '20191004_B01', '20190805_B03',
        '20190706_B07',
        '20190909_CLD', '20190825_B12', '20190606_B06', '20190909_B07',
        '20190805_B8A', '20190924_B09', '20190701_B07'
    ]

    df_all_test = df_all_test.drop(columns = features_to_drop)


    #Make sure I save these models     
    import pickle

    # Define file paths for all models
    model_paths = {
        "bc_model": "trained_models/bc_model.pkl",
        "cb_all_model": "trained_models/cb_all_model.pkl",
        "cb_pixels_model": "trained_models/cb_pixels_model.pkl",
        "cb2_all_model": "trained_models/cb2_all_model.pkl",
        "cb2_pixels_model": "trained_models/cb2_pixels_model.pkl"
    }

    # Load all models into a dictionary
    models = {}
    for model_name, path in model_paths.items():
        with open(path, "rb") as file:
            models[model_name] = pickle.load(file)


    cb = models["cb_all_model"]
    cb2 = models["cb2_all_model"]
    bc = models["bc_model"]

    test_preds_all_1 = cb.predict_proba(df_all_test)
    test_preds_all_2 = cb2.predict_proba(df_all_test)
    # test_preds_all_3 = bc.predict_proba(df_all_test)

    #write code to import a numpy file into test_preds_all_3
    test_preds_all_3 = np.load('test_preds_all_3.npy')

    # Level 1
    test_preds_all = (0.72 * test_preds_all_1) + ((1 - 0.72) * test_preds_all_2)
    # Level 2
    test_preds_all = (0.7 * test_preds_all) + ((1 - 0.7) * test_preds_all_3)

    cb_pixels = models["cb_pixels_model"]
    cb2_pixels = models["cb2_pixels_model"]

    test_preds_pixels_1 = cb_pixels.predict_proba(df_pixels_test)
    test_preds_pixels_2 = cb2_pixels.predict_proba(df_pixels_test)
    # test_preds_pixels_3 = bc.predict_proba(df_pixels_test)
    test_preds_pixels_3 = np.load('test_preds_pixels_3.npy')

    # Level 1
    test_preds_pixels = (0.76 * test_preds_pixels_1) + ((1 - 0.76) * test_preds_pixels_2)
    # Level 2
    test_preds_pixels = (0.67 * test_preds_pixels) + ((1 - 0.67) * test_preds_pixels_3)
    # Level 3
    test_preds = (0.68 * test_preds_all) + ((1 - 0.68) * test_preds_pixels)

    # Save the predictions
    test_preds = pd.DataFrame(test_preds)

    sample_submission.Crop_ID_1 = test_preds[0]
    sample_submission.Crop_ID_2 = test_preds[1]
    sample_submission.Crop_ID_3 = test_preds[2]
    sample_submission.Crop_ID_4 = test_preds[3]
    sample_submission.Crop_ID_5 = test_preds[4]
    sample_submission.Crop_ID_6 = test_preds[5]
    sample_submission.Crop_ID_7 = test_preds[6]

    # print("Program end reached")

    # sample_submission.to_csv("SampleSub.csv", index = False)
    # df_all.to_csv("df_all.csv", index = False)
    save_processed_data(sample_submission, df_all, current_date)

    # print("Sample and df_all saved")

    # Define new colors with higher contrast and less transparency
    # Using brighter, more saturated colors that will stand out against green/brown background
COLORS = [
    (1, 0, 0, 0.5),     # Bright red
    (0, 1, 0.3, 0.5),   # Bright lime
    (0.3, 0.3, 1, 0.5), # Bright blue
    (1, 1, 0, 0.5),     # Bright yellow
    (1, 0, 1, 0.5),     # Bright magenta
    (0, 1, 1, 0.5),     # Bright cyan
    (1, 0.5, 0, 0.5)    # Bright orange
]

def load_field_data():
    """Load and preprocess field location data and predictions"""
    field_data_path = hf_hub_download(
        repo_id="Armaan0510/data",
        filename="try_this.csv",
        repo_type="dataset"
    )
    
    # Download predictions file (SampleSub.csv) from Hub
    predictions_path = hf_hub_download(
        repo_id="Armaan0510/data",
        filename="SampleSub.csv",
        repo_type="dataset"
    )
    
    # Read the downloaded files into pandas DataFrames
    field_data = pd.read_csv(field_data_path)
    predictions = pd.read_csv(predictions_path)
    
    field_data['fid'] = field_data['fid'].astype(str)
    field_data['tile'] = field_data['tile'].astype(int)
    predictions['Field_ID'] = predictions['Field_ID'].astype(str)
    
    # print(f"Loaded {len(field_data)} field locations and {len(predictions)} predictions")
    return field_data, predictions

def get_predicted_crop(field_id, predictions_df):
    """Get the most likely crop type for a given field"""
    try:
        pred_row = predictions_df[predictions_df['Field_ID'] == field_id].iloc[0]
        crop_probs = [pred_row[f'Crop_ID_{i}'] for i in range(1, 8)]
        return np.argmax(crop_probs) + 1
    except (IndexError, KeyError) as e:
        return None

def create_field_patch(shape, row, col, size=5):
    """Create a circular patch around the field point"""
    patch = np.zeros(shape, dtype=bool)
    
    # Create bounds for the patch
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    mask = (rr - row)**2 + (cc - col)**2 <= size**2
    patch[mask] = True
    return patch

def fast_load_satellite_image(tile, date, base_path="data/ref_african_crops_kenya_02_source", scale_factor=2):
    """Optimized satellite image loading with downsampling"""
    cache_key = f"{tile}_{date}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    try:
        with Timer(f"Loading satellite image for tile {tile}"):
            def read_band(band):
                path = f"{base_path}/ref_african_crops_kenya_02_tile_{tile}_{date}/{band}.tif"
                with rasterio.open(path) as src:
                    # Read at reduced resolution
                    img = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
                    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            with ThreadPoolExecutor(max_workers=3) as executor:
                r, g, b = executor.map(read_band, ['B04', 'B03', 'B02'])

            rgb = cv2.merge([r, g, b])
            rgb = cv2.convertScaleAbs(rgb, alpha=1.5)
            
            result = (rgb, rgb.shape)
            image_cache[cache_key] = result
            return result
            
    except Exception as e:
        st.error(f"Error loading satellite image: {e}")
        return None, None

    # def create_prediction_overlay(field_data, predictions_df, image_shape, current_tile):
    #     """Create colored overlay showing predicted crops with larger patches"""
    #     overlay = np.zeros((*image_shape, 4), dtype=float)
        
    #     # Filter fields for current tile
    #     tile_fields = field_data[field_data['tile'] == current_tile].copy()
    #     # print(f"Processing {len(tile_fields)} fields for tile {current_tile}")
        
    #     # Create a mask for each crop type
    #     crop_masks = [np.zeros(image_shape, dtype=bool) for _ in range(7)]
        
    #     for _, field in tile_fields.iterrows():
    #         crop_type = get_predicted_crop(field['fid'], predictions_df)
            
    #         if crop_type is not None:
    #             row, col = int(field['row_loc']), int(field['col_loc'])
                
    #             if 0 <= row < image_shape[0] and 0 <= col < image_shape[1]:
    #                 # Create patch for this field
    #                 patch = create_field_patch(image_shape, row, col, size=7)
    #                 crop_masks[crop_type - 1] |= patch
        
    #     # Apply colors to masks
    #     for i, mask in enumerate(crop_masks):
    #         # Dilate the mask to create slightly larger areas
    #         dilated_mask = binary_dilation(mask, iterations=2)
    #         overlay[dilated_mask] = COLORS[i]
        
    #     return overlay

    # Add this new visualization function
    
def save_processed_data(sample_submission, df_all, current_date):
    """
    Saves the processed data files to a local folder.
    Creates folder if it doesn't exist and includes date in filenames
    to keep things organized.
    """
    # Create data folder if it doesn't exist
    # os.makedirs('processed_data', exist_ok=True)
    
    # Save files with date in filename
    # sample_submission.to_csv(f"processed_data/SampleSub.csv", index=False)
    # df_all.to_csv(f"processed_data/df_all.csv", index=False)
    
    # Store paths in session state so we know which files to use later
    st.session_state['current_date'] = current_date

def create_seaborn_visualization(rgb_image, field_data, predictions_df, current_tile, tile_str, scale_factor=2):
    """Create Seaborn visualization with optimized settings"""
    with Timer(f"Creating visualization for tile {tile_str}"):
        # Set plot style
        sns.set_style("dark")
        
        # Create figure with appropriate size and DPI
        plt.figure(figsize=(10, 10), dpi=100)
        
        # Display the satellite image
        plt.imshow(rgb_image)
        
        # Filter fields for current tile and prepare data
        tile_fields = field_data[field_data['tile'] == current_tile]
        
        # Create scatter plots for each crop type
        for crop_type in range(1, 8):
            crop_fields = tile_fields[
                tile_fields['fid'].map(lambda x: get_predicted_crop(x, predictions_df) == crop_type)
            ]
            
            if not crop_fields.empty:
                plt.scatter(
                    crop_fields['col_loc'] // scale_factor,
                    crop_fields['row_loc'] // scale_factor,
                    c=[COLORS[crop_type-1]],
                    label=crop_types[crop_type],
                    alpha=0.7,
                    s=50,
                    edgecolor='white',
                    linewidth=0.5
                )
        
        # Customize the plot
        plt.title(f'Predicted Crop Types - Tile {tile_str}', pad=20)
        plt.axis('off')
        
        # Optimize legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                borderaxespad=0., frameon=True, framealpha=0.8)
        
        plt.tight_layout()
        
        return plt.gcf()


# def display_visualizations():
#     """
#     Displays all images from the visualizations folder in order.
#     The function looks for images with standard naming patterns and
#     displays them with appropriate titles and descriptions.
#     """
#     viz_folder = Path("visualizations")
    
#     # Check if visualization folder exists
#     if not viz_folder.exists():
#         st.warning("No visualizations folder found. Please run the pipeline first.")
#         return
        
#     # Get all image files from the folder
#     image_files = sorted([f for f in viz_folder.glob("*.png")])  # Adjust extension if needed
    
#     if not image_files:
#         st.warning("No visualizations found in the folder.")
#         return
        
#     # Display each image with its information
#     for img_path in image_files:
#         # Create a nice title from the filename
#         title = img_path.stem.replace("_", " ").title()
#         st.subheader(title)
        
#         # Display the image
#         st.image(str(img_path), use_column_width=True)
        
#         # Add a separator between images
#         st.markdown("---")

# Add these functions after your pipeline function but before main()

def calculate_crop_areas(predictions_df, field_data):
    """
    Calculate the area covered by each crop type using Sentinel-2's 10m resolution.
    Each pixel represents 100 square meters (10m x 10m).
    
    Parameters:
        predictions_df: DataFrame containing crop predictions
        field_data: DataFrame containing field locations and information
    
    Returns:
        DataFrame containing area statistics for each crop type
    """
    # Initialize dictionary to store areas
    crop_areas = {i: 0 for i in range(1, 8)}
    
    # For each field
    for field_id in predictions_df['Field_ID']:
        # Get predicted crop probabilities for this field
        probs = predictions_df[predictions_df['Field_ID'] == field_id].iloc[0]
        crop_type = np.argmax([probs[f'Crop_ID_{i}'] for i in range(1, 8)]) + 1
        
        # Get field size (number of pixels)
        field_pixels = len(field_data[field_data['fid'] == field_id])
        
        # Calculate area in hectares (1 pixel = 100 sq meters, 1 hectare = 10000 sq meters)
        area_hectares = (field_pixels * 100) / 10000
        
        # Add to corresponding crop type
        crop_areas[crop_type] += area_hectares
    
    # Create DataFrame with results
    area_df = pd.DataFrame({
        'Crop Type': [crop_types[i] for i in range(1, 8)],
        'Area (hectares)': [crop_areas[i] for i in range(1, 8)]
    })
    
    return area_df

def count_fields_per_tile(predictions_df, field_data):
    """
    Count the number of fields for each crop type in each tile.
    
    Parameters:
        predictions_df: DataFrame containing crop predictions
        field_data: DataFrame containing field locations and information
    
    Returns:
        DataFrame containing field counts by tile and crop type
    """
    # Initialize dictionary to store counts
    tile_crop_counts = {}
    
    # Get unique tiles
    unique_tiles = sorted(field_data['tile'].unique())
    
    # For each tile
    for tile in unique_tiles:
        tile_fields = field_data[field_data['tile'] == tile]['fid']
        crop_counts = {i: 0 for i in range(1, 8)}
        
        # Count fields for each crop type
        for field_id in tile_fields:
            if str(field_id) in predictions_df['Field_ID'].values:
                probs = predictions_df[predictions_df['Field_ID'] == str(field_id)].iloc[0]
                crop_type = np.argmax([probs[f'Crop_ID_{i}'] for i in range(1, 8)]) + 1
                crop_counts[crop_type] += 1
        
        tile_crop_counts[tile] = crop_counts
    
    # Create multi-index DataFrame for better organization
    index_tuples = [(tile, crop_types[crop]) 
                    for tile in unique_tiles 
                    for crop in range(1, 8)]
    index = pd.MultiIndex.from_tuples(index_tuples, names=['Tile', 'Crop Type'])
    
    # Flatten the counts into a list
    counts = [tile_crop_counts[tile][crop] 
             for tile in unique_tiles 
             for crop in range(1, 8)]
    
    count_df = pd.DataFrame({'Number of Fields': counts}, index=index)
    
    return count_df

def main():
    st.set_page_config(layout="wide", page_title="Satellite Imagery Crop Classification")
    
    st.title("🛰️ Crop Classification Analysis")
    st.write("""
    This application analyzes satellite imagery to identify and classify crop patterns in Kenya. 
    Upload your satellite data to discover detailed insights about crop distribution and agricultural land use.
    """)
    
    uploaded_file = st.file_uploader("Upload your satellite imagery ZIP file", type="zip")
    
    directory = 'visualizations'
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    if uploaded_file:
        try:
            with st.spinner('Analyzing satellite imagery... This may take a few moments.'):
                pipeline(uploaded_file.name)
            
            st.success("Analysis complete! Explore the results below.")
            
            overview_tab, area_tab, visualization_tab = st.tabs([
                "📊 Overview", 
                "📈 Detailed Analysis",
                "🗺️ Satellite View"
            ])
            
            try:
                # First, get the current working directory
                # current_dir = os.getcwd()

                # # Create path to processed_data folder within current directory
                # processed_data_dir = os.path.join(current_dir, "processed_data")

                # # Create paths to specific files
                # field_data_path = os.path.join(processed_data_dir, "df_all.csv")
                # predictions_path = os.path.join(processed_data_dir, "SampleSub.csv")

                # Now read the files using these paths
                field_data = pd.read_csv('try_this.csv')
                predictions = pd.read_csv('SampleSub.csv')
                area_df = calculate_crop_areas(predictions, field_data)
                
                with overview_tab:
                    st.header("Agricultural Overview")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        total_area = area_df['Area (hectares)'].sum()
                        st.metric(
                            "Total Cultivated Area",
                            f"{total_area:,.2f} ha",
                            help="Total agricultural land analyzed in hectares"
                        )
                    
                    with metric_col2:
                        total_fields = len(predictions)
                        st.metric(
                            "Number of Fields",
                            f"{total_fields:,}",
                            help="Total number of distinct agricultural fields identified"
                        )
                    
                    with metric_col3:
                        avg_field_size = total_area / total_fields
                        st.metric(
                            "Average Field Size",
                            f"{avg_field_size:.2f} ha",
                            help="Average size of agricultural fields"
                        )
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("Crop Distribution Overview")
                        # Calculate percentages
                        total_area = area_df['Area (hectares)'].sum()
                        area_df['Percentage'] = area_df['Area (hectares)'] / total_area * 100

                        # Create Plotly pie chart
                        fig = px.pie(
                            area_df, 
                            values='Area (hectares)', 
                            names='Crop Type',
                            title='Distribution of Crop Types by Area',
                            color_discrete_sequence=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']
                        )

                        fig.update_traces(
                            texttemplate='%{label}<br>%{percent}',
                            textposition='inside'
                        )

                        fig.update_layout(
                            title_x=0.5,
                            title_font_size=16
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Top Crops by Area")
                        summary_df = area_df.copy()
                        summary_df['Area (hectares)'] = summary_df['Area (hectares)'].apply(lambda x: f'{x:,.2f}')
                        summary_df['Percentage'] = (area_df['Area (hectares)'] / total_area * 100).apply(lambda x: f'{x:.1f}%')
                        summary_df = summary_df.reset_index(drop=True)
                        st.table(summary_df)
                
                with area_tab:
                    st.header("Detailed Crop Analysis")
                    st.write("""
                    Explore the detailed breakdown of crop distribution and patterns across the analyzed region.
                    This analysis helps understand land use patterns and agricultural diversity.
                    """)
                    
                    # Create Plotly bar chart
                    fig = px.bar(
                        area_df.sort_values('Area (hectares)', ascending=False), 
                        x='Crop Type', 
                        y='Area (hectares)',
                        title='Area Coverage by Crop Type',
                        color='Crop Type',
                        color_discrete_sequence=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']
                    )

                    fig.update_layout(
                        xaxis_title='Crop Type',
                        yaxis_title='Area (hectares)',
                        title_x=0.5,
                        xaxis_tickangle=-45,
                        height=600
                    )

                    fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Key Observations")
                    dominant_crop = area_df.iloc[area_df['Area (hectares)'].argmax()]
                    intercropping_area = area_df[area_df['Crop Type'].str.contains('intercropping')]['Area (hectares)'].sum()
                    intercropping_percentage = (intercropping_area / total_area) * 100
                    
                    st.write(f"""
                    - The dominant crop is **{dominant_crop['Crop Type']}**, covering {dominant_crop['Area (hectares)']:,.2f} hectares
                    - Intercropping practices are used on {intercropping_area:,.2f} hectares ({intercropping_percentage:.1f}% of total area)
                    - The average field size varies by crop type, indicating different farming practices
                    """)
                    
            
                with visualization_tab:
                    st.header("Satellite Imagery Analysis")
                    st.write("""
                    These visualizations show the spatial distribution of crops across the analyzed region.
                    Each point represents a field, color-coded by crop type.
                    """)
                    
                    timing_info = st.empty()
                    st.write("Processing tiles...")
                    progress_bar = st.progress(0)
                    
                    try:
                        # Get unique tiles from our field data
                        tiles = sorted(field_data['tile'].unique())
                        date = uploaded_file.name.split('_')[-1].split('.')[0]
                        
                        # Create a container for our visualizations
                        viz_container = st.container()
                        
                        # Process each tile
                        for i, tile in enumerate(tiles):
                            tile_str = str(int(tile)).zfill(2)
                            
                            with Timer(f"Processing tile {tile_str}") as timer:
                                # Load satellite image directly from Hub
                                rgb_image, shape = fast_load_satellite_image(tile_str, date)
                                
                                if rgb_image is not None:
                                    with viz_container:
                                        st.subheader(f"Tile {tile_str} Analysis")
                                        
                                        # Create visualization
                                        fig = create_seaborn_visualization(
                                            rgb_image, 
                                            field_data, 
                                            predictions,
                                            int(tile), 
                                            tile_str
                                        )
                                        
                                        # Display figure directly in Streamlit
                                        st.pyplot(fig)
                                        
                                        # Clean up memory
                                        plt.close(fig)
                                        
                                        # Add visual separator
                                        st.markdown("---")
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(tiles))
                        
                        progress_bar.empty()
                        st.success("All tiles processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing visualizations: {str(e)}")
                        st.write("Please check the input data and try again.")   
                        
            except Exception as e:
                st.error(f"Error processing visualizations: {str(e)}")
                st.write("Please check the input data and try again.")
        
        except Exception as e:
            st.error(f"Error in processing: {str(e)}")
            st.write("There was an error processing your file. Please ensure it's in the correct format and try again.")

if __name__ == "__main__":
    main()