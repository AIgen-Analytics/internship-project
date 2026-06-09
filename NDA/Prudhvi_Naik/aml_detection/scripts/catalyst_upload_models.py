import os
import zcatalyst_sdk

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

# The ID of the folder in Catalyst File Store where models will be saved.
# Note: You MUST create this folder in the Catalyst Console first and paste the ID here.
FOLDER_ID = int(os.environ.get("CATALYST_MODEL_FOLDER_ID", "1234567890"))

app = zcatalyst_sdk.initialize()
filestore = app.filestore()

def upload_model(filename):
    print(f"Uploading {filename} to Catalyst File Store...")
    folder = filestore.folder(FOLDER_ID)
    
    file_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
        
    try:
        with open(file_path, 'rb') as f:
            # zcatalyst SDK expects a file stream or open file object for upload
            uploaded_file = folder.upload_file(f)
            
        print(f"✅ Successfully uploaded {filename}.")
        print(f"File ID: {uploaded_file.get_id()}")
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")

if __name__ == "__main__":
    print("--- Catalyst Model Uploader Script ---")
    if FOLDER_ID == 1234567890:
        print("⚠️ Please set CATALYST_MODEL_FOLDER_ID environment variable with your actual Folder ID.")
    else:
        upload_model('lgbm_is_aml.pkl')
        upload_model('lgbm_typology.pkl')
        upload_model('label_encoder_typology.pkl')
        
    print("Upload script execution finished.")
