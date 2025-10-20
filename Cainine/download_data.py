"""
Download Canine ECG Data from Google Drive

This script downloads the raw canine ECG data (all_data.zip) from Google Drive
and extracts it to the correct location for preprocessing.

Usage:
    python download_data.py

Requirements:
    - gdown (install with: pip install gdown)
    - At least 1.5 GB free disk space
"""

import os
import sys
import zipfile
import shutil

try:
    import gdown
except ImportError:
    print("ERROR: gdown library not found!")
    print("Please install it with: pip install gdown")
    print("Or install all requirements: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Drive file ID (from the shareable link)
GDRIVE_FILE_ID = "1WTGEGKsR8sfieb2B5f6dJ1Swgu7FIecg"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Download and extraction paths
DOWNLOAD_FILE = os.path.join(SCRIPT_DIR, "all_data.zip")
EXTRACT_DIR = SCRIPT_DIR
TARGET_FOLDER = os.path.join(SCRIPT_DIR, "all_data")

# Expected minimum size (in bytes) - adjust based on actual file size
MIN_EXPECTED_SIZE = 200 * 1024 * 1024  # 200 MB (compressed)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_size(size_bytes):
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def check_disk_space(required_gb=2.0):
    """Check if enough disk space is available"""
    stat = shutil.disk_usage(SCRIPT_DIR)
    free_gb = stat.free / (1024 ** 3)

    if free_gb < required_gb:
        print(f"WARNING: Low disk space!")
        print(f"  Free space: {free_gb:.2f} GB")
        print(f"  Required: ~{required_gb:.2f} GB")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    return True


def download_file():
    """Download file from Google Drive"""
    print("="*70)
    print("DOWNLOADING CANINE ECG DATA FROM GOOGLE DRIVE")
    print("="*70)
    print(f"\nSource: Google Drive")
    print(f"Destination: {DOWNLOAD_FILE}")
    print(f"\nThis may take several minutes depending on your connection...")
    print()

    try:
        # Download with progress bar
        gdown.download(GDRIVE_URL, DOWNLOAD_FILE, quiet=False)

        # Verify download
        if not os.path.exists(DOWNLOAD_FILE):
            print("\nERROR: Download failed - file not found")
            return False

        file_size = os.path.getsize(DOWNLOAD_FILE)
        print(f"\n✓ Download complete!")
        print(f"  File size: {format_size(file_size)}")

        # Basic size check
        if file_size < MIN_EXPECTED_SIZE:
            print(f"\nWARNING: Downloaded file seems too small!")
            print(f"  Expected at least: {format_size(MIN_EXPECTED_SIZE)}")
            print(f"  Got: {format_size(file_size)}")
            response = input("Continue with extraction anyway? (y/N): ")
            if response.lower() != 'y':
                return False

        return True

    except Exception as e:
        print(f"\nERROR during download: {e}")
        return False


def extract_zip():
    """Extract downloaded zip file"""
    print("\n" + "="*70)
    print("EXTRACTING DATA")
    print("="*70)
    print(f"\nExtracting to: {EXTRACT_DIR}")
    print("This may take a few minutes...")

    try:
        with zipfile.ZipFile(DOWNLOAD_FILE, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"Found {total_files} files in archive")

            # Extract with progress
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, EXTRACT_DIR)
                if i % 5 == 0 or i == total_files:  # Update every 5 files
                    print(f"  Extracted {i}/{total_files} files...", end='\r')

            print(f"\n✓ Extraction complete!")

        return True

    except zipfile.BadZipFile:
        print("\nERROR: Downloaded file is not a valid zip file!")
        print("The download may have been corrupted.")
        return False
    except Exception as e:
        print(f"\nERROR during extraction: {e}")
        return False


def cleanup_zip():
    """Remove downloaded zip file to save space"""
    if os.path.exists(DOWNLOAD_FILE):
        try:
            file_size = os.path.getsize(DOWNLOAD_FILE)
            os.remove(DOWNLOAD_FILE)
            print(f"\n✓ Cleaned up zip file (freed {format_size(file_size)})")
        except Exception as e:
            print(f"\nWARNING: Could not remove zip file: {e}")


def verify_data():
    """Verify that data was extracted correctly"""
    print("\n" + "="*70)
    print("VERIFYING DATA")
    print("="*70)

    if not os.path.exists(TARGET_FOLDER):
        print(f"\nERROR: Target folder not found: {TARGET_FOLDER}")
        return False

    # Count CSV files
    csv_files = [f for f in os.listdir(TARGET_FOLDER) if f.endswith('.csv')]
    signal_files = [f for f in csv_files if not f.endswith('_labels.csv')]
    label_files = [f for f in csv_files if f.endswith('_labels.csv')]

    print(f"\nData folder: {TARGET_FOLDER}")
    print(f"  Signal files: {len(signal_files)}")
    print(f"  Label files: {len(label_files)}")
    print(f"  Total CSV files: {len(csv_files)}")

    # Basic sanity check
    if len(signal_files) == 0 or len(label_files) == 0:
        print("\nWARNING: Expected to find signal and label files!")
        return False

    if len(signal_files) != len(label_files):
        print(f"\nWARNING: Mismatch between signal and label files!")
        print(f"  Signal files: {len(signal_files)}")
        print(f"  Label files: {len(label_files)}")

    # Show examples
    print(f"\nExample files:")
    for f in sorted(csv_files)[:6]:
        file_path = os.path.join(TARGET_FOLDER, f)
        size = format_size(os.path.getsize(file_path))
        print(f"  - {f} ({size})")

    if len(csv_files) > 6:
        print(f"  ... and {len(csv_files) - 6} more files")

    print("\n✓ Data verification complete!")
    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main download and setup process"""
    print("\n" + "="*70)
    print("CANINE ECG DATA DOWNLOAD SCRIPT")
    print("="*70)
    print(f"\nThis script will download and extract the raw canine ECG data.")
    print(f"The data will be placed in: {TARGET_FOLDER}")

    # Check if data already exists
    if os.path.exists(TARGET_FOLDER):
        print(f"\nWARNING: Data folder already exists!")
        print(f"Location: {TARGET_FOLDER}")
        response = input("\nDo you want to re-download and replace it? (y/N): ")
        if response.lower() != 'y':
            print("\nDownload cancelled.")
            print("If you need to re-download, delete the existing 'all_data' folder first.")
            return 0
        else:
            print(f"\nRemoving existing data folder...")
            try:
                shutil.rmtree(TARGET_FOLDER)
                print("✓ Old data removed")
            except Exception as e:
                print(f"ERROR: Could not remove old data: {e}")
                return 1

    # Check disk space
    if not check_disk_space():
        print("\nDownload cancelled due to insufficient disk space.")
        return 1

    # Download
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING FROM GOOGLE DRIVE")
    print("="*70)

    if not download_file():
        print("\n✗ Download failed!")
        return 1

    # Extract
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING DATA")
    print("="*70)

    if not extract_zip():
        print("\n✗ Extraction failed!")
        cleanup_zip()
        return 1

    # Verify
    if not verify_data():
        print("\n✗ Verification failed!")
        cleanup_zip()
        return 1

    # Cleanup
    print("\n" + "="*70)
    print("STEP 3: CLEANUP")
    print("="*70)
    cleanup_zip()

    # Success!
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE!")
    print("="*70)
    print(f"\nThe canine ECG data is ready to use at:")
    print(f"  {TARGET_FOLDER}")
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing: python DataPreProcessing/enhanced_preprocessing.py")
    print(f"  2. Create train/test split: python DataPreProcessing/create_train_test_split.py")
    print(f"  3. Train models in 'Transfer of Learning' folder")
    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("You can run this script again to resume/restart.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
