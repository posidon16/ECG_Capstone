"""
ECG Capstone - Centralized Data Download Script

Downloads the datasets from Google Drive that are too big for GitHub:
1. Canine ECG raw data (all_data.zip)
2. MIT-BIH ECG data (mitbih_data.zip) - train and test CSVs

Usage:
    python download_data.py              # Download all datasets
    python download_data.py --canine     # Download only canine data
    python download_data.py --mitbih     # Download only MIT-BIH data

Requirements:
    - gdown (install with: pip install gdown)
    - At least 3 GB free disk space
"""

import os
import sys
import zipfile
import shutil
import argparse

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

# Get script directory (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset configurations
DATASETS = {
    'canine': {
        'name': 'Canine ECG Raw Data',
        'gdrive_id': '1WTGEGKsR8sfieb2B5f6dJ1Swgu7FIecg',
        'zip_file': 'all_data.zip',
        'extract_to': os.path.join(PROJECT_ROOT, 'Cainine'),
        'target_folder': os.path.join(PROJECT_ROOT, 'Cainine', 'all_data'),
        'expected_files': '*.csv (signal and label files)',
        'size_mb': 600,
    },
    'mitbih': {
        'name': 'MIT-BIH ECG Data',
        'gdrive_id': '1i9qnTWU5Xk82iIMt3WVtBjAyzk3_TLD1',
        'zip_file': 'mitbih_data.zip',
        'extract_to': os.path.join(PROJECT_ROOT, 'MITBIH'),
        'target_folder': os.path.join(PROJECT_ROOT, 'MITBIH', 'Data'),
        'expected_files': 'mitbih_train.csv, mitbih_test.csv',
        'size_mb': 500,
    }
}

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


def check_disk_space(required_gb=3.0):
    """Check if enough disk space is available"""
    stat = shutil.disk_usage(PROJECT_ROOT)
    free_gb = stat.free / (1024 ** 3)

    if free_gb < required_gb:
        print(f"WARNING: Low disk space!")
        print(f"  Free space: {free_gb:.2f} GB")
        print(f"  Required: ~{required_gb:.2f} GB")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    return True


def download_dataset(dataset_key):
    """Download a specific dataset"""
    dataset = DATASETS[dataset_key]

    print("\n" + "="*70)
    print(f"DOWNLOADING: {dataset['name']}")
    print("="*70)

    zip_path = os.path.join(PROJECT_ROOT, dataset['zip_file'])
    gdrive_url = f"https://drive.google.com/uc?id={dataset['gdrive_id']}"

    # Check if target folder already exists
    if os.path.exists(dataset['target_folder']):
        print(f"\nWARNING: Data folder already exists!")
        print(f"Location: {dataset['target_folder']}")
        response = input(f"\nRe-download and replace? (y/N): ")
        if response.lower() != 'y':
            print(f"Skipping {dataset['name']}")
            return True
        else:
            print(f"Removing existing data...")
            try:
                shutil.rmtree(dataset['target_folder'])
                print("✓ Old data removed")
            except Exception as e:
                print(f"ERROR: Could not remove old data: {e}")
                return False

    # Download
    print(f"\nDownloading from Google Drive...")
    print(f"Destination: {zip_path}")
    print(f"Expected size: ~{dataset['size_mb']} MB")
    print(f"This may take several minutes...\n")

    try:
        gdown.download(gdrive_url, zip_path, quiet=False)

        if not os.path.exists(zip_path):
            print(f"\nERROR: Download failed - file not found")
            return False

        file_size = os.path.getsize(zip_path)
        print(f"\n✓ Download complete! ({format_size(file_size)})")

    except Exception as e:
        print(f"\nERROR during download: {e}")
        return False

    # Extract
    print(f"\n" + "="*70)
    print(f"EXTRACTING: {dataset['name']}")
    print("="*70)
    print(f"Extracting to: {dataset['extract_to']}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"Found {total_files} files in archive")

            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, dataset['extract_to'])
                if i % 5 == 0 or i == total_files:
                    print(f"  Extracted {i}/{total_files} files...", end='\r')

            print(f"\n✓ Extraction complete!")

    except zipfile.BadZipFile:
        print(f"\nERROR: Not a valid zip file!")
        return False
    except Exception as e:
        print(f"\nERROR during extraction: {e}")
        return False

    # Cleanup
    try:
        os.remove(zip_path)
        print(f"✓ Cleaned up zip file")
    except Exception as e:
        print(f"WARNING: Could not remove zip file: {e}")

    # Verify and handle files extracted to wrong location
    if not os.path.exists(dataset['target_folder']):
        # Check if files were extracted to parent directory instead
        extract_parent = dataset['extract_to']
        print(f"\n⚠ Target folder not found at expected location")
        print(f"  Checking parent directory: {extract_parent}")

        if os.path.exists(extract_parent):
            parent_files = [f for f in os.listdir(extract_parent)
                           if f.endswith('.csv') and os.path.isfile(os.path.join(extract_parent, f))]

            if parent_files:
                print(f"  Found {len(parent_files)} CSV files in parent directory")
                print(f"  Moving files into {os.path.basename(dataset['target_folder'])}/...")

                # Create target folder
                os.makedirs(dataset['target_folder'], exist_ok=True)

                # Move files
                for file in parent_files:
                    src = os.path.join(extract_parent, file)
                    dst = os.path.join(dataset['target_folder'], file)
                    shutil.move(src, dst)
                    print(f"    Moved: {file}")

                print(f"  ✓ Files moved to correct location")
            else:
                print(f"\nERROR: No CSV files found in parent directory!")
                return False
        else:
            print(f"\nERROR: Extraction directory not found!")
            return False

    # Final verification
    if os.path.exists(dataset['target_folder']):
        files = os.listdir(dataset['target_folder'])
        print(f"\n✓ Verification: Found {len(files)} files in {os.path.basename(dataset['target_folder'])}/")

        # Show some examples
        print(f"  Expected: {dataset['expected_files']}")
        if files:
            print(f"  Examples:")
            for f in sorted(files)[:4]:
                file_path = os.path.join(dataset['target_folder'], f)
                if os.path.isfile(file_path):
                    size = format_size(os.path.getsize(file_path))
                    print(f"    - {f} ({size})")
            if len(files) > 4:
                print(f"    ... and {len(files) - 4} more")

        return True
    else:
        print(f"\nERROR: Target folder still not found after attempting to fix!")
        return False


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main download process"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download ECG datasets from Google Drive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py              # Download all datasets
  python download_data.py --canine     # Download only canine data
  python download_data.py --mitbih     # Download only MIT-BIH data
        """
    )
    parser.add_argument('--canine', action='store_true',
                       help='Download only canine ECG data')
    parser.add_argument('--mitbih', action='store_true',
                       help='Download only MIT-BIH ECG data')

    args = parser.parse_args()

    # Determine which datasets to download
    if args.canine and args.mitbih:
        datasets_to_download = ['canine', 'mitbih']
    elif args.canine:
        datasets_to_download = ['canine']
    elif args.mitbih:
        datasets_to_download = ['mitbih']
    else:
        # Default: download all
        datasets_to_download = ['canine', 'mitbih']

    # Header
    print("\n" + "="*70)
    print("ECG CAPSTONE - DATA DOWNLOAD SCRIPT")
    print("="*70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"\nDatasets to download:")
    for key in datasets_to_download:
        print(f"  - {DATASETS[key]['name']} (~{DATASETS[key]['size_mb']} MB)")

    # Check disk space
    total_size_gb = sum(DATASETS[key]['size_mb'] for key in datasets_to_download) / 1024 + 1
    if not check_disk_space(total_size_gb):
        print("\nDownload cancelled due to insufficient disk space.")
        return 1

    # Download each dataset
    success_count = 0
    for dataset_key in datasets_to_download:
        if download_dataset(dataset_key):
            success_count += 1
        else:
            print(f"\n✗ Failed to download {DATASETS[dataset_key]['name']}")

    # Summary
    print("\n" + "="*70)
    if success_count == len(datasets_to_download):
        print("✓ ALL DOWNLOADS COMPLETE!")
        print("="*70)
        print(f"\nSuccessfully downloaded {success_count}/{len(datasets_to_download)} datasets")

        print("\nData locations:")
        for key in datasets_to_download:
            print(f"  {DATASETS[key]['name']}:")
            print(f"    → {DATASETS[key]['target_folder']}")
        print()
        return 0
    else:
        print("✗ SOME DOWNLOADS FAILED")
        print("="*70)
        print(f"\nSuccessfully downloaded {success_count}/{len(datasets_to_download)} datasets")
        print("\nPlease check error messages above and try again.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("You can run this script again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
