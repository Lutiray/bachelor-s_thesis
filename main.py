import os
import importlib.util

# List of directories and their corresponding Python scripts
directories = {
    "Preprocessing": ["gauss_filter.py", "median_filter.py", "cnn_denoise_saltPepperNois.py", "cnn_denoise_gaussNois.py"],
    "Segmentation": ["watershed.py", "unet_segmentation.py"],
    "Metrics": ["calculate_denoisMetrics.py", "calculate_segmentMetrics.py", "classification_test.py"],
    "Utils": ["image_loader.py", "image_saver.py"],
    "Model_NN": ["CNN_training_gausNois.py", "CNN_training_saltPepper.py", "ResNet50_training.py", "unet_training.py"]
}

def check_compilation(directory, filename):
    """Checks if a Python file compiles without errors."""
    file_path = os.path.join(directory, filename)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    try:
        spec = importlib.util.spec_from_file_location(filename, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Compiling the script
        print(f"{filename} compiles successfully!")
        return True
    except Exception as e:
        print(f"Error in {filename}: {e}")
        return False

if __name__ == "__main__":
    print("Checking all scripts for compilation errors...\n")
    all_passed = True

    for folder, scripts in directories.items():
        for script in scripts:
            if not check_compilation(folder, script):
                all_passed = False

    if all_passed:
        print("\nAll scripts compiled successfully!")
    else:
        print("\nSome scripts contain errors. Check logs above.")

def check_modules():
    """Checks if all required modules are loaded."""
    print("=== Module check ===")
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"[OK] {module} is successfully loaded.")
        except ImportError as e:
            print(f"[ERROR] Failed to load {module}: {e}")

# Basic check
if __name__ == "__main__":
    check_modules()
    print("Ready. The code can be run.")
