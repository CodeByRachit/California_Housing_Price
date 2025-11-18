import os

# Root directory name
project_root = "Project_01_Housing_Prices"

# Directory structure
structure = {
    "": ["README.md", "requirements.txt"],
    "data": ["california_housing.csv", "processed_data.csv"],
    "notebooks": [
        "01_data_exploration.ipynb",
        "02_closed_form.ipynb",
        "03_gradient_descent.ipynb",
        "04_bias_variance.ipynb"
    ],
    "src": [
        "model_closed_form.py",
        "model_gd.py",
        "utils.py"
    ],
    "results": [
        "metrics_comparison.csv"
    ],
    "results/plots": []
}

# Template text for README and requirements
readme_content = """# Project 01 — Housing prices: Bias–Variance & GD vs Closed-form

## Overview
This project predicts California housing prices using Linear Regression.
It compares:
- Closed-form solution (Normal Equation)
- Gradient Descent (GD)

and demonstrates the Bias–Variance tradeoff.

Run notebooks in the `notebooks/` directory for exploration and results.
"""

requirements_content = """numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
"""

# Create directories and files
def create_project_structure(base_dir, structure_dict):
    for folder, files in structure_dict.items():
        dir_path = os.path.join(base_dir, folder)
        os.makedirs(dir_path, exist_ok=True)
        for f in files:
            file_path = os.path.join(dir_path, f)
            # Create empty file or add template content
            if f == "README.md":
                content = readme_content
            elif f == "requirements.txt":
                content = requirements_content
            else:
                content = ""
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            print(f"Created: {file_path}")

# Run setup
if __name__ == "__main__":
    print(f"Creating project directory: {project_root}\\n")
    create_project_structure(project_root, structure)
    print("\\n✅ Project structure created successfully!")
