# 🔧 Setup Instructions for Dataset Comparison

## ⚠️ IMPORTANT: Update File Paths Before Running

Both Python scripts contain **hardcoded absolute paths** that you must update based on your system.

---

## 📝 Step-by-Step Setup

### 1. Update `convert_markdown_to_jsonl.py`

Open the file and find **lines 48-49** (in the `main()` function):

```python
# BEFORE (example - your path may be different):
input_file = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/collected_dataset.md')
output_file = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/real_dataset.jsonl')
```

**Replace with YOUR actual project path:**

```python
# AFTER (update with YOUR path):
input_file = Path('YOUR_PROJECT_PATH/datasets_comparison/collected_dataset.md')
output_file = Path('YOUR_PROJECT_PATH/datasets_comparison/real_dataset.jsonl')
```

**Path Examples:**
- **Windows**: `C:/Users/YourName/Projects/Synthetic_Review_Generator`
- **Linux/Mac**: `/home/yourname/projects/Synthetic_Review_Generator`

---

### 2. Update `compare_datasets.py`

Open the file and find **lines 129-132** (in the `main()` function):

```python
# BEFORE (example - your path may be different):
real_dataset = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/real_dataset.jsonl')
synthetic_dataset = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/dataset.jsonl')
output_dir = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/comparison_results')
```

**Replace with YOUR actual project path:**

```python
# AFTER (update with YOUR path):
real_dataset = Path('YOUR_PROJECT_PATH/datasets_comparison/real_dataset.jsonl')
synthetic_dataset = Path('YOUR_PROJECT_PATH/datasets_comparison/dataset.jsonl')
output_dir = Path('YOUR_PROJECT_PATH/datasets_comparison/comparison_results')
```

---

## 🚀 How to Find Your Project Path

### Windows:
1. Open File Explorer
2. Navigate to your project folder
3. Click on the address bar
4. Copy the path (e.g., `E:\Projects\Synthetic_Review_Generator`)
5. **Important**: Replace backslashes `\` with forward slashes `/`
   - From: `E:\Projects\Synthetic_Review_Generator`
   - To: `E:/Projects/Synthetic_Review_Generator`

### Linux/Mac:
1. Open Terminal
2. Navigate to your project folder: `cd path/to/project`
3. Run: `pwd`
4. Copy the output (e.g., `/home/user/projects/Synthetic_Review_Generator`)

---

## ✅ After Updating Paths

Run the scripts as documented in the main README:

```bash
# Step 1: Convert markdown to JSONL
python convert_markdown_to_jsonl.py

# Step 2: Run comparison
python compare_datasets.py
```

---

## 🆘 Troubleshooting

### Error: "No such file or directory"
- ✅ Double-check that your paths are correct
- ✅ Make sure you used forward slashes `/` (not backslashes `\`)
- ✅ Verify the file exists at that location

### Error: "ModuleNotFoundError"
```bash
pip install textblob matplotlib seaborn
```

### Still Having Issues?
1. Make sure you're running from the `datasets_comparison/` folder
2. Check that `collected_dataset.md` exists
3. Verify you copied `dataset.jsonl` from the datasets folder
