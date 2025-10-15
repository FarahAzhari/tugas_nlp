# Advanced NLP Assignment

This repository contains the code for an assignment in the **Advanced Natural Language Processing (NLP)** class. The project demonstrates concepts and techniques related to NLP using Python.

---

## 🚀 Getting Started

Follow the steps below to set up and run the project on your local machine.

### 1. Install Python

It's recommended to use **Python 3.11** for this project.

- [Download Python 3.11 here](https://www.python.org/downloads/release/python-3110/)

Make sure Python and `pip` are available in your system path:
```bash
python --version
pip --version
```

### 2. Create a Virtual Environment

You can create a virtual environment in two ways:

✅ General method:
```bash
python -m venv nlp-env
```

✅ Or using a specific Python version (if you have multiple versions installed):
```bash
python3.11 -m venv nlp-env
```

Then activate the environment:

On Windows:
```bash
nlp-env\Scripts\activate
```

On macOS/Linux:
```bash
source nlp-env/bin/activate
```

### 3. Install Required Libraries

With the virtual environment activated, install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Navigate to the scripts directory and run any script you'd like. For example, to run search_engine.py:

```bash
cd scripts
python search_engine.py
```

You can replace search_engine.py with any other script depending on what you want to test or run.