# 📄 CV Customizer

A Streamlit-based web application that lets you build, manage, and AI-customize your CV and Cover Letter for specific job listings.
It integrates with a backend service (cv-automation) to store user information, extract data from uploaded CVs, and generate tailored documents using AI.

---

## 🚀 Features

- **Personal Information Management**
Add and update personal details, work experience, education, skills, languages, certifications, and projects.

- **CV & Cover Letter Customization**
    - Upload your existing CV in PDF.
    - Extract structured information automatically.
    - Provide a job listing URL, preferred tone, and creativity settings.
    - Generate a tailored CV and Cover Letter optimized for the role.

- **Template System**
  
    Uses .docx templates to populate and generate PDF previews.

- **Database Integration**
  
    Communicates with a backend API to save, fetch, and update user data.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit

- **Models & Validation**: Pydantic

- **Templates**: docxtpl

- **PDF Conversion**: LibreOffice (headless mode)

- **Environment Management**: python-dotenv

---

## 📂 Project Structure
```
.
├── public/
│   ├── cv_template.docx      # Default CV template
│   ├── _cvl_template.docx    # Default Cover Letter template
├── cv_customizer.py          # Main Streamlit app
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
```
git clone https://github.com/marcoslashpro/cv_automation_server.git
cd cv_automation_server
```
2. **Create a virtual environment & install dependencies**
  - With uv
  ```
  uv sync
  ```
  - With pip
  ```
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows
  
  pip install -r requirements.txt 
  ```

3. **Install LibreOffice** (required for `.docx → .pdf` conversion)
  
  - Linux:
  ```
  sudo apt-get install libreoffice
  ```
  
  - Mac (Homebrew):
  ```
  brew install --cask libreoffice
  ```
  
  - Windows: [Download](https://www.libreoffice.org/download/download/)

### ▶️ Running the App
```
streamlit run main.py
```
or with uv:
```
uv run streamlit run main .py
```
The app will open in your browser (default: http://localhost:8501)

---

### 🔑 Usage

- **Model Preferences**
    - Adjust creativity and tone.
    - Add optional custom instructions.

- **Add Information**
    - Fill in tabs for experience, projects, education, skills, etc. or upload an existing PDF CV to extract information.
    - Manage Personal Info by updating values

- **Customize for Job Listing**
    - Paste a job listing URL.
    - Click Customize to generate a tailored CV & Cover Letter.

- **Preview & Export**
    - View generated CV & Cover Letter as PDFs.
    - Download the results.

---

### 📌 Notes

The app relies on an external backend (cv-automation) running at:
```
http://107.21.44.255/cv-automation/cv
```

If the backend is unavailable, saving and fetching data will fail.

More templates (`cv_template.docx`, `_cvl_template.docx`) can be added under the public/ directory.

---

### 🧑‍💻 Development

To modify UI components:

- Edit `BaseUI` and its child classes (`PIIUI`, `ExperienceUI`, `SkillUI`, etc.)

- Each `UI` model maps to a backend `Pydantic` schema for validation.
