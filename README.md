# AI CV Upgrader (CLI + Streamlit GUI + Cover Letter)

This project reads your CV (DOCX or PDF), fetches a job description from a URL (or you can paste it manually), and uses OpenAI to generate a tailored, ATS-friendly upgraded CV. It can also generate a matching cover letter.

## Features
- Read **DOCX** or **PDF** CVs
- Fetch **job description** from a URL (basic scrape) or paste text directly
- Generate **tailored CV** using GPT
- Generate an **optional cover letter**
- Save output to **DOCX**
- **Streamlit GUI** and **CLI**

> ⚠️ Note: Scraping may not work for every site (due to HTML structure/anti-bot/ToS). If scraping fails, paste the job description manually.

## Quick Start

### 1) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate      # Windows PowerShell
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add your OpenAI key
Copy `.env.example` to `.env` and paste your key:
```env
OPENAI_API_KEY=your_api_key_here
```

### 4) Run the CLI
```bash
python cv_upgrader.py --cv ./sample_cv.docx --job-url "https://example.com/job"
# Or paste the job description:
python cv_upgrader.py --cv ./sample_cv.docx --job-text "Product designer role focusing on..."
# Add a cover letter:
python cv_upgrader.py --cv ./sample_cv.docx --job-url "https://example.com/job" --cover-letter
```

Output files are saved next to your script: `Upgraded_CV.docx` and optionally `Cover_Letter.docx`.

### 5) Run the Streamlit GUI
```bash
streamlit run app.py
```
Then follow the prompts in your browser. You can upload a CV, paste a job link or job text, and download the generated DOCX files.

## Notes
- If your PDF CV text extracts poorly, convert it to DOCX (you can use Google Docs or Word) for more reliable results.
- Keep temperature low (0.2–0.4) for consistent, professional output.
- Always proofread the generated CV/letter before sending.

## License
MIT
