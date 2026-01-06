import io
import os
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pdfplumber

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-large").strip()

def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        out = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                if t.strip():
                    out.append(t)
        return "\n".join(out)
    raise HTTPException(400, "Unsupported file type. Use PDF or TXT.")

def build_prompt(project_name: str, output: str, guide_type: str, brochure_text: str) -> str:
    base = f"""You are a Bayut content expert.
Rules:
- Use ONLY the brochure text provided.
- Do NOT hallucinate.
- If something is missing, list it under: MISSING INFORMATION.
PROJECT NAME: {project_name}
"""
    parts = []
    if output in ("lpv", "both"):
        parts.append("A) LPV DESCRIPTION (short, conversion-focused)")
    if output in ("guide", "both"):
        if guide_type == "area":
            parts.append("""B) AREA GUIDE using ONLY these headers:
HIGHLIGHTS
ABOUT <PROJECT NAME>
IN A NUTSHELL
PROPERTY
PAYMENT PLAN
LOCATION
FAQs ABOUT <PROJECT NAME>""")
        else:
            parts.append("""B) BUILDING GUIDE using ONLY these headers:
HIGHLIGHTS
ABOUT <BUILDING NAME>
IN A NUTSHELL
BUILDING DETAILS
TYPES OF UNITS
AMENITIES
TRANSPORTATION NEAR <BUILDING NAME>
NEARBY AMENITIES
THINGS TO CONSIDER
FAQs ABOUT <BUILDING NAME>""")
    return base + "\nTASKS:\n" + "\n".join(parts) + "\n\nBROCHURE TEXT:\n" + brochure_text

def call_hf(prompt: str) -> str:
    if not HF_TOKEN:
        raise HTTPException(500, "HF_TOKEN is not set.")
    url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("generated_text", "")
    return str(data)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/process_project")
async def process_project(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    output: str = Form(...),
    guide_type: str = Form("area")
):
    output = output.lower().strip()
    guide_type = guide_type.lower().strip()

    if output not in ("lpv", "guide", "both"):
        raise HTTPException(400, "output must be lpv, guide, or both")
    if output in ("guide", "both") and guide_type not in ("area", "building"):
        raise HTTPException(400, "guide_type must be area or building")

    data = await file.read()
    brochure_text = extract_text(file.filename, data)
    if not brochure_text.strip():
        raise HTTPException(400, "No text extracted from file.")

    prompt = build_prompt(project_name, output, guide_type, brochure_text)
    generated = call_hf(prompt)

    return {
        "status": "success",
        "project_name": project_name,
        "output": output,
        "guide_type": guide_type,
        "content": generated
    }
