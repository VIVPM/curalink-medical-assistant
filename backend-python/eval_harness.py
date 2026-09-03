"""
eval_harness.py — Pipeline quality evaluation.

Runs a gold-standard query set against the REAL FastAPI pipeline (/pipeline/run),
scores each response on retrieval, grounding, structure, and abstention, and
writes a results report.

Cost: $0 — uses the same free-tier HF / CF calls the app normally makes.
Time: ~3-5 min for 20 queries (rate-limited by HF free tier).

Usage:
    cd backend-python
    python eval_harness.py                  # full eval
    python eval_harness.py --selftest       # check eval set validity only
    python eval_harness.py --query 3        # run a single query by index

Requires: the FastAPI server running on localhost:8000 (or --base-url).
    uvicorn main:app --port 8000
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "eval_results")

import httpx


# =============================================================================
# Gold-standard evaluation set
# =============================================================================

EVAL_SET = [
    # --- Well-studied diseases: should return rich results ---
    {
        "id": "Q01",
        "disease": "Parkinson's disease",
        "intent": "Deep Brain Stimulation",
        "location": "Toronto, Canada",
        "message": "What are the latest outcomes of deep brain stimulation for Parkinson's?",
        "expect": {
            "should_abstain": False,
            "min_insights": 2,
            "min_trials": 1,
            "topic_keywords": ["deep brain stimulation", "DBS", "motor", "subthalamic"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q02",
        "disease": "Type 2 Diabetes",
        "intent": "GLP-1 receptor agonists",
        "location": "New York, USA",
        "message": "How effective are GLP-1 receptor agonists for weight loss in type 2 diabetes?",
        "expect": {
            "should_abstain": False,
            "min_insights": 2,
            "min_trials": 0,
            "topic_keywords": ["GLP-1", "semaglutide", "liraglutide", "weight", "HbA1c"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q03",
        "disease": "Breast cancer",
        "intent": "immunotherapy",
        "location": "London, UK",
        "message": "What is the role of immune checkpoint inhibitors in triple-negative breast cancer?",
        "expect": {
            "should_abstain": False,
            "min_insights": 2,
            "min_trials": 0,
            "topic_keywords": ["checkpoint", "PD-1", "PD-L1", "triple-negative", "immunotherapy", "pembrolizumab", "atezolizumab"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q04",
        "disease": "Alzheimer's disease",
        "intent": "amyloid-targeting therapies",
        "location": "Boston, USA",
        "message": "What do recent trials say about lecanemab and donanemab for early Alzheimer's?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["lecanemab", "donanemab", "amyloid", "cognitive decline"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q05",
        "disease": "Major depressive disorder",
        "intent": "ketamine treatment",
        "location": "Melbourne, Australia",
        "message": "How does intranasal esketamine compare to traditional antidepressants for treatment-resistant depression?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["esketamine", "ketamine", "treatment-resistant", "NMDA", "spravato"],
            "source_types": ["pubmed"],
        },
    },

    # --- Clinical trials focus ---
    {
        "id": "Q06",
        "disease": "Non-small cell lung cancer",
        "intent": "targeted therapy",
        "location": "Houston, USA",
        "message": "Are there active clinical trials for EGFR-mutated non-small cell lung cancer?",
        "expect": {
            "should_abstain": False,
            "min_insights": 0,
            "min_trials": 1,
            "topic_keywords": ["EGFR", "lung cancer", "NSCLC", "osimertinib"],
            "source_types": [],
        },
    },
    {
        "id": "Q07",
        "disease": "Rheumatoid arthritis",
        "intent": "JAK inhibitors",
        "location": "Berlin, Germany",
        "message": "What are the safety concerns with JAK inhibitors like tofacitinib in rheumatoid arthritis?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["JAK", "tofacitinib", "baricitinib", "cardiovascular", "thrombosis", "safety"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q08",
        "disease": "Crohn's disease",
        "intent": "biologic therapy",
        "location": "Chicago, USA",
        "message": "How do anti-IL-23 biologics compare to anti-TNF agents for Crohn's disease?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["IL-23", "TNF", "ustekinumab", "risankizumab", "infliximab", "adalimumab"],
            "source_types": ["pubmed"],
        },
    },

    # --- Rare / niche conditions ---
    {
        "id": "Q09",
        "disease": "Spinal muscular atrophy",
        "intent": "gene therapy",
        "location": "Philadelphia, USA",
        "message": "What are the long-term outcomes of onasemnogene abeparvovec for spinal muscular atrophy?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["onasemnogene", "Zolgensma", "SMA", "gene therapy", "SMN"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q10",
        "disease": "Sickle cell disease",
        "intent": "CRISPR gene editing",
        "location": "Atlanta, USA",
        "message": "What is the current status of CRISPR-based therapies for sickle cell disease?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["CRISPR", "Cas9", "exa-cel", "Casgevy", "gene editing", "hemoglobin"],
            "source_types": ["pubmed"],
        },
    },

    # --- Should-abstain queries (non-medical or unanswerable) ---
    {
        "id": "Q11",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "What is the capital of France?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },
    {
        "id": "Q12",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "Write me a poem about the ocean.",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },

    # --- Edge cases ---
    {
        "id": "Q13",
        "disease": "Hypertension",
        "intent": "lifestyle interventions",
        "location": "Mumbai, India",
        "message": "Does the DASH diet reduce blood pressure as effectively as medication?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["DASH", "diet", "blood pressure", "hypertension", "systolic", "diastolic"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q14",
        "disease": "COVID-19",
        "intent": "long COVID treatment",
        "location": "San Francisco, USA",
        "message": "What treatments show promise for long COVID fatigue and brain fog?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["long COVID", "post-acute", "fatigue", "cognitive", "PASC"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q15",
        "disease": "Epilepsy",
        "intent": "surgical treatment",
        "location": "Seoul, South Korea",
        "message": "When is surgery recommended over medication for drug-resistant epilepsy?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["drug-resistant", "refractory", "surgery", "resection", "temporal lobe", "seizure"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q16",
        "disease": "Multiple sclerosis",
        "intent": "disease-modifying therapies",
        "location": "Stockholm, Sweden",
        "message": "How do high-efficacy therapies like ocrelizumab compare to first-line DMTs for relapsing MS?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["ocrelizumab", "DMT", "relapsing", "multiple sclerosis", "B-cell"],
            "source_types": ["pubmed"],
        },
    },

    # --- Another abstain test ---
    {
        "id": "Q17",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "How do I cook pasta al dente?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },

    # --- Broad queries ---
    {
        "id": "Q18",
        "disease": "Atopic dermatitis",
        "intent": "biologic therapy",
        "location": "Tokyo, Japan",
        "message": "What biologics are approved for moderate-to-severe atopic dermatitis and how do they compare?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["dupilumab", "tralokinumab", "IL-4", "IL-13", "eczema", "atopic"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q19",
        "disease": "Heart failure",
        "intent": "SGLT2 inhibitors",
        "location": "Paris, France",
        "message": "What evidence supports SGLT2 inhibitors for heart failure with preserved ejection fraction?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["SGLT2", "empagliflozin", "dapagliflozin", "HFpEF", "ejection fraction"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q20",
        "disease": "Migraine",
        "intent": "CGRP antibodies",
        "location": "Madrid, Spain",
        "message": "How effective are CGRP monoclonal antibodies for chronic migraine prevention?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["CGRP", "erenumab", "fremanezumab", "galcanezumab", "migraine", "prevention"],
            "source_types": ["pubmed"],
        },
    },

    # --- Pediatric / age-specific ---
    {
        "id": "Q21",
        "disease": "Acute lymphoblastic leukemia",
        "intent": "CAR-T therapy",
        "location": "Philadelphia, USA",
        "message": "What are the outcomes of CAR-T cell therapy in pediatric relapsed ALL?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["CAR-T", "tisagenlecleucel", "CD19", "leukemia", "pediatric", "remission"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q22",
        "disease": "ADHD",
        "intent": "non-stimulant treatment",
        "location": "Toronto, Canada",
        "message": "How does viloxazine compare to atomoxetine for ADHD in children?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["viloxazine", "atomoxetine", "ADHD", "non-stimulant", "norepinephrine"],
            "source_types": ["pubmed"],
        },
    },

    # --- Surgical / procedural ---
    {
        "id": "Q23",
        "disease": "Obesity",
        "intent": "bariatric surgery",
        "location": "Dallas, USA",
        "message": "What is the long-term weight loss comparison between gastric sleeve and gastric bypass?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["sleeve", "bypass", "bariatric", "weight loss", "Roux-en-Y", "gastrectomy"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q24",
        "disease": "Coronary artery disease",
        "intent": "revascularization",
        "location": "Cleveland, USA",
        "message": "When is CABG preferred over PCI for multivessel coronary artery disease?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["CABG", "PCI", "multivessel", "coronary", "revascularization", "SYNTAX"],
            "source_types": ["pubmed"],
        },
    },

    # --- Infectious disease ---
    {
        "id": "Q25",
        "disease": "Tuberculosis",
        "intent": "drug-resistant treatment",
        "location": "Cape Town, South Africa",
        "message": "What are the current regimens for multidrug-resistant tuberculosis?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["MDR-TB", "bedaquiline", "pretomanid", "linezolid", "drug-resistant"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q26",
        "disease": "Hepatitis C",
        "intent": "direct-acting antivirals",
        "location": "Cairo, Egypt",
        "message": "What is the sustained virologic response rate with current DAA regimens for hepatitis C?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["DAA", "sofosbuvir", "SVR", "hepatitis C", "virologic", "cure"],
            "source_types": ["pubmed"],
        },
    },

    # --- Mental health ---
    {
        "id": "Q27",
        "disease": "PTSD",
        "intent": "psychedelic-assisted therapy",
        "location": "Denver, USA",
        "message": "What does the clinical evidence say about MDMA-assisted therapy for PTSD?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["MDMA", "PTSD", "psychedelic", "psychotherapy", "trauma"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q28",
        "disease": "Schizophrenia",
        "intent": "long-acting injectables",
        "location": "Amsterdam, Netherlands",
        "message": "How do long-acting injectable antipsychotics compare to oral formulations for relapse prevention in schizophrenia?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["long-acting", "injectable", "LAI", "relapse", "paliperidone", "aripiprazole"],
            "source_types": ["pubmed"],
        },
    },

    # --- Women's health ---
    {
        "id": "Q29",
        "disease": "Endometriosis",
        "intent": "hormonal management",
        "location": "Sydney, Australia",
        "message": "What are the benefits and risks of GnRH antagonists like elagolix for endometriosis pain?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["elagolix", "GnRH", "endometriosis", "pain", "dysmenorrhea"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q30",
        "disease": "Polycystic ovary syndrome",
        "intent": "metformin vs lifestyle",
        "location": "Bangalore, India",
        "message": "Is metformin more effective than lifestyle intervention alone for PCOS management?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["metformin", "PCOS", "lifestyle", "insulin resistance", "ovulation"],
            "source_types": ["pubmed"],
        },
    },

    # --- More abstain tests ---
    {
        "id": "Q31",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "What is the best programming language to learn in 2026?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },
    {
        "id": "Q32",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "Tell me a joke about doctors.",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },
    {
        "id": "Q33",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "How tall is Mount Everest?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },

    # --- Ophthalmology ---
    {
        "id": "Q34",
        "disease": "Age-related macular degeneration",
        "intent": "anti-VEGF therapy",
        "location": "Zurich, Switzerland",
        "message": "How does faricimab compare to aflibercept for wet AMD treatment intervals?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["faricimab", "aflibercept", "anti-VEGF", "AMD", "macular", "injection"],
            "source_types": ["pubmed"],
        },
    },

    # --- Orthopedics ---
    {
        "id": "Q35",
        "disease": "Osteoarthritis",
        "intent": "knee replacement alternatives",
        "location": "Minneapolis, USA",
        "message": "What is the evidence for platelet-rich plasma injections versus hyaluronic acid for knee osteoarthritis?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["PRP", "platelet-rich plasma", "hyaluronic acid", "knee", "osteoarthritis"],
            "source_types": ["pubmed"],
        },
    },

    # --- Dermatology ---
    {
        "id": "Q36",
        "disease": "Psoriasis",
        "intent": "IL-17 inhibitors",
        "location": "Vienna, Austria",
        "message": "How do IL-17 inhibitors compare to IL-23 inhibitors for plaque psoriasis clearance?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["IL-17", "IL-23", "secukinumab", "ixekizumab", "guselkumab", "risankizumab", "PASI"],
            "source_types": ["pubmed"],
        },
    },

    # --- Gastroenterology ---
    {
        "id": "Q37",
        "disease": "Ulcerative colitis",
        "intent": "JAK inhibitors",
        "location": "Barcelona, Spain",
        "message": "What is the efficacy of tofacitinib for moderate-to-severe ulcerative colitis?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["tofacitinib", "JAK", "ulcerative colitis", "remission", "mucosal healing"],
            "source_types": ["pubmed"],
        },
    },

    # --- Endocrinology ---
    {
        "id": "Q38",
        "disease": "Hypothyroidism",
        "intent": "combination therapy",
        "location": "Copenhagen, Denmark",
        "message": "Is there benefit to adding T3 (liothyronine) to levothyroxine for persistent hypothyroid symptoms?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["T3", "liothyronine", "levothyroxine", "hypothyroid", "combination", "thyroid"],
            "source_types": ["pubmed"],
        },
    },

    # --- Nephrology ---
    {
        "id": "Q39",
        "disease": "Chronic kidney disease",
        "intent": "finerenone",
        "location": "Munich, Germany",
        "message": "What are the renal and cardiovascular outcomes with finerenone in diabetic kidney disease?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["finerenone", "kidney", "FIDELIO", "FIGARO", "MRA", "albuminuria", "eGFR"],
            "source_types": ["pubmed"],
        },
    },

    # --- Hematology ---
    {
        "id": "Q40",
        "disease": "Multiple myeloma",
        "intent": "bispecific antibodies",
        "location": "Rochester, USA",
        "message": "How effective are bispecific antibodies like teclistamab for relapsed multiple myeloma?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["teclistamab", "bispecific", "BCMA", "myeloma", "relapsed"],
            "source_types": ["pubmed"],
        },
    },

    # --- Vague / borderline queries (should still attempt, not abstain) ---
    {
        "id": "Q41",
        "disease": "Diabetes",
        "intent": "general",
        "location": "Lagos, Nigeria",
        "message": "What are the newest treatments for diabetes?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["diabetes", "insulin", "GLP", "SGLT2", "metformin"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q42",
        "disease": "Asthma",
        "intent": "biologic therapy",
        "location": "Singapore",
        "message": "Which biologics are recommended for severe eosinophilic asthma?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["eosinophilic", "mepolizumab", "benralizumab", "dupilumab", "IL-5", "asthma"],
            "source_types": ["pubmed"],
        },
    },

    # --- Neurology ---
    {
        "id": "Q43",
        "disease": "Amyotrophic lateral sclerosis",
        "intent": "disease-modifying therapy",
        "location": "Baltimore, USA",
        "message": "What is the evidence for tofersen in SOD1-ALS?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["tofersen", "SOD1", "ALS", "antisense", "neurofilament"],
            "source_types": ["pubmed"],
        },
    },

    # --- Cardiology ---
    {
        "id": "Q44",
        "disease": "Atrial fibrillation",
        "intent": "ablation vs medication",
        "location": "Milan, Italy",
        "message": "Is catheter ablation superior to antiarrhythmic drugs as first-line therapy for atrial fibrillation?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["ablation", "antiarrhythmic", "atrial fibrillation", "rhythm", "EAST-AFNET", "pulmonary vein"],
            "source_types": ["pubmed"],
        },
    },

    # --- Pulmonology ---
    {
        "id": "Q45",
        "disease": "Idiopathic pulmonary fibrosis",
        "intent": "antifibrotic therapy",
        "location": "Nashville, USA",
        "message": "How do nintedanib and pirfenidone compare for slowing FVC decline in IPF?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["nintedanib", "pirfenidone", "IPF", "FVC", "antifibrotic", "pulmonary fibrosis"],
            "source_types": ["pubmed"],
        },
    },

    # --- More abstain (tricky borderline) ---
    {
        "id": "Q46",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "What stocks should I invest in for healthcare companies?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },
    {
        "id": "Q47",
        "disease": "None",
        "intent": "general",
        "location": "",
        "message": "Can you help me write a medical school application essay?",
        "expect": {
            "should_abstain": True,
            "min_insights": 0,
            "min_trials": 0,
            "topic_keywords": [],
            "source_types": [],
        },
    },

    # --- Clinical trials heavy ---
    {
        "id": "Q48",
        "disease": "Glioblastoma",
        "intent": "immunotherapy trials",
        "location": "Boston, USA",
        "message": "Are there active immunotherapy clinical trials for recurrent glioblastoma?",
        "expect": {
            "should_abstain": False,
            "min_insights": 0,
            "min_trials": 1,
            "topic_keywords": ["glioblastoma", "immunotherapy", "checkpoint", "vaccine", "GBM"],
            "source_types": [],
        },
    },
    {
        "id": "Q49",
        "disease": "Cystic fibrosis",
        "intent": "CFTR modulators",
        "location": "Dublin, Ireland",
        "message": "What is the impact of elexacaftor-tezacaftor-ivacaftor on lung function in cystic fibrosis?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["elexacaftor", "Trikafta", "CFTR", "FEV1", "cystic fibrosis", "modulator"],
            "source_types": ["pubmed"],
        },
    },
    {
        "id": "Q50",
        "disease": "Prostate cancer",
        "intent": "PARP inhibitors",
        "location": "Los Angeles, USA",
        "message": "What is the role of olaparib in metastatic castration-resistant prostate cancer with BRCA mutations?",
        "expect": {
            "should_abstain": False,
            "min_insights": 1,
            "min_trials": 0,
            "topic_keywords": ["olaparib", "PARP", "BRCA", "prostate", "castration-resistant", "HRR"],
            "source_types": ["pubmed"],
        },
    },
]


# =============================================================================
# Scoring
# =============================================================================

def score_response(query, response):
    """Score a single pipeline response against expectations. Returns a dict of checks."""
    expect = query["expect"]
    checks = {}
    is_error = "error" in response or "detail" in response
    skip_retrieval = response.get("skip_retrieval", False)

    # 1. Abstention accuracy
    abstained = bool(response.get("abstain_reason")) or skip_retrieval or is_error
    if expect["should_abstain"]:
        checks["abstain_correct"] = abstained
    else:
        checks["abstain_correct"] = not abstained

    # If it's an error/abstain and shouldn't be, mark remaining checks as failed
    if is_error or (skip_retrieval and not expect["should_abstain"]):
        checks["has_overview"] = False
        checks["min_insights_met"] = False
        checks["min_trials_met"] = False
        checks["topic_hit"] = False
        checks["citations_grounded"] = False
        checks["has_structure"] = False
        return checks

    # If expected abstain and it did abstain, skip quality checks (they're N/A)
    if expect["should_abstain"] and abstained:
        return checks

    # 2. Has overview
    overview = response.get("overview", "")
    checks["has_overview"] = bool(overview) and len(overview) > 30

    # 3. Insight count
    insights = response.get("insights", [])
    checks["min_insights_met"] = len(insights) >= expect["min_insights"]

    # 4. Trial count
    trials = response.get("trials", [])
    checks["min_trials_met"] = len(trials) >= expect["min_trials"]

    # 5. Topic relevance — at least one keyword appears in overview or insight findings
    if expect["topic_keywords"]:
        text_blob = overview.lower()
        for ins in insights:
            text_blob += " " + ins.get("finding", "").lower()
        hit = any(kw.lower() in text_blob for kw in expect["topic_keywords"])
        checks["topic_hit"] = hit
    else:
        checks["topic_hit"] = True  # no keywords to check

    # 6. Citation grounding — every insight has at least one source_detail with a title
    if insights:
        grounded = all(
            any(sd.get("title") for sd in ins.get("source_details", []))
            for ins in insights
        )
        checks["citations_grounded"] = grounded
    else:
        checks["citations_grounded"] = expect["min_insights"] == 0

    # 7. Structural validity — response has expected top-level keys
    required_keys = {"overview", "insights", "trials", "pipelineMeta"}
    checks["has_structure"] = required_keys.issubset(response.keys())

    return checks


# =============================================================================
# Runner
# =============================================================================

def run_eval(base_url, queries, delay=2.0):
    """Run all queries against the pipeline, score each, return results."""
    results = []
    for i, q in enumerate(queries):
        qid = q["id"]
        payload = {
            "static": {
                "disease": q["disease"],
                "intent": q["intent"],
                "location": q["location"],
                "patientName": "Eval Harness",
            },
            "dynamic": {},
            "current": {"userMessage": q["message"]},
        }

        print(f"  [{i+1}/{len(queries)}] {qid}: {q['message'][:60]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{base_url}/pipeline/run", json=payload, timeout=120)
            elapsed = round((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                response = {"error": r.text[:200], "status_code": r.status_code}
            else:
                response = r.json()
        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000)
            response = {"error": str(exc)}

        checks = score_response(q, response)
        passed = all(checks.values())
        n_pass = sum(1 for v in checks.values() if v)
        n_total = len(checks)

        status = "PASS" if passed else "FAIL"
        print(f"{status} ({n_pass}/{n_total}) [{elapsed}ms]")

        # Extract pipeline meta for the report
        meta = response.get("pipelineMeta", {})

        results.append({
            "id": qid,
            "disease": q["disease"],
            "message": q["message"],
            "status": status,
            "checks": checks,
            "elapsed_ms": elapsed,
            "n_insights": len(response.get("insights", [])),
            "n_trials": len(response.get("trials", [])),
            "abstained": bool(response.get("abstain_reason") or response.get("skip_retrieval")),
            "stage_timings_ms": meta.get("stage_timings_ms", {}),
            "retrieval_counts": meta.get("retrieval_counts", {}),
            "citation_stats": meta.get("citation_stats", {}),
            "warnings": meta.get("warnings", []),
        })

        # Rate-limit courtesy: don't hammer HF free tier
        if i < len(queries) - 1:
            time.sleep(delay)

    return results


# =============================================================================
# Report
# =============================================================================

def print_report(results):
    """Print a summary report to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    print("\n" + "=" * 78)
    print("PIPELINE QUALITY EVALUATION REPORT")
    print("=" * 78)

    # Per-check aggregates
    all_checks = {}
    for r in results:
        for check, val in r["checks"].items():
            all_checks.setdefault(check, {"pass": 0, "total": 0})
            all_checks[check]["total"] += 1
            if val:
                all_checks[check]["pass"] += 1

    print(f"\n  Overall: {passed}/{total} queries passed all checks ({failed} failed)")
    print(f"\n  Per-check breakdown:")
    for check, counts in sorted(all_checks.items()):
        pct = counts["pass"] / counts["total"] * 100 if counts["total"] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {check:25s} {counts['pass']:2d}/{counts['total']:2d}  {bar}  {pct:.0f}%")

    # Latency summary
    medical = [r for r in results if not r["abstained"] and r["elapsed_ms"] > 0]
    if medical:
        times = sorted(r["elapsed_ms"] for r in medical)
        avg = sum(times) / len(times)
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        print(f"\n  Latency (medical queries only, {len(medical)} queries):")
        print(f"    avg {avg:.0f}ms   p50 {p50}ms   p95 {p95}ms   min {times[0]}ms   max {times[-1]}ms")

    # Per-stage timing averages
    stage_sums = {}
    stage_counts = {}
    for r in medical:
        for stage, ms in r["stage_timings_ms"].items():
            if isinstance(ms, (int, float)):
                stage_sums[stage] = stage_sums.get(stage, 0) + ms
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
    if stage_sums:
        print(f"\n  Average per-stage latency:")
        for stage in ["query_expansion", "retrieval", "normalization", "ranking",
                      "context_build", "llm", "assembly", "total"]:
            if stage in stage_sums:
                avg_ms = stage_sums[stage] / stage_counts[stage]
                print(f"    {stage:20s} {avg_ms:8.0f}ms")

    # Retrieval summary
    if medical:
        avg_insights = sum(r["n_insights"] for r in medical) / len(medical)
        avg_trials = sum(r["n_trials"] for r in medical) / len(medical)
        print(f"\n  Retrieval (medical queries):")
        print(f"    avg insights/query: {avg_insights:.1f}")
        print(f"    avg trials/query:   {avg_trials:.1f}")

    # Citation grounding
    total_cit = sum(r["citation_stats"].get("total", 0) for r in medical)
    verified_cit = sum(r["citation_stats"].get("verified", 0) for r in medical)
    unverified_cit = sum(r["citation_stats"].get("unverified", 0) for r in medical)
    if total_cit:
        ground_rate = verified_cit / total_cit * 100
        print(f"\n  Citations:")
        print(f"    total: {total_cit}   verified: {verified_cit}   unverified: {unverified_cit}")
        print(f"    grounding rate: {ground_rate:.1f}%")

    # Failed queries detail
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"\n  Failed queries:")
        for r in failures:
            failed_checks = [k for k, v in r["checks"].items() if not v]
            print(f"    {r['id']}: {r['message'][:50]}...")
            print(f"           failed: {', '.join(failed_checks)}")

    print("=" * 78)


def save_results(results, args):
    """Save detailed results as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(RESULTS_DIR, f"eval_{stamp}.json")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "n_queries": len(results),
        "n_passed": sum(1 for r in results if r["status"] == "PASS"),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Detailed results: {path}")


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Curalink pipeline quality evaluation")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="FastAPI base URL (default: localhost:8000)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="seconds between queries (rate-limit courtesy)")
    ap.add_argument("--query", type=int, default=None,
                    help="run a single query by index (0-based)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate eval set structure only, no server needed")
    args = ap.parse_args()

    if args.selftest:
        # Validate eval set
        ids = set()
        for i, q in enumerate(EVAL_SET):
            assert "id" in q, f"query {i} missing id"
            assert q["id"] not in ids, f"duplicate id {q['id']}"
            ids.add(q["id"])
            assert "disease" in q and "message" in q, f"{q['id']} missing fields"
            assert "expect" in q, f"{q['id']} missing expect"
            e = q["expect"]
            assert "should_abstain" in e, f"{q['id']} missing should_abstain"
            assert "min_insights" in e, f"{q['id']} missing min_insights"
        print(f"selftest ok — {len(EVAL_SET)} queries, {len(ids)} unique IDs")
        # Count by type
        medical = sum(1 for q in EVAL_SET if not q["expect"]["should_abstain"])
        abstain = len(EVAL_SET) - medical
        print(f"  {medical} medical queries, {abstain} should-abstain queries")
        return

    queries = EVAL_SET
    if args.query is not None:
        if args.query < 0 or args.query >= len(EVAL_SET):
            sys.exit(f"--query must be 0-{len(EVAL_SET)-1}")
        queries = [EVAL_SET[args.query]]

    # Health check
    try:
        r = httpx.get(f"{args.base_url}/health", timeout=10)
        if r.status_code != 200:
            sys.exit(f"Server health check failed: {r.status_code}")
    except Exception as exc:
        sys.exit(f"Cannot reach {args.base_url}: {exc}")

    print(f"Running {len(queries)} eval queries against {args.base_url}")
    print(f"Delay between queries: {args.delay}s\n")

    results = run_eval(args.base_url, queries, delay=args.delay)
    print_report(results)
    save_results(results, args)


if __name__ == "__main__":
    main()
