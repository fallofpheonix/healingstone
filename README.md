# Healing Stones: Reconstructing Digitized Cultural Heritage Artifacts with AI

## GSoC 2025 | HumanAI

### Purpose
The Healing Stones project aims to use machine learning to reconstruct fragmented objects using 3D digital model data. Over history, works of art and architecture have been subjected to fragmentation due to various factors like collecting, political iconoclasm, or natural decay. This project proposes to use AI in combination with existing digital scan models of fragments to develop a means for reconstructing these artifacts in a virtual space.

---

## Project Overview
Art historians and archaeologists seek to reconstruct fragmented works to more fully understand their cultural meaning and value. Traditional methods of physical refitting are labor-intensive and often impossible when fragments are dispersed globally. This project leverages AI to automate and assist in the reconstruction of these artifacts using high-quality digital scans.

## Task: 3D Mayan Stele Reconstruction
The primary task is to develop code to reconstruct a fragmented 3D Mayan stele from 12 randomly rotated sections.

### Dataset
- **3D Model Data:** [CERNBox 3D Fragments](https://cernbox.cern.ch/s/hQO24HxuKi6VeQo)
- **2D Fragments (Optional):** [CERNBox 2D Fragments](https://cernbox.cern.ch/s/kOdhPJxQrMzGdTN)

The dataset includes 12 PLY/OBJ files representing sections of the stele. These fit together to form one object, with some gaps where material has been lost.

### Project Structure
```text
.
├── DataSet/            # Raw 3D and 2D fragments
│   ├── 2D/
│   └── 3D/
├── src/                # Source code for reconstruction
├── models/             # Trained models and checkpoints
├── docs/               # Documentation and reports
├── tests/              # Verification and testing scripts
├── README.md           # Project overview
├── CONTRIBUTING.md     # Guidelines for contributors
└── LICENSE             # MIT License
```

## Deliverables
- **Reconstruction Script:** A comprehensive script (or set of scripts) in the GitHub repository that includes the entire process: data processing, model creation, training, and testing.
- **Automation:** The script must be runnable from start to finish without user intervention.
- **Metrics/Plots:** Clear visualization of the reconstruction progress and final integrity metrics.
- **Generalization:** The data augmentation and processing should be automated to allow testing on other `.PLY` or `.OBJ` files.
- **Pre-trained Models:** Links to pre-trained models (if used) should be provided via Google Drive.
- **CV/Resume:** An accessible link to the contributor's CV/Resume.

## Metrics
- **Usability:** Efficiency and clarity of the scripts produced.
- **Integrity:** The structural and visual accuracy of the generated data/reconstruction.

## Installation & Usage
*(Coming soon - to be updated as the implementation progresses)*

```bash
# Example setup
pip install -r requirements.txt
python src/reconstruct.py --data DataSet/3D/
```

## Contact
Please email test results and CV to **human-ai@cern.ch** with the subject line **“GSoC Healing Stones”**.

---
*HumanAI - Reconstructing the past with the intelligence of the future.*
