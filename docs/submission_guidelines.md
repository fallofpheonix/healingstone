# Submission Guidelines & Constraints: HealingStones

## What to Build
Develop an automated machine learning pipeline to reconstruct a fragmented 3D Mayan stele. The pipeline must:
1. Conduct data reduction and re-topologize if needed.
2. Classify surfaces and detect precise break lines.
3. Automatically align matched break surfaces accounting for data errors, gaps, and lost material.
4. Produce clear metrics and integrity plots on generated data.

## Evaluation Test (Required for Application)
Applicants must complete the **GSoC Healing Stones Test**.
- **The Dataset**: Using the 3D model data fragments from [**CERN Box 3D Space**](https://cernbox.cern.ch/s/hQO24HxuKi6VeQo). Develop code to reconstruct the fragmented Stele. The folder contains 12 rotated sections at the same pixel scale.
- **Fallback Data**: If needed, 2D fragments are also provided here: [**CERN Box 2D Space**](https://cernbox.cern.ch/s/kOdhPJxQrMzGdTN).

## Deliverables to Mentor
Send an email to **human-ai@cern.ch** with the subject **"GSoC Healing Stones"** containing:
- Your CV/Resume as an accessible link.
- A GitHub link containing the automated script covering model creation, training, and testing. Must run from start to finish without user intervention.
- A Google Drive link to any pre-trained models utilized.
- **Do NOT contact mentors directly.**

## Constraints
- **Language**: Python with previous ML experience.
- **Technical**: Must be able to manipulate `.PLY` and `.OBJ` formats programmatically.
- **Automation**: The evaluation pipeline must be completely automated to allow the selection committee to test it immediately on other `.PLY` / `.OBJ` fragments.
- **Difficulty**: Medium (175 hours).
