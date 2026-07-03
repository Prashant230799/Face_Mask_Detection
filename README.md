# FaceMaskDetection

Simple face mask detection project (training and realtime inference).

How to upload to GitHub

1. Initialize & commit (already done locally):

   git init
   git checkout -b main
   git add .
   git commit -m "Initial commit"

2. Create a GitHub repository and push:

   # Create remote using GitHub CLI (recommended):
   gh repo create <repo-name> --public --source=. --remote=origin --push

   # Or add remote manually and push:
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main

Notes:
- Consider not pushing large model files; use Git LFS or store models elsewhere.
- Adjust `.gitignore` if you want to include the `models/` directory.

Downloading the model
---------------------

If you do not want to include model binaries in the repository, host the trained model file somewhere (S3, Google Drive, GitHub Releases, Hugging Face, etc.) and let users download it at runtime. This repo includes a small helper script `download_model.py` that downloads a model file and optionally verifies its SHA256 checksum.

Example:

```bash
python download_model.py --url https://example.com/path/to/facemask_detector.keras \
   --dest models/facemask_detector.keras \
   --sha256 <expected-sha256-hex>
```

Notes:
- If the model is large, consider using Git LFS or a hosting provider and linking to it in the README.
- For automated CI or deployments, you can call `download_model.py` as part of your setup steps.

