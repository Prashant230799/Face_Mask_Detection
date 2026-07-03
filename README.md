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
