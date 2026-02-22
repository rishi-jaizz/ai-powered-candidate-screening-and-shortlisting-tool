"""
Run this script ONCE from your project folder to fix the import error.
It clears Python cache and verifies all files are correct.

Usage:
    cd C:\\Users\\hp\\OneDrive\\Desktop\\candidate_shortlister1\\candidate_shortlister
    python fix_cache.py
"""

import os
import shutil
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Project folder: {project_dir}")

# ── 1. Delete __pycache__ folders ────────────────────────────────────────────
deleted = 0
for root, dirs, files in os.walk(project_dir):
    for d in dirs:
        if d == "__pycache__":
            full = os.path.join(root, d)
            shutil.rmtree(full)
            print(f"  ✅ Deleted cache: {full}")
            deleted += 1

if deleted == 0:
    print("  (No __pycache__ folders found — already clean)")

# ── 2. Delete any .pyc files ──────────────────────────────────────────────────
for root, dirs, files in os.walk(project_dir):
    for f in files:
        if f.endswith(".pyc"):
            full = os.path.join(root, f)
            os.remove(full)
            print(f"  ✅ Deleted .pyc: {full}")

# ── 3. Verify embeddings.py has the correct functions ────────────────────────
emb_path = os.path.join(project_dir, "embeddings.py")
if not os.path.exists(emb_path):
    print("\n❌ ERROR: embeddings.py not found in project folder!")
    print("   Make sure you copied the new embeddings.py here.")
    sys.exit(1)

with open(emb_path, "r") as f:
    content = f.read()

checks = {
    "get_embeddings_pair": "def get_embeddings_pair" in content,
    "cosine_similarity":   "def cosine_similarity"   in content,
    "fresh vectorizer":    "TfidfVectorizer" in content and "def get_embeddings_pair" in content,
}

print("\n── embeddings.py function check ──")
all_ok = True
for name, ok in checks.items():
    status = "✅" if ok else "❌"
    print(f"  {status} {name}")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n❌ embeddings.py is still the OLD version.")
    print("   Please replace it with the new file from outputs/embeddings.py")
    sys.exit(1)

# ── 4. Quick import test ─────────────────────────────────────────────────────
print("\n── Import test ──")
try:
    sys.path.insert(0, project_dir)
    import importlib
    import embeddings as emb_module
    importlib.reload(emb_module)
    v1, v2 = emb_module.get_embeddings_pair("python machine learning", "python developer")
    score  = emb_module.cosine_similarity(v1, v2)
    print(f"  ✅ get_embeddings_pair: OK")
    print(f"  ✅ cosine_similarity:   {score:.4f}  (expect > 0)")
    print(f"\n🎉 All checks passed! You can now run: streamlit run app.py")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    print("   Ensure you replaced embeddings.py with the new version.")
    sys.exit(1)