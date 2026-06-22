# -*- coding: utf-8 -*-
"""
Download the three public datasets used in this project.

  UCI (5MB)          : public, committed to the repo already — skipped if present.
  Lending Club (~195MB): public LendingClub archive (no auth).
  Home Credit (~560MB) : Kaggle dataset mirror — requires ~/.kaggle/kaggle.json.

GitHub note: Lending Club & Home Credit files exceed GitHub's 100MB limit and
(for Home Credit) carry Kaggle terms, so they are NOT committed. Run this once.

Usage:  python download_data.py
"""
import os, glob, io, zipfile, urllib.request

def _get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/lc", exist_ok=True)
    os.makedirs("data/hc", exist_ok=True)

    # 1) UCI — Default of Credit Card Clients (Taiwan)
    if not os.path.exists("data/default of credit card clients.xls"):
        print("[UCI] downloading ...")
        url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
        zipfile.ZipFile(io.BytesIO(_get(url))).extractall("data")
    print("[UCI] ready")

    # 2) Lending Club — matured vintages 2007-2013 (Charged Off / Fully Paid)
    for f in ["LoanStats3a.csv", "LoanStats3b.csv"]:
        if not os.path.exists("data/lc/" + f):
            print("[LC] downloading", f, "...")
            zipfile.ZipFile(io.BytesIO(_get("https://resources.lendingclub.com/" + f + ".zip"))).extractall("data/lc")
    print("[LC] ready")

    # 3) Home Credit — credit_card_balance + application_train (Kaggle dataset mirror)
    need = [f for f in ["credit_card_balance.csv", "application_train.csv"] if not os.path.exists("data/hc/" + f)]
    if need:
        print("[HC] downloading via Kaggle API (needs ~/.kaggle/kaggle.json):", need)
        import kaggle
        kaggle.api.authenticate()
        for fn in need:
            kaggle.api.dataset_download_file("minhthuanha/homecreditpractice", fn, path="data/hc", quiet=False)
        for z in glob.glob("data/hc/*.zip"):
            zipfile.ZipFile(z).extractall("data/hc"); os.remove(z)
    print("[HC] ready")
    print("All data ready under ./data/")

if __name__ == "__main__":
    main()
