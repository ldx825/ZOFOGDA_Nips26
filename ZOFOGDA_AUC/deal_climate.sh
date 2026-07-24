#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
TMP_DIR="${ROOT_DIR}/_tmp_datasets"

mkdir -p "${DATA_DIR}" "${TMP_DIR}"

echo "[INFO] ROOT_DIR = ${ROOT_DIR}"
echo "[INFO] DATA_DIR = ${DATA_DIR}"
echo "[INFO] TMP_DIR  = ${TMP_DIR}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1"; exit 1; }; }
need_cmd wget
need_cmd unzip
need_cmd bzip2
need_cmd python

WGET_OPTS=(
  "--timeout=20"
  "--read-timeout=20"
  "--tries=20"
  "--waitretry=5"
  "--retry-connrefused"
  "--continue"
  "--progress=bar:force"
  "--max-redirect=20"
  "--content-disposition"
)

FAILED=()

fail() {
  local name="$1"
  echo "[FAIL] ${name}"
  FAILED+=("${name}")
  return 0
}

# -----------------------------
# download helper (skip if exists)
# -----------------------------
dl_if_missing() {
  # $1=url $2=out_path
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo "[SKIP] Exists: ${out}"
    return 0
  fi
  echo "[DL] ${url}"
  if ! wget "${WGET_OPTS[@]}" -O "${out}" "${url}"; then
    echo "[WARN] Download failed: ${url}"
    rm -f "${out}"
    return 1
  fi
}

# -----------------------------
# LIBSVM split helper
# -----------------------------
split_libsvm_if_needed() {
  # $1=train_path  $2=test_path  $3=seed  $4=test_size
  local tr="$1"
  local te="$2"
  local seed="${3:-42}"
  local ratio="${4:-0.2}"

  if [[ -s "${te}" ]]; then
    echo "[SKIP] Test exists: ${te}"
    return 0
  fi
  if [[ ! -s "${tr}" ]]; then
    echo "[WARN] Train file missing, cannot split: ${tr}"
    return 1
  fi

  echo "[SPLIT] Generate ${te} from ${tr} (seed=${seed}, test_size=${ratio})"
  TRAIN_PATH="${tr}" TEST_PATH="${te}" SEED="${seed}" RATIO="${ratio}" python - <<'PY'
import os, numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split

tr = os.environ["TRAIN_PATH"]
te = os.environ["TEST_PATH"]
seed = int(os.environ.get("SEED","42"))
ratio = float(os.environ.get("RATIO","0.2"))

X, y = load_svmlight_file(tr)
y = np.asarray(y)
idx = np.arange(X.shape[0])

strat = y if len(np.unique(y)) > 1 else None
tr_idx, te_idx = train_test_split(idx, test_size=ratio, random_state=seed, stratify=strat)

dump_svmlight_file(X[tr_idx], y[tr_idx], tr, zero_based=True)
dump_svmlight_file(X[te_idx], y[te_idx], te, zero_based=True)
print("[OK] Split:", tr, te, "sizes:", len(tr_idx), len(te_idx))
PY
}

# -----------------------------
# check python deps
# -----------------------------
check_py_deps() {
  python - <<'PY' >/dev/null 2>&1
import pandas, sklearn, scipy
PY
}

echo
echo "=============================="
echo "[1] LIBSVM (binary page) datasets"
echo "=============================="

dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"    "${DATA_DIR}/a9a"    || fail "a9a"
dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t" "${DATA_DIR}/a9a.t"  || fail "a9a.t"

# svmguide1 (bz2)
if [[ ! -s "${DATA_DIR}/svmguide1" ]]; then
  dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1.bz2" "${TMP_DIR}/svmguide1.bz2" \
    && bunzip2 -c "${TMP_DIR}/svmguide1.bz2" > "${DATA_DIR}/svmguide1" \
    && echo "[UNBZ2] -> ${DATA_DIR}/svmguide1" \
    || fail "svmguide1"
else
  echo "[SKIP] Exists: ${DATA_DIR}/svmguide1"
fi

if [[ ! -s "${DATA_DIR}/svmguide1.t" ]]; then
  dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1.t.bz2" "${TMP_DIR}/svmguide1.t.bz2" \
    && bunzip2 -c "${TMP_DIR}/svmguide1.t.bz2" > "${DATA_DIR}/svmguide1.t" \
    && echo "[UNBZ2] -> ${DATA_DIR}/svmguide1.t" \
    || fail "svmguide1.t"
else
  echo "[SKIP] Exists: ${DATA_DIR}/svmguide1.t"
fi

# covtype (bz2, then split)
if [[ ! -s "${DATA_DIR}/covtype" ]]; then
  dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2" "${TMP_DIR}/covtype.bz2" \
    && bunzip2 -c "${TMP_DIR}/covtype.bz2" > "${DATA_DIR}/covtype" \
    && echo "[UNBZ2] -> ${DATA_DIR}/covtype" \
    || fail "covtype"
else
  echo "[SKIP] Exists: ${DATA_DIR}/covtype"
fi
split_libsvm_if_needed "${DATA_DIR}/covtype" "${DATA_DIR}/covtype.t" 42 0.2 || fail "covtype.t"

# german (plain, no official .t -> split)
dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer" "${DATA_DIR}/german" || fail "german"
split_libsvm_if_needed "${DATA_DIR}/german" "${DATA_DIR}/german.t" 42 0.2 || fail "german.t"

# ijcnn1 (bz2)
if [[ ! -s "${DATA_DIR}/ijcnn1" ]]; then
  dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2" "${TMP_DIR}/ijcnn1.bz2" \
    && bunzip2 -c "${TMP_DIR}/ijcnn1.bz2" > "${DATA_DIR}/ijcnn1" \
    && echo "[UNBZ2] -> ${DATA_DIR}/ijcnn1" \
    || fail "ijcnn1"
else
  echo "[SKIP] Exists: ${DATA_DIR}/ijcnn1"
fi

if [[ ! -s "${DATA_DIR}/ijcnn1.t" ]]; then
  dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2" "${TMP_DIR}/ijcnn1.t.bz2" \
    && bunzip2 -c "${TMP_DIR}/ijcnn1.t.bz2" > "${DATA_DIR}/ijcnn1.t" \
    && echo "[UNBZ2] -> ${DATA_DIR}/ijcnn1.t" \
    || fail "ijcnn1.t"
else
  echo "[SKIP] Exists: ${DATA_DIR}/ijcnn1.t"
fi

echo
echo "=============================="
echo "[2] UCI climate -> LIBSVM (drop non-numeric like 'Study') + split"
echo "=============================="

if [[ -s "${DATA_DIR}/climate" && -s "${DATA_DIR}/climate.t" ]]; then
  echo "[SKIP] climate already prepared."
else
  if ! check_py_deps; then
    echo "[WARN] Python deps missing (need pandas, scikit-learn, scipy)."
    echo "       Install: pip install -U pandas scikit-learn scipy"
    fail "climate(py_deps)"
  else
    dl_if_missing "https://archive.ics.uci.edu/static/public/252/climate%2Bmodel%2Bsimulation%2Bcrashes.zip" "${TMP_DIR}/climate.zip" \
      || { fail "climate.zip"; goto_climate=0; }
    goto_climate=1

    if [[ "${goto_climate}" == "1" ]]; then
      rm -rf "${TMP_DIR}/climate_unz"
      mkdir -p "${TMP_DIR}/climate_unz"
      unzip -o "${TMP_DIR}/climate.zip" -d "${TMP_DIR}/climate_unz" >/dev/null || { fail "climate.unzip"; goto_climate=0; }
    fi

    if [[ "${goto_climate}" == "1" ]]; then
      CLIMATE_RAW="$(find "${TMP_DIR}/climate_unz" -maxdepth 3 -type f \( -name "*.dat" -o -name "*.data" -o -name "*.csv" -o -name "*.txt" \) | head -n 1 || true)"
      if [[ -z "${CLIMATE_RAW}" ]]; then
        echo "[ERROR] Cannot find climate raw file in zip."
        fail "climate.raw"
      else
        echo "[INFO] Found climate file: ${CLIMATE_RAW}"
        CLIMATE_RAW="${CLIMATE_RAW}" OUT_TR="${DATA_DIR}/climate" OUT_TE="${DATA_DIR}/climate.t" SEED=42 python - <<'PY' \
          || fail "climate.convert"
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import dump_svmlight_file

raw = os.environ["CLIMATE_RAW"]
out_tr = os.environ["OUT_TR"]
out_te = os.environ["OUT_TE"]
seed = int(os.environ.get("SEED","42"))

df = pd.read_csv(raw, sep=r"\s+", engine="python")

y = df.iloc[:, -1].astype(str)
X = df.iloc[:, :-1]
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

y_enc = LabelEncoder().fit_transform(y)

strat = y_enc if len(np.unique(y_enc)) > 1 else None
Xtr, Xte, ytr, yte = train_test_split(
    X.values, y_enc, test_size=0.2, random_state=seed, stratify=strat
)
dump_svmlight_file(Xtr, ytr, out_tr, zero_based=True)
dump_svmlight_file(Xte, yte, out_te, zero_based=True)

print("[OK] climate saved:", out_tr, out_te, "shapes:", Xtr.shape, Xte.shape, "classes:", np.unique(y_enc).size)
PY
      fi
    fi
  fi
fi

echo
echo "=============================="
echo "[3] UCI spambase -> LIBSVM + split"
echo "=============================="

if [[ -s "${DATA_DIR}/spambase" && -s "${DATA_DIR}/spambase.t" ]]; then
  echo "[SKIP] spambase already prepared."
else
  if ! check_py_deps; then
    echo "[WARN] Python deps missing (need pandas, scikit-learn, scipy)."
    echo "       Install: pip install -U pandas scikit-learn scipy"
    fail "spambase(py_deps)"
  else
    dl_if_missing "https://archive.ics.uci.edu/static/public/94/spambase.zip" "${TMP_DIR}/spambase.zip" \
      || { fail "spambase.zip"; goto_spam=0; }
    goto_spam=1

    if [[ "${goto_spam}" == "1" ]]; then
      rm -rf "${TMP_DIR}/spambase_unz"
      mkdir -p "${TMP_DIR}/spambase_unz"
      unzip -o "${TMP_DIR}/spambase.zip" -d "${TMP_DIR}/spambase_unz" >/dev/null || { fail "spambase.unzip"; goto_spam=0; }
    fi

    if [[ "${goto_spam}" == "1" ]]; then
      SPAM_RAW="${TMP_DIR}/spambase_unz/spambase.data"
      if [[ ! -f "${SPAM_RAW}" ]]; then
        SPAM_RAW="$(find "${TMP_DIR}/spambase_unz" -maxdepth 3 -type f -name "spambase.data" | head -n 1 || true)"
      fi

      if [[ -z "${SPAM_RAW}" || ! -f "${SPAM_RAW}" ]]; then
        echo "[ERROR] spambase.data not found after unzip."
        fail "spambase.raw"
      else
        SPAM_RAW="${SPAM_RAW}" OUT_TR="${DATA_DIR}/spambase" OUT_TE="${DATA_DIR}/spambase.t" SEED=42 python - <<'PY' \
          || fail "spambase.convert"
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file

raw = os.environ["SPAM_RAW"]
out_tr = os.environ["OUT_TR"]
out_te = os.environ["OUT_TE"]
seed = int(os.environ.get("SEED","42"))

df = pd.read_csv(raw, header=None)
X = df.iloc[:, :-1].astype(float).values
y = df.iloc[:, -1].astype(int).values

strat = y if len(np.unique(y)) > 1 else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)

dump_svmlight_file(Xtr, ytr, out_tr, zero_based=True)
dump_svmlight_file(Xte, yte, out_te, zero_based=True)

print("[OK] spambase saved:", out_tr, out_te, "shapes:", Xtr.shape, Xte.shape, "classes:", np.unique(y).size)
PY
      fi
    fi
  fi
fi

echo
echo "=============================="
echo "[4] LIBSVM multiclass datasets (connect, acoustic) + ensure .t"
echo "=============================="

# connect-4 -> split
dl_if_missing "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/connect-4" "${DATA_DIR}/connect" \
  || fail "connect"
split_libsvm_if_needed "${DATA_DIR}/connect" "${DATA_DIR}/connect.t" 42 0.2 || fail "connect.t"

# ------------------------------------------------------------
# acoustic (SensIT Vehicle acoustic)  ✅ FIXED:
# correct files are under .../multiclass/vehicle/ and are .bz2
# official page lists acoustic/acoustic.t/acoustic_scale/acoustic_scale.t (bz2 links) :contentReference[oaicite:1]{index=1}
# ------------------------------------------------------------
ACOUSTIC_BASE="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle"

# Prefer scaled version (acoustic_scale) as you originally wanted
if [[ ! -s "${DATA_DIR}/acoustic" ]]; then
  dl_if_missing "${ACOUSTIC_BASE}/acoustic_scale.bz2" "${TMP_DIR}/acoustic_scale.bz2" \
    && bunzip2 -c "${TMP_DIR}/acoustic_scale.bz2" > "${DATA_DIR}/acoustic" \
    && echo "[UNBZ2] -> ${DATA_DIR}/acoustic" \
    || fail "acoustic"
else
  echo "[SKIP] Exists: ${DATA_DIR}/acoustic"
fi

if [[ ! -s "${DATA_DIR}/acoustic.t" ]]; then
  dl_if_missing "${ACOUSTIC_BASE}/acoustic_scale.t.bz2" "${TMP_DIR}/acoustic_scale.t.bz2" \
    && bunzip2 -c "${TMP_DIR}/acoustic_scale.t.bz2" > "${DATA_DIR}/acoustic.t" \
    && echo "[UNBZ2] -> ${DATA_DIR}/acoustic.t" \
    || fail "acoustic.t"
else
  echo "[SKIP] Exists: ${DATA_DIR}/acoustic.t"
fi

echo
echo "=============================="
echo "[DONE] Prepared datasets in: ${DATA_DIR}"
echo "=============================="
ls -lh "${DATA_DIR}" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "=============================="
  echo "[SUMMARY] Some steps failed:"
  printf ' - %s\n' "${FAILED[@]}"
  echo "=============================="
  exit 2
else
  echo
  echo "[SUMMARY] All done successfully."
fi
