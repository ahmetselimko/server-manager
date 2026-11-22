#!/bin/bash
# Azure Public Dataset V2 VM CPU izini (195 dosya, ~156 GB) diske yazmadan indirip abonelik bazında toplar.
# Kullanım (data_prep klasöründen):
#   curl -sSL -o vmtable.csv.gz https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz
#   python3 azure_aggregate.py index vmtable.csv.gz out && mkdir -p locks
#   seq 1 195 | xargs -P 5 -I{} ./download_and_aggregate.sh {}      # ~1 saat; 5 paralel ≈ 5 GB RAM
#   python3 azure_aggregate.py merge out && cp out/azure_sub_demand.npz ../data/
# Tek bir parçayı indir+işle (kilitli; yarım çıktı bırakmaz); başarısız olursa 2 kez daha dene
B=https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2
i=$1; f=out/part_$(printf %03d $i).npz
[ -f "$f" ] && exit 0
mkdir "locks/$i" 2>/dev/null || exit 0
for try in 1 2 3; do
  set -o pipefail
  if curl -sSL --fail --retry 3 "$B/trace_data_vm_cpu_readings_vm_cpu_readings-file-$i-of-195.csv.gz" \
     | python3 "$(dirname "$0")/azure_aggregate.py" part out $i; then exit 0; fi
  echo "part $i: deneme $try başarısız" >&2; rm -f "$f"; sleep 5
done
echo "part $i: BAŞARISIZ" >&2
