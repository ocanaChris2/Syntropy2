#!/usr/bin/env python3
"""Fetch H.264 conformance bitstreams (ITU-T H.264.1, public at itu.int) for parser testing.

    python3 bench/conformance/fetch.py [names...]

Each archive holds a bitstream and its decoded pictures; only the bitstream is kept, under
bench/conformance/h264/. The streams are test material only: never part of nsc or its tests'
fixtures. `python3 bench/conformance/run.py` wraps them in MP4 (ffmpeg, stream copy) and runs
`nsc h264-test` on each.
"""
import io
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance/"
# CABAC streams, frame coding (what the recoder handles), and a few it must refuse cleanly
SETS = {
    "AVCv1": ["CABA1_SVA_B", "CABA2_SVA_B", "CABA3_SVA_B", "CABA1_Sony_D", "CABA2_Sony_E", "CABA3_Sony_C", "CABA3_TOSHIBA_E", "CANL1_SVA_B",
              "CANL2_SVA_B", "CANL3_SVA_B", "CANL4_SVA_B", "CANL1_TOSHIBA_G", "CABACI3_Sony_B", "CAQP1_Sony_B", "CACQP3_Sony_D", "CAPCM1_Sand_E",
              "CAPCMNL1_Sand_E", "CABREF3_Sand_D", "CAWP1_TOSHIBA_E", "CAWP5_TOSHIBA_E", "CAPM3_Sony_D", "CABAST3_Sony_E", "cabac_mot_frm0_full",
              "cabac_mot_mbaff0_full", "CAMA1_Sony_C", "CAFI1_SVA_C", "cabac_mot_fld0_full", "cabac_mot_picaff0_full", "CANL1_Sony_E",
              "CANL2_Sony_E", "CANL3_Sony_C", "CABASTBR3_Sony_B", "CAPAMA3_Sand_F",
              # CAVLC (Baseline and Main): frame coding, and some with slice groups or fields (refused)
              "BA1_Sony_D", "BA2_Sony_F", "BA3_SVA_C", "BA_MW_D", "BANM_MW_D", "BA1_FT_C", "NL1_Sony_D", "NL2_Sony_H", "NL3_SVA_E", "SVA_BA1_B",
              "SVA_BA2_D", "SVA_Base_B", "SVA_CL1_E", "SVA_FM1_E", "SVA_NL1_B", "SVA_NL2_E", "MIDR_MW_D", "NRF_MW_E", "MPS_MW_A", "BAMQ1_JVC_C",
              "BAMQ2_JVC_C", "LS_SVA_D", "CI_MW_D", "CVPCMNL1_SVA_C", "CVPCMNL2_SVA_C", "CVSE2_Sony_B", "CVWP1_TOSHIBA_E", "CVWP2_TOSHIBA_E",
              "CVBS3_Sony_C", "MR1_BT_A", "MR1_MW_A", "MR2_MW_A", "HCBP2_HHI_A", "HCMP1_HHI_A", "BASQP1_Sony_C", "CI1_FT_B", "FM1_BT_B", "CVSE3_Sony_H",
              "CVFC1_Sony_C", "AUD_MW_E", "SL1_SVA_B", "NLMQ1_JVC_C", "NLMQ2_JVC_C", "cavlc_mot_frm0_full_B", "CVMANL1_TOSHIBA_B"],
    "FRExt": ["FRExt1_Panasonic_D", "FRExt2_Panasonic_C", "FRExt3_Panasonic_E", "FRExt4_Panasonic_B", "HCAFR1_HHI_C", "HCAFR2_HHI_A", "HCAFR3_HHI_A",
              "HCAFR4_HHI_A", "HCAFF1_HHI_B", "HCAMFF1_HHI_B", "FREXT01_JVC_D", "FREXT02_JVC_C", "FREXT1_TANDBERG_A", "FREXT2_TANDBERG_A",
              "FREXT3_TANDBERG_A", "HPCAMOLQ_BRCM_B", "HPCAQ2LQ_BRCM_B", "HPCA_BRCM_C", "HPCALQ_BRCM_B", "HPCADQ_BRCM_B", "HPCANL_BRCM_C",
              "freh12_b", "freh1_b", "freh2_b", "freh7_b", "brcm_freh3", "brcm_freh4", "brcm_freh5", "brcm_freh8", "brcm_freh9", "brcm_freh10",
              "brcm_freh11", "alphaconformanceG", "HPCV_BRCM_A", "HPCVNL_BRCM_A", "HPCVMOLQ_BRCM_B"],
}
SKIP = (".yuv", ".txt", ".doc", ".pdf", ".md5", ".qcif", ".cif", ".rtf", ".docx", ".log", ".cfg", ".dat", ".h", ".xls")


def main():
    out = os.path.join(HERE, "h264")
    os.makedirs(out, exist_ok=True)
    want = set(sys.argv[1:])
    for d, names in SETS.items():
        for n in names:
            if want and n not in want:
                continue
            if any(f.startswith(n + ".") or f.startswith(n + "__") for f in os.listdir(out)):
                continue
            try:
                data = urllib.request.urlopen(BASE + d + "/" + n + ".zip", timeout=120).read()
                z = zipfile.ZipFile(io.BytesIO(data))
            except Exception as e:  # noqa: BLE001
                print(f"{n}: {e}", flush=True)
                continue
            kept = []
            for m in z.infolist():
                low = m.filename.lower()
                if m.is_dir() or low.endswith(SKIP) or "yuv" in low.rsplit(".", 1)[-1]:
                    continue
                body = z.read(m)
                # an Annex B stream starts with a start code
                if not (body[:3] == b"\0\0\1" or body[:4] == b"\0\0\0\1"):
                    continue
                base = os.path.basename(m.filename)
                name = n + "__" + base if not base.lower().startswith(n.lower()) else base
                with open(os.path.join(out, name), "wb") as f:
                    f.write(body)
                kept.append(f"{name} ({len(body)} B)")
            print(f"{n}: {', '.join(kept) or 'no bitstream found: ' + ', '.join(m.filename for m in z.infolist())}", flush=True)


if __name__ == "__main__":
    main()
