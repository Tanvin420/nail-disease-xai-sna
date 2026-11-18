"""Stage 04 - build near-duplicate components and curate the development and external test sets.

Links: same ko-pin (class, stem) group  |  identical bytes/pixels (md5)  |  pHash <= phash_link
       |  SSCD >= calibrated threshold  (manual audit labels in audit_pairs.csv override pair links).
Development components (containing ko-pin images) are excluded when they
  - contain ko-pin images of more than one class (label conflict),
  - are linked to Han A2 / have A2-derived stems (same hospital as the Han A1 external test),
  - match an 'exclude' rule of the label policy (e.g. beau_s linked to Kaggle "pitting").
External test images are kept only if their component contains no ko-pin image; one representative per
component and source is used.

Outputs data/dedup/{components.csv, flow.json, flow.md, component_sizes.csv},
        data/splits/{dev_components.csv, external_<source>.csv, legacy2272.csv}
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.dedup import chain_edges, components  # noqa: E402


def load_links(cfg, idx: pd.DataFrame, dedup_dir: Path) -> tuple[np.ndarray, dict]:
    thr = json.loads((dedup_dir / "threshold.json").read_text())
    t_sscd, t_ham = thr["sscd_threshold"], thr["phash_threshold"]
    pairs = pd.read_parquet(dedup_dir / "pairs.parquet")
    link = (pairs.sscd >= t_sscd) | (pairs.ham <= t_ham)
    audit_info = {"n_audited": 0}
    af = dedup_dir / "audit_pairs.csv"
    if af.exists():
        aud = pd.read_csv(af, dtype={"is_duplicate": str}, keep_default_na=False)
        aud = aud[aud.is_duplicate.isin(["0", "1"])]
        if len(aud):
            key = pairs.a + "|" + pairs.b
            lab = dict(zip(aud.a + "|" + aud.b, aud.is_duplicate == "1"))
            hit = key.isin(lab)
            link[hit] = key[hit].map(lab).astype(bool)
            audit_info = {"n_audited": int(len(aud)), "audit_precision_at_threshold":
                          float((aud[aud.sscd >= t_sscd].is_duplicate == "1").mean()) if (aud.sscd >= t_sscd).any() else None}
    pos = pd.Series(np.arange(len(idx)), index=idx.image_id)
    e_pairs = np.stack([pos[pairs.a[link]].to_numpy(), pos[pairs.b[link]].to_numpy()], 1)
    kop = idx[idx.source == "kopin"]
    e_stem = chain_edges([g.index.to_numpy() for _, g in kop.groupby(["label", "stem"])])
    dmd5 = idx[idx.md5.duplicated(keep=False)]
    e_md5 = chain_edges([g.index.to_numpy() for _, g in dmd5.groupby("md5")])
    info = {"sscd_threshold": t_sscd, "phash_threshold": t_ham, "n_pair_links": int(link.sum()),
            "n_stem_links": len(e_stem), "n_md5_links": len(e_md5), **audit_info}
    return np.concatenate([e_pairs, e_stem, e_md5]), info


def main():
    cfg = load_config()
    cur, seed = cfg.curation, cfg.seed
    rng = np.random.default_rng(seed)
    dedup_dir, split_dir = cfg.path("dedup"), cfg.path("splits")
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    edges, link_info = load_links(cfg, idx, dedup_dir)
    comp = components(len(idx), edges)
    idx["comp"] = [f"c{c:06d}" for c in comp]
    idx[["image_id", "comp"]].to_csv(dedup_dir / "components.csv", index=False)

    is_kop = idx.source == "kopin"
    dev_comps = set(idx.comp[is_kop])
    stem_re = [re.compile(r) for r in cur["dev_exclude_stem_regex"]]
    flow = {"links": link_info}

    # ---------------- development components
    rows = []
    for c, g in idx[idx.comp.isin(dev_comps)].groupby("comp", sort=True):
        k = g[g.source == "kopin"]
        other = g[g.source != "kopin"]
        labels = sorted(set(k.label))
        linked = sorted(set(other.source))
        linked_lab = set(zip(other.source, other.raw_label))
        status, reason, tags = "included", "", []
        if len(labels) > 1:
            status, reason = "excluded", "label_conflict"
        elif set(linked) & set(cur["dev_exclude_if_linked_to"]) or any(r.search(s) for r in stem_re for s in k.stem):
            status, reason = "excluded", "same_hospital_as_HanA1"
        else:
            for rule in cur["label_policy"]:
                if labels[0] == rule["dev_class"] and (rule["source"], rule["raw_label"]) in linked_lab:
                    if rule["action"] == "exclude" and status == "included":
                        status, reason = "excluded", f"label_policy:{rule['source']}:{rule['raw_label']}"
                    elif rule["action"] == "flag":
                        tags.append(rule["tag"])
        pref = k[k.vendor_split.isin(["test", "valid"])]
        rep = rng.choice(sorted((pref if len(pref) else k).image_id))
        rows.append({"comp": c, "label": labels[0] if len(labels) == 1 else "|".join(labels), "status": status,
                     "reason": reason, "tags": ";".join(tags), "n_images": len(k), "n_stems": k.stem.nunique(),
                     "vendor_splits": ";".join(sorted(set(k.vendor_split))), "linked_sources": ";".join(linked),
                     "rep_image_id": rep})
    dev = pd.DataFrame(rows)
    dev.to_csv(split_dir / "dev_components.csv", index=False)
    inc = dev[dev.status == "included"]
    kop = idx[is_kop]
    flow["dev"] = {
        "kopin_files": int(len(kop)),
        "kopin_class_stem_groups": int(kop.groupby(["label", "stem"]).ngroups),
        "dev_components": int(len(dev)),
        "components_spanning_vendor_splits": int((dev.vendor_splits.str.count(";") > 0).sum()),
        "components_in_all_three_vendor_splits": int((dev.vendor_splits.str.count(";") == 2).sum()),
        "excluded_by_reason": dev[dev.status == "excluded"].reason.value_counts().to_dict(),
        "included_components": int(len(inc)),
        "included_components_by_class": inc.label.value_counts().sort_index().to_dict(),
        "included_files_by_class": inc.groupby("label").n_images.sum().sort_index().to_dict(),
        "flagged": dev.tags[dev.tags != ""].value_counts().to_dict(),
        "largest_components": dev.nlargest(10, "n_images")[["comp", "label", "n_images", "n_stems"]].to_dict("records"),
    }
    big = dev.n_images.max() / len(kop)
    if big > 0.01:
        print(f"WARNING: largest dev component holds {big:.1%} of ko-pin files - inspect chaining")
    dev.n_images.value_counts().sort_index().rename("n_components").to_csv(dedup_dir / "component_sizes.csv")

    # ---------------- external test sets
    flow["external"] = {}
    for src in cfg.sources:
        if src["role"] not in ("external", "legacy", "provenance"):
            continue
        sid = src["id"]
        g = idx[(idx.source == sid) & ((idx.label != "") | (idx.is_probe == 1))].copy()
        g["linked_dev"] = g.comp.isin(dev_comps).astype(int)
        if src["role"] != "legacy":
            # every mapped image of the source, duplicates kept and flagged: "run the models on every dataset" table
            g[g.label != ""][["image_id", "label", "comp", "linked_dev"]].to_csv(split_dir / f"full_{sid}.csv", index=False)
        if src["role"] == "provenance":
            continue
        if src["role"] == "legacy":
            # origin of each image of the submitted external set, via near-duplicate links to the public sources:
            # Kaggle nikhilgurav21, or Roboflow mohit-fc0fl (= Kaggle 17-class + josephrasanjana re-packaged)
            comp_src = idx.groupby("comp").source.agg(set)
            def origin(c):
                s = comp_src.get(c, set())
                if "nikhilgurav21" in s:
                    return "Kaggle nikhilgurav21"
                if s & {"kaggle17", "josephrasanjana"}:
                    return "Roboflow mohit-fc0fl"
                return "unmatched"
            g["origin"] = g.comp.map(origin)
            g[["image_id", "label", "raw_label", "comp", "linked_dev", "origin"]].to_csv(split_dir / f"{sid}.csv", index=False)
            flow["external"][sid] = {"images": int(len(g)), "linked_to_dev": int(g.linked_dev.sum()),
                                     "linked_by_class": g.groupby("label").linked_dev.agg(["sum", "count"]).to_dict("index"),
                                     "by_origin": g.groupby("origin").linked_dev.agg(["sum", "count"]).to_dict("index")}
            continue
        free = g[g.linked_dev == 0]
        keyed = free.assign(key=free.label.where(free.label != "", "probe:" + free.raw_label))
        nlab = keyed.groupby("comp").key.nunique()
        conflict = set(nlab[nlab > 1].index)
        keep = keyed[~keyed.comp.isin(conflict)]
        reps = keep.groupby("comp", sort=True).image_id.apply(lambda s: rng.choice(sorted(s)))
        sel = keep[keep.image_id.isin(set(reps))].copy()
        sel["n_in_comp"] = sel.comp.map(keep.comp.value_counts())
        sel[["image_id", "source", "label", "raw_label", "is_probe", "comp", "n_in_comp"]].to_csv(
            split_dir / f"external_{sid}.csv", index=False)
        flow["external"][sid] = {
            "images": int(len(g)), "linked_to_dev": int(g.linked_dev.sum()),
            "linked_to_dev_by_key": g.assign(key=g.label.where(g.label != "", "probe:" + g.raw_label))
                                     .groupby("key").linked_dev.sum().to_dict(),
            "within_source_conflict_components": len(conflict),
            "test_images_mapped": sel[sel.label != ""].label.value_counts().sort_index().to_dict(),
            "test_images_probe": sel[sel.label == ""].raw_label.value_counts().sort_index().to_dict(),
        }
    (dedup_dir / "flow.json").write_text(json.dumps(flow, indent=2, default=int))

    # human-readable flow table
    L = ["# Data flow (generated by s04_components.py)", "", f"Links: {link_info}", "", "## Development set (ko-pin)"]
    for k, v in flow["dev"].items():
        if k != "largest_components":
            L.append(f"- **{k}**: {v}")
    L += ["", "## External test sets", "", "| source | images | linked to dev (removed) | test images (mapped) | probes |",
          "|---|---|---|---|---|"]
    for sid, v in flow["external"].items():
        L.append(f"| {sid} | {v['images']} | {v['linked_to_dev']} | {v.get('test_images_mapped', '')} | "
                 f"{sum(v.get('test_images_probe', {}).values()) if 'test_images_probe' in v else ''} |")
    (dedup_dir / "flow.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    write_meta(dedup_dir, cfg, stage="s04_components", n_dev_included=len(inc))


if __name__ == "__main__":
    main()
