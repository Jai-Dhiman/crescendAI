from pathlib import Path
import yaml

PLAYBOOK = Path(__file__).parent / "playbook.yaml"

EXPECTED = {
    "Technical-corrective feedback",
    "Artifact-based teaching",
    "Positive-encouragement / praise",
    "Motivational / autonomy-supportive statements",
    "Guided-discovery / scaffolding feedback",
}


def _norm(name: str) -> str:
    return (name.replace("‑", "-").replace("–", "-").replace("—", "-")
            .replace("“", "").replace("”", "").strip())


def test_playbook_loads():
    data = yaml.safe_load(PLAYBOOK.read_text())
    assert "teaching_playbook" in data
    assert "clusters" in data["teaching_playbook"]


def test_five_clusters_present():
    data = yaml.safe_load(PLAYBOOK.read_text())
    names = {_norm(c["name"]) for c in data["teaching_playbook"]["clusters"]}
    expected = {_norm(n) for n in EXPECTED}
    assert names == expected


def test_each_cluster_has_triggers_score():
    data = yaml.safe_load(PLAYBOOK.read_text())
    for cluster in data["teaching_playbook"]["clusters"]:
        assert "triggers" in cluster
        assert "score" in cluster["triggers"]
        assert isinstance(cluster["triggers"]["score"], str)
        assert len(cluster["triggers"]["score"]) > 0
