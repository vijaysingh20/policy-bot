from pathlib import Path
from backend.schemas.evaluation import EvaluationRecord

def append_evaluation_record(
    record: EvaluationRecord,
    log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_record = record.model_dump_json()
    with log_path.open("a", encoding="utf-8") as file:
        file.write(serialized_record + "\n")