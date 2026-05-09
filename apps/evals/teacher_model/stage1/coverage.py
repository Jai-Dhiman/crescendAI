from collections import defaultdict
from dataclasses import dataclass

from teacher_model.stage1.schema import Stage1Example, Stage1ToolUseBlock


@dataclass(frozen=True)
class Cell:
    tool: str
    cell: str
    have: int
    want: int


class CoverageMatrix:
    def __init__(self, targets: dict[str, dict[str, int]]) -> None:
        self._targets = targets
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def record(self, example: Stage1Example) -> None:
        for block in example.assistant.content:
            if not isinstance(block, Stage1ToolUseBlock):
                continue
            tool = block.name
            tool_targets = self._targets.get(tool, {})
            for cell_key in tool_targets:
                if self._cell_present(cell_key, block.input):
                    self._counts[(tool, cell_key)] += 1

    def unfilled_cells(self) -> list[Cell]:
        out: list[Cell] = []
        for tool, cells in self._targets.items():
            for cell_key, want in cells.items():
                have = self._counts[(tool, cell_key)]
                if have < want:
                    out.append(Cell(tool=tool, cell=cell_key, have=have, want=want))
        return out

    def is_satisfied(self) -> bool:
        return not self.unfilled_cells()

    @staticmethod
    def _cell_present(cell_key: str, payload: dict) -> bool:
        field, _, expected = cell_key.partition(":")
        return _scan_for_field_value(payload, field, expected)


def _scan_for_field_value(obj, field: str, expected: str) -> bool:
    if isinstance(obj, dict):
        if str(obj.get(field)) == expected:
            return True
        return any(_scan_for_field_value(v, field, expected) for v in obj.values())
    if isinstance(obj, list):
        return any(_scan_for_field_value(item, field, expected) for item in obj)
    return False
