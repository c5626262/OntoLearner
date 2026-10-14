"""Small immutable extraction contracts; no reference annotations on inputs."""

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from research.llms4ol2026.schema import InferenceDocument


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str

    def __post_init__(self):
        for value in (self.subject, self.predicate, self.object):
            if not isinstance(value, str) or not value.strip() or len(value) > 2000:
                raise ValueError("Triple components must be nonempty strings of at most 2000 characters")

    def as_dict(self):
        return asdict(self)

    def as_tuple(self):
        return self.subject, self.predicate, self.object


class TextBackend(Protocol):
    """An interchangeable local/model/mock implementation; no API dependency."""

    def generate(self, messages: Sequence[dict[str, str]]) -> str: ...


class Postprocessor(Protocol):
    def process(self, document: InferenceDocument, triples: Sequence[Triple]) -> dict: ...
