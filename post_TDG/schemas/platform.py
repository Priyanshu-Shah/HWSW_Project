"""Platform schema: processing elements, interfaces between them, and budgets."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from schemas.common import SCHEMA_VERSION, Ident, Provenance, StrictModel


class PE(StrictModel):
    """A processing element.

    concurrency = exclusive: runs one task at a time (CPU core, fixed-function accelerator).
    concurrency = spatial: each mapped task gets its own hardware and tasks run concurrently, limited by
    `capacity` (FPGA fabric).
    """

    id: Ident
    type: str = Field(description="Matched against Implementation.pe_type.")
    kind: Literal["cpu", "fpga", "gpu", "npu", "dsp", "accelerator"]
    concurrency: Literal["exclusive", "spatial"]
    capacity: dict[str, float] = Field(default_factory=dict, description="e.g. LUT, FF, DSP, BRAM18K, mem_bytes.")
    static_power_mw: float = Field(0.0, ge=0, description="Charged for the whole iteration on every allowed PE.")
    usd: float = Field(0.0, ge=0, description="Cost incurred if any task is mapped to this PE.")
    provenance: Provenance | None = None


class Interface(StrictModel):
    """A link between PEs. Transfer time = setup_us / 1000 + bytes / (bandwidth_mb_s * 1e3) ms."""

    id: Ident
    kind: Literal[
        "shared_memory", "axi_lite", "axi_stream", "axi_hp", "axi_dma", "pcie", "noc", "bus", "other"
    ]
    connects: list[Ident] = Field(min_length=2, description="Any pair of these PEs can communicate over it.")
    setup_us: float = Field(ge=0)
    bandwidth_mb_s: float = Field(gt=0, description="1 MB = 1e6 bytes.")
    energy_nj_per_byte: float = Field(0.0, ge=0)
    shared: bool = Field(True, description="Transfers are serialised (bus contention) when true.")
    provenance: Provenance | None = None

    def transfer_ms(self, nbytes: float) -> float:
        return self.setup_us / 1000.0 + nbytes / (self.bandwidth_mb_s * 1e3)

    def transfer_mj(self, nbytes: float) -> float:
        return nbytes * self.energy_nj_per_byte * 1e-6


class Platform(StrictModel):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    id: Ident
    name: str | None = None
    description: str | None = None
    provenance: Provenance
    pes: list[PE] = Field(min_length=1)
    interfaces: list[Interface] = Field(default_factory=list)
    budgets: dict[str, float] = Field(default_factory=dict, description="Board-level budgets, e.g. usd, power_mw.")

    @model_validator(mode="after")
    def _check(self) -> "Platform":
        pe_ids = [p.id for p in self.pes]
        if len(pe_ids) != len(set(pe_ids)):
            raise ValueError("duplicate PE ids")
        if_ids = [i.id for i in self.interfaces]
        if len(if_ids) != len(set(if_ids)):
            raise ValueError("duplicate interface ids")
        for itf in self.interfaces:
            unknown = set(itf.connects) - set(pe_ids)
            if unknown:
                raise ValueError(f"interface {itf.id} connects unknown PEs {sorted(unknown)}")
        return self

    def pe(self, pe_id: str) -> PE:
        for p in self.pes:
            if p.id == pe_id:
                return p
        raise KeyError(pe_id)

    def interfaces_between(self, a: str, b: str) -> list[Interface]:
        return [i for i in self.interfaces if a in i.connects and b in i.connects]
