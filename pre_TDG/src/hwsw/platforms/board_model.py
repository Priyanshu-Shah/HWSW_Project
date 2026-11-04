"""Typed view of a board file. The format is defined in library/README.md (library team's)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

# Same keys as the TDG cost map's `resources` and the board's `fpga.capacity`.
RESOURCE_KEYS = ("lut", "ff", "dsp", "bram18k", "uram")


class _BoardSection(BaseModel):
    # Adding optional fields to board files is allowed (library/README.md), so ignore unknown ones.
    model_config = ConfigDict(extra="ignore")


class CpuSpec(_BoardSection):
    name: str
    cores: int
    freq_hz: float
    ops_per_cycle: dict[str, float]  # per core, per data type; one multiply-accumulate = one op
    efficiency: float  # fraction of peak reached on typical compute kernels


class FpgaSpec(_BoardSection):
    part: str
    fclk_hz: float  # clock assumed for HLS designs
    capacity: dict[str, int]  # keys: RESOURCE_KEYS


class DramSpec(_BoardSection):
    bytes: int
    bandwidth_Bps: float  # sustained, shared by CPU and FPGA  # noqa: N815


class InterfaceSpec(_BoardSection):
    bandwidth_Bps: float  # sustained CPU<->FPGA transfer rate  # noqa: N815
    setup_s: float  # fixed latency per transfer


class Board(_BoardSection):
    id: str
    name: str
    summary: str
    cpu: CpuSpec
    fpga: FpgaSpec
    dram: DramSpec
    interfaces: dict[str, InterfaceSpec]
    default_interface: str

    @model_validator(mode="before")
    @classmethod
    def _split_default_interface_name(cls, data: Any) -> Any:
        """In the file, `interfaces.default` is the *name* of one of the other entries."""
        interfaces = dict(data["interfaces"])
        return {**data, "default_interface": interfaces.pop("default"), "interfaces": interfaces}

    @model_validator(mode="after")
    def _default_interface_exists(self) -> "Board":
        if self.default_interface not in self.interfaces:
            raise ValueError(
                f"interfaces.default is '{self.default_interface}' but the board only defines "
                f"{sorted(self.interfaces)}"
            )
        return self

    @property
    def default_interface_spec(self) -> InterfaceSpec:
        return self.interfaces[self.default_interface]
