"""Boards, example graphs, and the validate / cost / bounds operations on a TDG."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from hwsw.conversation import list_example_names, load_example_graph

from hwsw_app.dependencies import get_service
from hwsw_app.pipeline_service import PipelineService
from hwsw_app.schemas import (
    BoardsResponse,
    BoundsOut,
    CostResponse,
    ExamplesResponse,
    TdgRequest,
    ValidateResponse,
)

router = APIRouter(prefix="/api")
Service = Annotated[PipelineService, Depends(get_service)]


@router.get("/boards")
def boards(service: Service) -> BoardsResponse:
    return service.boards()


@router.get("/examples")
def examples() -> ExamplesResponse:
    return ExamplesResponse(names=list_example_names())


@router.get("/examples/{name}")
def example(name: str) -> dict:
    """A ready-made graph-only TDG."""
    try:
        return load_example_graph(name)
    except KeyError as unknown:
        raise HTTPException(status_code=404, detail=str(unknown.args[0])) from unknown


@router.post("/tdg/validate")
def validate(request: TdgRequest, service: Service) -> ValidateResponse:
    return service.validate(request.tdg)


@router.post("/tdg/cost")
def cost(request: TdgRequest, service: Service) -> CostResponse:
    return service.cost(request.tdg)


@router.post("/tdg/bounds")
def bounds(request: TdgRequest, service: Service) -> BoundsOut:
    try:
        return service.bounds(request.tdg)
    except (KeyError, ValueError) as unusable:
        raise HTTPException(status_code=422, detail=f"not a costed TDG: {unusable}") from unusable
