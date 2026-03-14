from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from plugins_market.core.database import get_db
from plugins_market.schemas.common import ResponseModel
from plugins_market.schemas.plugin import PluginPublishForm, PluginPublishResult
from plugins_market.services import PublishError, publish as plugin_publish


plugin_router = APIRouter(prefix="/plugin", tags=["plugin"])


def _parse_form_bool(value: Optional[str]) -> bool:
    if not value:
        return False
    return str(value).strip().lower() in ("true", "1", "on")


def _optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = value.strip() if isinstance(value, str) else str(value).strip()
    return s or None


def _form_value(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    return raw if isinstance(raw, str) else str(raw)


async def get_publish_form(request: Request) -> PluginPublishForm:
    form = await request.form()
    space_id = _form_value(form.get("space_id"))
    if not space_id or not space_id.strip():
        raise HTTPException(status_code=422, detail="space_id is required")
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")
    return PluginPublishForm(
        space_id=space_id.strip(),
        file=file,
        plugin_id=_optional_str(_form_value(form.get("plugin_id"))),
        plugin_version=_optional_str(_form_value(form.get("plugin_version"))),
        version_desc=_optional_str(_form_value(form.get("version_desc"))),
        force=_parse_form_bool(_form_value(form.get("force"))),
    )


@plugin_router.post("", response_model=ResponseModel[PluginPublishResult])
async def publish_plugin(
    form: PluginPublishForm = Depends(get_publish_form),
    db: Session = Depends(get_db),
):
    content = await form.file.read()
    try:
        result = plugin_publish(
            space_id=form.space_id,
            content=content,
            filename=form.file.filename,
            plugin_id=form.plugin_id,
            plugin_version=form.plugin_version,
            version_desc=form.version_desc,
            force=form.force,
            db=db,
        )
    except PublishError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    return ResponseModel(
        code=status.HTTP_200_OK,
        message="Publish plugin successfully",
        data=result,
    )


router = plugin_router
