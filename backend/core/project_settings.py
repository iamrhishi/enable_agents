"""
Encrypted project-scoped settings - currently just AI provider keys, set by
a project's owner/admin so every AI action taken inside that project uses
the project's own key instead of whoever happens to be using it that day.

Mirrors core/settings.py's UserSettings API (set/get/list/delete) and
reuses its "ai" category definition for UI consistency, but only ever
touches the "ai" category - there's no project-scoped equivalent of
personal connectors/scraping/oauth settings.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from core.crypto import encrypt as _encrypt, decrypt as _decrypt, mask_value as _mask_value
from core.settings import SETTING_DEFINITIONS

logger = logging.getLogger(__name__)

PROJECT_SETTING_KEYS = {"openai_key", "anthropic_key", "preferred_model"}


class ProjectSettings:
    """Encrypted per-project settings storage (AI provider keys only)."""

    def set(self, project_id: str, key: str, value: Any) -> None:
        from core.database import db
        from core.models import ProjectSettingModel

        if key not in PROJECT_SETTING_KEYS:
            raise ValueError(f"Unsupported project setting key: {key}")

        encrypted = _encrypt(json.dumps(value))
        existing = ProjectSettingModel.query.filter_by(project_id=project_id, category="ai", key=key).first()
        if existing:
            existing.value_encrypted = encrypted
            existing.updated_at = datetime.utcnow()
        else:
            db.session.add(ProjectSettingModel(project_id=project_id, category="ai", key=key, value_encrypted=encrypted))
        db.session.commit()
        logger.info(f"Saved project setting ai/{key} for project {project_id}")

    def get(self, project_id: str, key: str, default: Any = None) -> Any:
        from core.models import ProjectSettingModel

        setting = ProjectSettingModel.query.filter_by(project_id=project_id, category="ai", key=key).first()
        if not setting:
            return default
        try:
            return json.loads(_decrypt(setting.value_encrypted))
        except Exception as e:
            logger.error(f"Failed to decrypt project setting ai/{key}: {e}")
            return default

    def list(self, project_id: str, include_values: bool = False) -> Dict[str, Any]:
        from core.models import ProjectSettingModel

        saved = {s.key: s for s in ProjectSettingModel.query.filter_by(project_id=project_id, category="ai").all()}
        ai_def = SETTING_DEFINITIONS["ai"]

        settings_out = {}
        for key, setting_def in ai_def["settings"].items():
            entry = {**setting_def, "configured": key in saved}
            if include_values and key in saved:
                try:
                    value = json.loads(_decrypt(saved[key].value_encrypted))
                    entry["value"] = _mask_value(str(value)) if setting_def.get("type") == "password" else value
                except Exception:
                    entry["value"] = None
            settings_out[key] = entry

        return {"ai": {"label": ai_def["label"], "description": ai_def["description"], "icon": ai_def.get("icon"), "settings": settings_out}}

    def delete(self, project_id: str, key: str) -> bool:
        from core.database import db
        from core.models import ProjectSettingModel

        deleted = ProjectSettingModel.query.filter_by(project_id=project_id, category="ai", key=key).delete()
        db.session.commit()
        if deleted:
            logger.info(f"Deleted project setting ai/{key} for project {project_id}")
        return deleted > 0


def get_project_setting(project_id: str, key: str, default: Any = None) -> Any:
    """Shortcut to get a single project setting."""
    return ProjectSettings().get(project_id, key, default)
