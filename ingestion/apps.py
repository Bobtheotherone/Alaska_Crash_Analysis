from django.apps import AppConfig
import logging


logger = logging.getLogger(__name__)


class IngestionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ingestion"

    def ready(self) -> None:  # type: ignore[override]
        """App initialisation hook.

        We use this to install a small, side-effect-only patch that:

        * Extends :func:`ingestion.validation.load_dataframe_from_bytes` to
          understand Excel files (``.xlsx``, ``.xls``, ``.xlsm``).
        * Updates the ``INGESTION_ALLOWED_EXTENSIONS`` setting at runtime to
          include ``.xlsx`` so the upload endpoint accepts Excel files.

        If anything goes wrong while importing the patch module we log a
        warning but deliberately do *not* prevent Django from starting.
        This keeps CSV / Parquet ingestion working even if Excel support
        cannot be initialised for some reason (for example missing
        ``openpyxl``).
        """
        try:
            from . import excel_support  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Excel ingestion support could not be initialised: %s", exc)
            return

        try:
            excel_support.apply()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Excel ingestion support apply() failed: %s", exc)
