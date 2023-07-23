"""
Config management for llm-api
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseSettings

logger = logging.getLogger("llm-api.config")


class Settings(BaseSettings):  # pylint: disable=too-few-public-methods
    """
    Settings class to configure the app
    """

    models_dir: str = "./models"
    model_family: str
    model_params: Optional[Dict[str, Any]] = {}
    setup_params: Dict[str, Any] = {}
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    class Config:  # pylint: disable=too-few-public-methods
        """
        Configurations customizations
        """

        env_prefix = "LLM_API_"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            """
            Customization of the config hooks to add a yaml file settings source
            """
            return (
                init_settings,
                env_settings,
                yaml_config_settings_source,
                file_secret_settings,
            )


def yaml_config_settings_source(
    base_settings: BaseSettings,  # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """
    YAML file settings source
    """
    if not os.path.exists("config.yaml"):
        logger.warning("no config file found")
        return {}
    try:
        logger.info("loading config file config.yaml")
        with open("config.yaml", encoding="utf-8") as conf_file:
            data = yaml.load(conf_file, Loader=yaml.FullLoader)
        if data is None:
            logger.warning("config file is empty")
            return {}
        logger.info(str(data))
        return data
    except yaml.YAMLError as exp:
        raise exp


settings = Settings()
