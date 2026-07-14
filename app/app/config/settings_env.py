# Objective: Import-time database connection defaults read from the environment.
"""Env-derived DB defaults shared by settings_dynamic and its typed-property mixin.

Kept in a leaf module so both the settings facade and the extracted property mixin
can import them without a circular dependency."""

from __future__ import annotations

import os

DB_HOST_ENV = os.getenv("DB_HOST", "mariadb")
DB_USER_ENV = os.getenv("DB_USER", "router_user")
DB_PASS_ENV = os.getenv("DB_PASS", "")
DB_NAME_ENV = os.getenv("DB_NAME", "routerdb")
DB_PORT_ENV = int(os.getenv("DB_PORT", "3306"))
