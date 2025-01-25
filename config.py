from dynaconf import Dynaconf

config = Dynaconf(
    envvar_prefix="KONATAGGER",
    settings_files=["config.toml", ".secrets.toml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
