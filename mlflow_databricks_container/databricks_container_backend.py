import logging
import textwrap
from shlex import quote as shlex_quote

from mlflow import tracking
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.databricks import (
    DatabricksSubmittedRun,
    DatabricksJobRunner as CoreDatabricksJobRunner,
    _get_tracking_uri_for_run,
    _get_cluster_mlflow_run_cmd,
)
from mlflow.projects.utils import fetch_and_validate_project, get_or_create_run
import mlflow.projects.databricks
import mlflow_databricks_container

_logger = logging.getLogger(__name__)

FF_PROJECTS_BASE = "/srv/texture_embedding"


def databricks_container_backend_builder() -> AbstractBackend:
    return DatabricksContainerBackend()


class DatabricksContainerBackend(AbstractBackend):
    def run(
        self,
        uri,
        entry_point,
        parameters,
        version,
        backend_config,
        tracking_uri,
        experiment_id,
    ):
        _logger.info("Launching MLflow project on Databricks Container backend")
        work_dir = fetch_and_validate_project(uri, version, entry_point, parameters)
        _logger.info(work_dir)
        remote_run = get_or_create_run(
            None, uri, experiment_id, work_dir, version, entry_point, parameters
        )

        run_id = remote_run.info.run_id

        _logger.info(run_id)

        db_job_runner = DatabricksJobRunner(
            databricks_profile_uri=tracking.get_tracking_uri()
        )

        _logger.info(db_job_runner)
        db_run_id = db_job_runner.run_databricks(
            uri,
            entry_point,
            work_dir,
            parameters,
            experiment_id,
            backend_config,
            run_id,
        )
        submitted_run = DatabricksSubmittedRun(db_run_id, run_id, db_job_runner)
        submitted_run._print_description_and_log_tags()
        return submitted_run


class DatabricksJobRunner(CoreDatabricksJobRunner):
    def run_databricks(
        self,
        uri,
        entry_point,
        work_dir,
        parameters,
        experiment_id,
        cluster_spec,
        run_id,
    ):
        tracking_uri = _get_tracking_uri_for_run()
        env_vars = {
            tracking._TRACKING_URI_ENV_VAR: tracking_uri,
            tracking._EXPERIMENT_ID_ENV_VAR: experiment_id,
        }
        _logger.info(
            "=== Running entry point %s of project %s on Databricks ===",
            entry_point,
            uri,
        )
        # Launch run on Databricks
        command = _get_databricks_run_cmd(
            FF_PROJECTS_BASE, run_id, entry_point, parameters
        )
        return self._run_shell_command_job(uri, command, env_vars, cluster_spec)


def _get_databricks_run_cmd(work_dir, run_id, entry_point, parameters):
    """
    Generate MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks.
    """
    # Strip ".gz" and ".tar" file extensions from base filename of the tarfile
    _logger.info("running _get_databricks_run_cmd from databricks_container_backend")

    mlflow_run_arr = _get_cluster_mlflow_run_cmd(
        work_dir, run_id, entry_point, parameters
    )
    mlflow_run_cmd = " ".join([shlex_quote(elem) for elem in mlflow_run_arr])
    shell_command = textwrap.dedent(
        """
    export PATH=$PATH:$DB_HOME/python/bin &&
    mlflow --version &&
    {mlflow_run}
    """.format(
            mlflow_run=mlflow_run_cmd,
        )
    )
    return ["bash", "-c", shell_command]