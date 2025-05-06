import os

from typing import Dict

from conductor.client.automator.task_handler import TaskHandler # type: ignore
from conductor.client.configuration.configuration import Configuration # type: ignore
from conductor.client.configuration.settings.authentication_settings import AuthenticationSettings # type: ignore
from conductor.client.worker.worker_task import worker_task #type: ignore
from conductor.client.orkes_clients import OrkesClients #type: ignore
from conductor.client.http.models import StartWorkflowRequest #type: ignore

class ConductorManager:
    api_config = None
    task_handler = None
    workflow_client = None

    def __init__(self):
        self.api_config = Configuration(
            server_api_url=os.getenv("CONDUCTOR_URL",""),
            authentication_settings=AuthenticationSettings(
                key_id=os.getenv("CONDUCTOR_AUTH_KEY",""),
                key_secret=os.getenv("CONDUCTOR_AUTH_SECRET","")
            )
        )

        # Create workflow client for use
        self.workflow_client = OrkesClients(configuration=self.api_config).get_workflow_client()

        # Start the workers
        self.task_handler = TaskHandler(
            workers=[],
            configuration=self.api_config,
            scan_for_annotated_workers=True,
            import_modules=[]
        )

        self.task_handler.start_processes()

    def __del__(self):
        self.task_handler.stop_processes()